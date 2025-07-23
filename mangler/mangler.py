import yaml
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial

import sncosmo
from astropy.table import Table

import mangler
from mangler.sed import SED
from mangler.photometry import Photometry
from mangler.gp_fit import fit_gp_model, gp_predict

plt.rcParams["font.family"] = "P052"
plt.rcParams['mathtext.fontset'] = "cm"

###############
# Plot Config #
###############
mangler_dir = Path(mangler.__path__[0])
with open(mangler_dir / 'filters.yml', 'r') as file:
    filters_config = yaml.safe_load(file)
    
class SEDMangler(object):
    """Object for building a colour-matched SED model.
    """
    def __init__(self, 
                 data: str | pd.DataFrame | Table,
                 z: float,
                 mwebv: float = 0.0, 
                 source: str = 'stretched-hsiao',
                ):
        """
        Parameters
        ----------
        data: supernova photometry in sncosmo format.
        z: redshift.
        mwebv: Milky-Way dust extinction.
        phase_range: phase range to be used for the SED model. 
        source: SED source model: 'stretched-hsiao' or 'hsiao+lu'.
        """
        self.z = z
        self.mwebv = mwebv
        self.phot = Photometry(data)
        self.bands = np.unique(self.phot.band)
        self.source = source
        self.sed = SED(self.source, z, mwebv, np.unique(self.bands))
        # get the inital colour-stretch of the SED model
        self.st = np.copy(self.sed.st)
        
    def _setup_phase_range(self):
        """Sets up the rest-frame phase range of the object.
        
        Note: this considers both the range of the photometry and 
        SED source model.
        """
        self.phot.phase = (self.phot.time - self.t0) / (1 + self.z)
        # consider limits for the SED model
        min_phase = np.floor(np.max([self.phot.phase.min(),
                                     self.sed.times.min() * self.sed.scale]
                                    )
                             )
        max_phase = np.ceil(np.min([self.phot.phase.max(), 
                                    self.sed.times.max() * self.sed.scale]
                                   )
                            )
        # rest-frame phases
        self.pred_phase = np.arange(min_phase, max_phase + 0.1, 0.1)
        
    def fit_model(self, t0: float = None):
        """Fits the Source SED model to the photometry.
        
        Parameters
        ----------
        t0: initial guess for the reference time of the SED model.
        """
        # select parameters to fit
        params_to_exclude = ['z', 'mwebv', 'mwr_v']
        parameters = [param for param in self.sed.model.param_names 
                      if param not in params_to_exclude]
        # fit model
        if t0 is not None:
            bounds={'t0':(t0 - 5, t0 + 5)}
        else:
            bounds = None
        data = self.phot.data.copy()
        try:
            result, fitted_model = sncosmo.fit_lc(self.phot.data, 
                                                  self.sed.model, 
                                                  parameters,
                                                  bounds=bounds,
                                                  )
        except:
            bands = [band for band in self.bands.copy() if band.endswith('b') | 
                     band.endswith('g') | band.endswith('v')
                     ]
            data_ = data[np.isin(data['band'], bands)]
            result, fitted_model = sncosmo.fit_lc(data_, 
                                                  self.sed.model, 
                                                  parameters,
                                                  bounds=bounds,
                                                  )
                    
        # get t0 and results
        id_t0 = result.param_names.index("t0")
        self.t0 = result.parameters[id_t0]
        self.fitted_model = fitted_model  # for plotting
        self.result = result
        # update colour stretch
        idst = result.param_names.index('sBV')
        self.st = result.parameters[idst]
        self.sed.set_st(self.st)
        self._setup_phase_range()
        
    def plot_sncosmo_fit(self):
        """Plots the sncosmo fit.
        """
        sncosmo.plot_lc(self.phot.data, 
                        model=self.fitted_model, 
                        errors=self.result.errors)

    def mangle_sed(self, k1: str = 'ExpSquared', fit_mean: bool = True, 
                   time_scale: float = None, wave_scale: float = None, t0: float = None):
        """Modifies the SED model to match the observations using Gaussian Process (GP) 
        regression.

        Parameters
        ----------
        k1: GP kernel for the time axis.
        fit_mean: whether to fit a mean function (constant).
        t0: initial guess for the reference time of the SED model.
        """   
        # initial fit; this sets t0 and sBV
        self.fit_model(t0=t0)
        # get phase range to use 
        self.phase_mask = ((self.pred_phase.min() <= self.phot.phase) & 
                            (self.phot.phase <=  self.pred_phase.max()
                                )
                            )
        # flux ratios between observations and (observer-frame) SED model
        model_flux = self.sed.model.bandflux(self.phot.band, 
                                            self.phot.phase * (1 + self.z),
                                            zp=self.phot.zp, 
                                            zpsys=self.phot.zpsys)
        # sometimes the model gives zero flux
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ratio_flux = self.phot.flux / model_flux
            self.ratio_error = self.phot.flux_err / model_flux
        # fit mangling surface in observer frame (i.e. input in observer frame)
        # for phases in rest-frame: give phases in observer frame
        # for wavelength in rest-frame: give wavelength in rest-frame
        self.gp_model = fit_gp_model(self.phot.phase[self.phase_mask] * (1 + self.z), 
                                     self.phot.eff_wave[self.phase_mask], 
                                     self.ratio_flux[self.phase_mask], 
                                     self.ratio_error[self.phase_mask], 
                                     k1=k1, fit_mean=fit_mean, 
                                     time_scale=time_scale,wave_scale=wave_scale
                                     )
        # store the model for later predictions (easy use when needed)
        self.gp_predict = partial(gp_predict, 
                                  ratio_pred=self.ratio_flux[self.phase_mask],
                                  error_pred=self.ratio_error[self.phase_mask],
                                  gp_model=self.gp_model,
                                  )
        # get colour-stretch
        self.st, self.st_err = self.calculate_colour("csp::b", "csp::v", 
                                                     return_st=True, plot=False)
        
    def _compute_colour_curve(self, fluxes: np.ndarray, cov: np.ndarray, 
                        zp1: float | np.ndarray, zp2: float | np.ndarray):
        """Computes the colour curve from concatenated flux arrays from two bands.

        Note: flux = [flux1_0, flux1_1, ...flux1_N, # band1
                      flux2_N+1, flux2_N+2, ...flux2_2N]  # band2
        Parameters
        ----------
        flux: Flux from two bands.
        cov: Covariance from two bands.
        zp1: Zero point of first band.
        zp2: Zero point of second band.
        """
        N = fluxes.shape[0] // 2
        f1 = fluxes[:N]
        f2 = fluxes[N:]
        # variance and covariance
        cov_11 = cov[:N, :N]
        cov_22 = cov[N:, N:]
        cov_12 = cov[:N, N:]
        
        # error propagation
        prefactor = 2.5 / np.log(10)
        var_colour = (
            (np.diag(cov_11) / (f1 ** 2)) +  # variance
            (np.diag(cov_22) / (f2 ** 2)) -  # variance
            2 * np.diag(cov_12) / (f1 * f2)  # covariance
        ) * (prefactor ** 2)

        colour = -2.5 * np.log10(f1 / f2) + (zp1 - zp2)
        colour_err = np.sqrt(var_colour)
        self.colour, self.colour_err = colour, colour_err
        
    def calculate_colour(self, band1: str, band2: str, zp: float = 30, zpsys: str = 'ab', 
                         return_st: bool = False, plot: bool = True):
        """Calculates rest-frame colour using the colour-matched SED.

        Note: Colour = band1 - band2
        
        Parameters
        ----------
        band1: First band.
        band2: Second band.
        zp: Zeropoint for both bands. Only used if the photometry does not 
            include any of the given bands.
        zpsys: Magnitude system for both bands. Only used if the photometry
            does not  include any of the given bands.
        set_st: Whether to return the colour-stretch parameter.
        plot: Whether to plot the colour curve.

        Results
        -------
        colour: Colour curve.
        colour_err: Uncertainty.
        """
        # phase range to use
        pred_rest_phase = self.pred_phase.copy()
        pred_obs_phase = pred_rest_phase * (1 + self.z)

        if (band1 not in self.phot.band) | (band2 not in self.phot.band):
            eff_wave1 = sncosmo.get_bandpass(band1).wave_eff
            eff_wave2 = sncosmo.get_bandpass(band2).wave_eff
            zp1 = zp2 = zp
            zpsys1 = zpsys2 = zpsys
        else:
            # create band and phase mask
            band_mask1 = self.phot.band == band1
            band_mask2 = self.phot.band == band2
            mask1 = self.phase_mask & band_mask1
            mask2 = self.phase_mask & band_mask2
            # apply mask
            eff_wave1 = self.phot.eff_wave[mask1][0]
            eff_wave2 = self.phot.eff_wave[mask2][0]
            zp1 = self.phot.zp[mask1][0]
            zp2 = self.phot.zp[mask2][0]
            zpsys1 = self.phot.zpsys[mask1][0]
            zpsys2 = self.phot.zpsys[mask2][0]
        
        # wavelength array
        pred_rest_wave = np.array([eff_wave1 * (1 + self.z)] * len(pred_obs_phase) + 
                                  [eff_wave2 * (1 + self.z)] * len(pred_obs_phase) 
                                  )
        # flux array
        rest_model_flux1 = self.sed.rest_model.bandflux(band1, 
                                                        pred_rest_phase, 
                                                        zp=zp1, 
                                                        zpsys=zpsys1)
        rest_model_flux2 = self.sed.rest_model.bandflux(band2, 
                                                        pred_rest_phase, 
                                                        zp=zp2, 
                                                        zpsys=zpsys2)
        rest_model_flux = np.r_[rest_model_flux1, rest_model_flux2]
        # mangle rest-frame SED
        pred_obs_phase_ = np.r_[pred_obs_phase, pred_obs_phase]
        ratio_fit, cov_fit = self.gp_predict(pred_obs_phase_, pred_rest_wave, return_cov=True)
        self.ratio_fit, self.cov_fit = ratio_fit, cov_fit
        rest_mangled_flux = rest_model_flux * ratio_fit
        rest_mangled_cov = np.outer(rest_model_flux, rest_model_flux) * cov_fit
        self.colour_flux_ratio = rest_mangled_flux
        self.colour_flux_cov = rest_mangled_cov
        
        # computes colour curve and stores it to late calculate colour stretch
        self._compute_colour_curve(rest_mangled_flux, rest_mangled_cov, zp1, zp2)
        if plot is True:
            fig, ax = plt.subplots()
            ax.plot(pred_rest_phase, self.colour)
            ax.fill_between(pred_rest_phase, 
                            self.colour - self.colour_err, 
                            self.colour + self.colour_err, 
                            alpha=0.2)
            ax.set_ylabel(fr'$({band1} - {band2})$ (mag)', fontsize=16)
            ax.set_xlabel(r'Rest-frame days since $t_0$', fontsize=16)
            ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
            ax.tick_params('both', labelsize=14)
            plt.show()
            
        if return_st is True:
            return  self._compute_colour_stretch()

    def _compute_colour_stretch(self):
        """Computes the colour stretch parameter for the bands used 
        in the colour curve calculation.
        """
        idmax = np.argmax(self.colour)
        st_phase = self.pred_phase[idmax]

        # monte-carlo sampling to estimate the standard deviation
        # use a constrained phase range to speed up things
        mask = (st_phase - 9 < self.pred_phase) & (self.pred_phase < st_phase + 9)  
        pred_phase = self.pred_phase[mask]
        length = len(self.colour[mask])
        colours = np.random.normal(self.colour[mask], 
                                self.colour_err[mask], 
                                size=(10000, length)
                                )
        # calculate the standard deviation
        st_list = []
        for colour in colours:
            st_idx = np.argmax(colour)
            st_list.append(pred_phase[st_idx])
        # divided by 30 assuming sBV
        st, st_err = st_phase / 30, np.std(st_list) / 30
        return st, st_err
        
    def plot_fit(self, plot_mag: bool = False):
        """Plots the light-curve fit and ratio between the observations and SED.
        
        Note: epochs with magnitude errors above 2 mag are not plotted in
        the top panel.
        
        Parameters
        ----------
        plot_mag: If True, plots magnitudes instead of flux.
        """
        # phase range to use
        pred_rest_phase = self.pred_phase.copy()
        pred_obs_phase = pred_rest_phase * (1 + self.z)
        
        fig, ax = plt.subplots(2, 1, height_ratios=(3, 1), gridspec_kw={"hspace":0.05})
        for band in self.bands:
            # select band and phase range to use
            band_mask = self.phot.band == band
            mask = self.phase_mask & band_mask
            # apply mask
            time = self.phot.time[mask]
            flux, flux_err = self.phot.flux[mask], self.phot.flux_err[mask]
            zp, zpsys = self.phot.zp[mask], self.phot.zpsys[mask]
            eff_wave = sncosmo.get_bandpass(band).wave_eff
            # flux ratios for plotting
            ratio_flux = self.ratio_flux[mask]
            ratio_error = self.ratio_error[mask]
            
            ########################
            # observer-frame model #
            ########################            
            pred_obs_wave = np.array([eff_wave] * len(pred_obs_phase))
            obs_ratio_fit, obs_var_fit = self.gp_predict(pred_obs_phase, 
                                                         pred_obs_wave)
            obs_std_fit = np.sqrt(obs_var_fit)
            # mangle light curves
            obs_model_flux = self.sed.model.bandflux(band, 
                                                     pred_obs_phase, 
                                                     zp=zp[0], 
                                                     zpsys=zpsys[0])
            obs_mangled_flux = obs_model_flux * obs_ratio_fit
            obs_mangled_error = obs_model_flux * obs_std_fit

            ########
            # Plot #
            ########
            if band in filters_config.keys():
                colour = filters_config[band]['colour']
                marker = filters_config[band]['marker']
            else:
                colour = marker = None
                
            if plot_mag is False:
                # data
                ax[0].errorbar(time, flux, flux_err, 
                            ls="", marker=marker, color=colour, label=band)
                # model
                ax[0].plot(pred_obs_phase + self.t0, obs_mangled_flux, color=colour)
                ax[0].fill_between(pred_obs_phase + self.t0, 
                                obs_mangled_flux - obs_mangled_error, 
                                obs_mangled_flux + obs_mangled_error, 
                                alpha=0.2,
                                color=colour)
            else:
                # flux to mag
                mag = -2.5 * np.log10(flux) + zp[0]
                mag_err = 1.0857 * (flux_err / flux)
                phot_mask = mag_err < 2
                obs_mangled_mag = -2.5 * np.log10(obs_mangled_flux) + zp[0]
                obs_mangled_magerr = 1.0857 * (obs_mangled_error / obs_mangled_flux)
                # data
                ax[0].errorbar(time[phot_mask], mag[phot_mask], mag_err[phot_mask], 
                               ls="", marker=marker, color=colour, label=band)
                # model
                ax[0].plot(pred_obs_phase + self.t0, obs_mangled_mag, color=colour)
                ax[0].fill_between(pred_obs_phase + self.t0, 
                                   obs_mangled_mag - obs_mangled_magerr, 
                                   obs_mangled_mag + obs_mangled_magerr, 
                                   alpha=0.2,
                                   color=colour)
        
            # residuals
            norm = np.average(ratio_flux, weights=1 / ratio_error ** 2)  # for plotting only
            # ratio
            ax[1].errorbar(time, ratio_flux / norm, ratio_error / norm, 
                           ls="", marker=marker, color=colour)
            # fit
            ax[1].plot(pred_obs_phase + self.t0, obs_ratio_fit / norm, color=colour)
            ax[1].fill_between(pred_obs_phase + self.t0, 
                               (obs_ratio_fit - obs_std_fit) / norm, 
                               (obs_ratio_fit + obs_std_fit) / norm, 
                               alpha=0.2, color=colour)
            
        # config
        if plot_mag is True:
            ax[0].invert_yaxis()
            ax[0].set_ylabel(f'{zpsys[0].upper()} Magnitude', fontsize=16)
        else:
            ax[0].set_ylabel(r'$F_{\lambda}$', fontsize=16)
        ax[1].set_xlabel(r'Observer-frame time', fontsize=16)
        ax[1].set_ylabel(r'$F_{\lambda}^{\rm data} / F_{\lambda}^{\rm SED}$', 
                            fontsize=16)
        ax[0].set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        for i in range(2):
            ax[i].tick_params('both', labelsize=14)
        ax[0].set_xticklabels([])
        ax[0].legend(fontsize=14, framealpha=0)
        plt.show()
        
    def plot_lightcurves(self, plot_mag: bool = False):
        """Plots the rest-frame light-curves from the mangled SED.
        
        Parameters
        ----------
        plot_mag: If True, plots magnitudes instead of flux.
        """
        # phase range to use
        pred_rest_phase = self.pred_phase.copy()
        pred_obs_phase = pred_rest_phase * (1 + self.z)
        
        fig, ax = plt.subplots()
        for band in self.bands:
            # select band and phase range to use
            band_mask = self.phot.band == band
            mask = self.phase_mask & band_mask
            # apply mask
            zp, zpsys = self.phot.zp[mask], self.phot.zpsys[mask]
            eff_wave = sncosmo.get_bandpass(band).wave_eff

            ####################
            # rest-frame model #
            ####################
            pred_rest_wave = np.array([eff_wave * (1 + self.z)] * len(pred_rest_phase))
            rest_ratio_fit, rest_var_fit = self.gp_predict(pred_obs_phase, pred_rest_wave)
            rest_std_fit = np.sqrt(rest_var_fit)
            # mangle SED
            rest_model_flux = self.sed.rest_model.bandflux(band, 
                                                           pred_rest_phase, 
                                                           zp=zp[0], 
                                                           zpsys=zpsys[0])
            rest_mangled_flux = rest_model_flux * rest_ratio_fit
            rest_mangled_error = rest_model_flux * rest_std_fit

            ########
            # Plot #
            ########
            if band in filters_config.keys():
                colour = filters_config[band]['colour']
            else:
                colour = None
                
            if plot_mag is False:
                # model
                ax.plot(pred_rest_phase, rest_mangled_flux, color=colour, label=band)
                ax.fill_between(pred_rest_phase, 
                                rest_mangled_flux - rest_mangled_error, 
                                rest_mangled_flux + rest_mangled_error, 
                                alpha=0.2,
                                color=colour)
            else:
                # flux to mag
                rest_mangled_mag = -2.5 * np.log10(rest_mangled_flux) + zp[0]
                rest_mangled_magerr = 1.0857 * (rest_mangled_error / rest_mangled_flux)
                # model
                ax.plot(pred_rest_phase, rest_mangled_mag, color=colour, label=band)
                ax.fill_between(pred_rest_phase, 
                                rest_mangled_mag - rest_mangled_magerr, 
                                rest_mangled_mag + rest_mangled_magerr, 
                                alpha=0.2,
                                color=colour)
                
        # config
        if plot_mag is True:
            ax.invert_yaxis()
            ax.set_ylabel(f'{zpsys[0].upper()} Magnitude', fontsize=16)
        else:
            ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
        ax.set_xlabel(r'Rest-frame days since $t_0$', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14, framealpha=0)
        plt.show()
        
    def plot_sed_surface(self, minphase: float = None, maxphase: float = None, 
                         minwave: float = None, maxwave: float = None, 
                         delta_phase: float = 2, delta_wave: float = 30, 
                         alpha: float = 1.):
        """Plots the mangled SED in the observer frame.
        
        Note: if running on jupyter lab, it is recommended to run with ipympl for an
        interactive plot (ipympl package required)
        >>> %matplotlib ipympl
        >>> self.plot_sed_surface()

        Parameters
        ----------
        minphase: Minimum observer-frame phase to plot.
        maxphase: Maximum observer-frame phase to plot.
        minwave: Minimum observer-frame wavelength to plot.
        maxwave: Maximum observer-frame wavelength to plot.
        delta_phase: Step in the phase axis.
        delta_wave: Step in the wavelength axis.
        alpha: Transparency of the surface.
        """
        if minphase is None:
            minphase = self.sed.model.mintime()
        if maxphase is None:
            maxphase = self.sed.model.maxtime()
        if minwave is None:
            minwave = self.sed.minwave * (1 + self.z)
        if maxwave is None:
            maxwave = self.sed.maxwave * (1 + self.z)

        obs_phases = np.arange(minphase, maxphase + delta_phase, delta_phase)
        obs_waves = np.arange(minwave, maxwave + delta_wave, delta_wave).astype(float)

        FLUXES = self.sed.model.flux(obs_phases, obs_waves)
        PHASES, WAVES = np.meshgrid(obs_phases, obs_waves)

        # mangle the SED
        mang_list = []
        for phase in obs_phases:
            pred_phases = np.array([phase] * len(obs_waves))
            mang_epoch, _ = self.gp_predict(pred_phases, obs_waves)
            mang_list.append(mang_epoch)
        mangling_surface = np.array(mang_list)
        FLUXES *=  mangling_surface  # mangling

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # plot the 3D surface
        ax.plot_surface(PHASES, WAVES, FLUXES.T, edgecolor='royalblue', 
                        lw=0.5, rstride=8, cstride=8, alpha=alpha)

        """
        # scale observations to match observer-frame SED model
        wave_mask = self.phot.eff_wave==self.phot.eff_wave[0]
        phot_obs_phase = self.phot.phase[wave_mask] * (1 + self.z)
        phot_obs_flux = self.phot.flux[wave_mask]
        obs_eff_wave = self.phot.eff_wave[0] #* (1 + self.z)
        idwave = np.argmin(np.abs(obs_waves - obs_eff_wave))
        flux_band = np.interp(phot_obs_phase, obs_phases, FLUXES.T[idwave])
        scale = np.mean(flux_band / phot_obs_flux)
        # plot observations
        ax.scatter(self.phot.phase * (1 + self.z), 
                   self.phot.eff_wave * (1 + self.z), 
                   self.phot.flux * scale, 
                   s=40, c='k')
        """

        # Plot projections on the 'walls' of the graph.
        ax.contour(PHASES, WAVES, FLUXES.T, zdir='x', offset=obs_phases.min(), cmap='viridis_r')
        ax.contour(PHASES, WAVES, FLUXES.T, zdir='y', offset=obs_waves.max(), cmap='coolwarm')

        ax.set(xlim=(obs_phases.min(), obs_phases.max()), ylim=(obs_waves.min(), obs_waves.max()),
            xlabel='Observer-frame time', ylabel=r'Observer-frame wavelength ($\AA$)', zlabel=r'$F_{\lambda}$')
        plt.tight_layout()
        plt.show()
        
    def plot_magling_surface(self, minphase: float = None, maxphase: float = None, 
                             minwave: float = None, maxwave: float = None, 
                             delta_phase: float = 2, delta_wave: float = 30, 
                             alpha: float = 0.5):
        """Plots the mangling surface in the observer frame.
        
        Note: if running on jupyter lab, it is recommended to run with ipympl for an
        interactive plot (ipympl package required)
        >>> %matplotlib ipympl
        >>> self.plot_magling_surface()

        Parameters
        ----------
        minphase: Minimum observer-frame phase to plot.
        maxphase: Maximum observer-frame phase to plot.
        minwave: Minimum observer-frame wavelength to plot.
        maxwave: Maximum observer-frame wavelength to plot.
        delta_phase: Step in the phase axis.
        delta_wave: Step in the wavelength axis.
        alpha: Transparency of the surface.
        """
        if minphase is None:
            minphase = self.sed.model.mintime()
        if maxphase is None:
            maxphase = self.sed.model.maxtime()
        if minwave is None:
            minwave = self.sed.minwave * (1 + self.z)
        if maxwave is None:
            maxwave = self.sed.maxwave * (1 + self.z)

        obs_phases = np.arange(minphase, maxphase + delta_phase, delta_phase)
        obs_waves = np.arange(minwave, maxwave + delta_wave, delta_wave).astype(float)
        PHASES, WAVES = np.meshgrid(obs_phases, obs_waves)

        # mangle the SED
        mang_list = []
        for phase in obs_phases:
            pred_phases = np.array([phase] * len(obs_waves))
            mang_epoch, _ = self.gp_predict(pred_phases, obs_waves)
            mang_list.append(mang_epoch)
        MANGLING_SURFACE = np.array(mang_list)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot the 3D surface
        ax.plot_surface(PHASES, WAVES, MANGLING_SURFACE.T, edgecolor='royalblue', 
                        lw=0.5, rstride=8, cstride=8, alpha=alpha)
        # plot effective wavelength of the bands at "rest frame"
        for eff_wave in np.unique(self.phot.eff_wave):
            rest_waves = np.array([eff_wave * (1 + self.z)] * len(obs_phases))
            idwave = np.argmin(np.abs(obs_waves - eff_wave))
            mangling_band = MANGLING_SURFACE.T[idwave]
            idband = np.argmin(np.abs(self.phot.eff_wave - eff_wave))
            # get band to plot with colour
            band = self.phot.band[idband]
            if band in filters_config.keys():
                colour = filters_config[band]['colour']
            else:
                colour = None
            ax.plot(obs_phases, rest_waves, mangling_band, color=colour)

        # plot flux ratios (data / SED)
        for band in self.bands:
            if band in filters_config.keys():
                colour = filters_config[band]['colour']
            else:
                colour = None
            # create mask
            phot_phase = self.phot.phase * (1 + self.z)
            phase_mask = (minphase <= phot_phase) & (phot_phase <= maxphase)
            phot_wave = self.phot.eff_wave
            wave_mask = (minwave <= phot_wave) & (phot_wave <= maxwave)
            mask = (self.phase_mask) & (self.phot.band==band) & phase_mask& wave_mask
            # apply mask
            phot_phase = phot_phase[mask]
            phot_wave = phot_wave[mask]
            ratio_flux = self.ratio_flux[mask]
            ratio_error = self.ratio_error[mask]
            # plot
            ax.errorbar(phot_phase, phot_wave, ratio_flux, ratio_error,
                        ls='', marker='o', color=colour, label=band)

        # Plot projections on the 'walls' of the graph.
        ax.contour(PHASES, WAVES, MANGLING_SURFACE.T, zdir='x', offset=obs_phases.min(), cmap='viridis_r')
        ax.contour(PHASES, WAVES, MANGLING_SURFACE.T, zdir='y', offset=obs_waves.max(), cmap='coolwarm')
        ax.legend(fontsize=14, framealpha=0)
        ax.set(xlim=(obs_phases.min(), obs_phases.max()), ylim=(obs_waves.min(), obs_waves.max()),
            xlabel='Observer-frame time', ylabel=r'Observer-frame wavelength ($\AA$)', zlabel='Mangling Surface (data/model)')
        plt.tight_layout()
        plt.show()