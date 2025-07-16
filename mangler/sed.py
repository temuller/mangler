import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import sncosmo

plt.rcParams["font.family"] = "P052"
plt.rcParams['mathtext.fontset'] = "cm"

class SED(object):
    """Creates a Spectral Energy Distribution (SED) object from
    an sncosmo source.
    """
    def __init__(self, 
                 source: str, 
                 z: float, 
                 mwebv: float = 0.0, 
                 phase_range: tuple[float, float] = (-10, 90), 
                 bands: list = ['ztf::g', 'ztf::r', 'ztf::i'],
                 mw_dust_law: sncosmo.PropagationEffect = None,
                 **kwargs: dict):
        """
        Parameters
        ----------
        source: sncosmo source.
        z: redshift.
        mwebv: Milky-Way dust extinction.
        phase_range: phase range to be used for the SED model. 
        bands: Bands to use.
        """
        self.source = source
        self.z = z
        self.mwebv = mwebv
        # load model and set parameters
        self.load_model(mw_dust_law)
        params_dict = {"z":z, "mwebv":mwebv} | kwargs
        self.model.set(**params_dict)
        # set bands and plot params
        self.bands = bands
        self._set_wavelength_coverage()
        self.colours = {'ztf::g':"green", 'ztf::r':"red", 'ztf::i':"gold"}
        # time range
        self.phase_range = phase_range
        self.times = np.arange(self.phase_range[0], 
                               self.phase_range[1] + 0.1,
                               0.1
                              )
        self._set_ref_st()
        self.set_st(self.st_ref)
    
    def load_model(self, mw_dust_law: sncosmo.PropagationEffect = None) -> sncosmo.models.Model:
        """Loads the SED model from an sncosmo Source.
        """
        self.model = sncosmo.Model(source=self.source)
        self.rest_model = deepcopy(self.model)  # model @ z=0, without corrections
        # Milky-Way dust law
        if mw_dust_law is None:
            mw_dust_law = sncosmo.CCM89Dust()
        self.model.add_effect(mw_dust_law, 'mw', 'obs')
    
    def _set_wavelength_coverage(self):
        bands_wave = np.empty(0)
        for band in self.bands:
            bands_wave = np.r_[bands_wave, sncosmo.get_bandpass(band).wave]
        self.minwave = bands_wave.min()
        self.maxwave = bands_wave.max()
        
    def _set_ref_st(self):
        """Calculates the colour-stretch of the SED model.
        """
        magB = self.rest_model.bandmag("csp::b", "ab", self.times)
        magV = self.rest_model.bandmag("csp::v", "ab", self.times)
        idmax = np.argmax(np.abs(magB - magV))
        self.st_ref = self.times[idmax] / 30
        
    def set_st(self, st):
        """Updates the sBV parameter of the model, if it uses it.
        """
        self.st = np.copy(st)
        self.scale = self.st / self.st_ref
        if 'sBV' in self.model.param_names:
            idst = self.model.param_names.index('sBV')
            self.rest_model.parameters[idst] = st
            self.model.parameters[idst] = st

    def plot_lightcurves(self, restframe: bool = False, zpsys = 'ab'):
        """Plots the model light curves.
        """
        # chose between observer- and rest-frame model
        if restframe is True:
            model = self.rest_model
            z = 0.0
            label = 'Rest-frame'
        else:
            model = self.model
            z = self.z
            label = 'Observer-frame'
        times = self.times * (1 + z)

        # plot light curves
        fig, ax = plt.subplots(figsize=(6, 4))
        for band in self.bands:
            if band in self.colours:
                colour = self.colours[band]
            else:
                colour = None
            mag = model.bandmag(band, zpsys, times)
            ax.plot(times, mag, label=band, color=colour)
        # config
        plt.gca().invert_yaxis()
        ax.set_xlabel(fr'{label} days since $t_0$', fontsize=16)
        ax.set_ylabel('Apparent Magnitude', fontsize=16)
        ax.set_title(f'"{self.source}" SED source @ z={z}', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_sed(self, obs_phase: float = 0.0, minwave: float = None, maxwave: float = None):
        """Plots the SED model at a given observer-frame phase.
        """
        obs_phase = np.array(obs_phase)
        if minwave is None:
            minwave = self.minwave
        if maxwave is None:
            maxwave = self.maxwave

        # get flux
        rest_wave = np.arange(self.rest_model.minwave(), self.rest_model.maxwave() )
        rest_flux = self.rest_model.flux(obs_phase, rest_wave)
        wave = np.arange(self.model.minwave(), self.model.maxwave())
        flux = self.model.flux(obs_phase, wave)
        # plot SED
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rest_wave, rest_flux, 
                label=fr"Rest-frame (phase$={obs_phase / (1 + self.z):.1f}$)")
        ax.plot(wave, flux, 
                label=fr"Observer-frame (phase$={obs_phase:.1f}$)")
        # plot filters
        ax2 = ax.twinx() 
        for band in self.bands:
            band_wave = sncosmo.get_bandpass(band).wave
            band_trans = sncosmo.get_bandpass(band).trans
            ax2.plot(band_wave, band_trans, color=self.colours[band], alpha=0.4)
        # config
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
        ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.set_xlim(minwave, maxwave)
        ax2.set_ylim(None, 8)
        ax2.set_yticks([])
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_kcorr(self, zp: float = 30, zpsys: str = 'ab'):
        """Plots the same-filter K-correction.
        """
        # plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for band, colour in self.colours.items():
            rest_flux = self.rest_model.bandflux(band, self.times, zp=zp, zpsys=zpsys)
            flux = self.model.bandflux(band, self.times, zp=zp, zpsys=zpsys) 
            kcorr = -2.5 * np.log10(rest_flux / flux)
            ax.plot(self.times, kcorr, label=band, color=colour)
        
        ax.set_xlabel(r'Days since $t_0$', fontsize=16)
        ax.set_ylabel(r'$K$-correction (mag)', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()