import numpy as np
import pandas as pd
from pathlib import Path

try:
    import snpy
except:
    snpy = None
import sncosmo
from sncosmo import TimeSeriesSource

###########
# Sources #
########### 
class HsiaoPlusLuSource(TimeSeriesSource):
    """Creates a Hsiao+Lu Source SED.
    """
    def __init__(self):
        if snpy is None:
            raise ImportError('snpy module not found.')
        
        # retrieve Hsiao+Lu SED template
        fluxes = []
        step = 1
        phases = np.arange(-8, 50 + step, step)
        for phase in phases:
            wave, flux = snpy.kcorr.get_SED_H3_plus_L(phase, sBV=1.0)
            wave_mask = wave <= 20_000  # no data beyond H band
            wave, flux = wave[wave_mask], flux[wave_mask]
            fluxes.append(flux)
        fluxes = np.array(fluxes)
        # create the initial source SED
        base = sncosmo.TimeSeriesSource(phase=phases, wave=wave, flux=fluxes)
        super().__init__(base._phase, base._wave, 
                         base._flux(base._phase, base._wave),
                         name='hsiao+lu',
                         zero_before=True)
        
        # add colour-stretch parameter
        self._param_names = self._param_names + ['sBV']
        self.param_names_latex = self.param_names_latex + ['sBV']
        self._parameters = np.array([1., 0.96])
        # reference colour-stretch from the Hsiao template
        self.st_ref = 0.96  # using cspb - cspv9844
        
    def _flux(self, phase, wave):
        st = self._parameters[-1]
        scale = st / self.st_ref
        base_flux = super()._flux(phase / scale, wave)
        return base_flux

class StretchedHsiaoSource(TimeSeriesSource):
    """Creates a Hsiao Source SED that scales the
    phase by the colour-stretch parameter (sBV).
    """
    def __init__(self):
        base = sncosmo.get_source('hsiao')
        super().__init__(base._phase, base._wave, 
                         base._flux(base._phase, base._wave),
                         name='stretched-hsiao',
                         zero_before=True)
        # add colour-stretch parameter
        self._param_names = self._param_names + ['sBV']
        self.param_names_latex = self.param_names_latex + ['sBV']
        self._parameters = np.array([1., 0.96])
        # reference colour-stretch from the Hsiao template
        self.st_ref = 0.96  # using cspb - cspv9844
        
    def _flux(self, phase, wave):
        st = self._parameters[-1]
        scale = st / self.st_ref
        base_flux = super()._flux(phase / scale, wave)
        return base_flux
    
# 91bg templates
# TODO
    
# add sources to sncosmo registry
sncosmo.register(StretchedHsiaoSource(), force=True)
if snpy is not None:
    sncosmo.register(HsiaoPlusLuSource(), force=True)

#########
# Bands #
#########
# add the CSP band with a more common naming
band = sncosmo.get_bandpass("cspu")
sncosmo.register(band, 'csp::u', force=True)
band = sncosmo.get_bandpass("cspb")
sncosmo.register(band, 'csp::b', force=True)
band = sncosmo.get_bandpass("cspv9844")
sncosmo.register(band, 'csp::v', force=True)
band = sncosmo.get_bandpass("cspg")
sncosmo.register(band, 'csp::g', force=True)
band = sncosmo.get_bandpass("cspr")
sncosmo.register(band, 'csp::r', force=True)
band = sncosmo.get_bandpass("cspi")
sncosmo.register(band, 'csp::i', force=True)
band = sncosmo.get_bandpass("cspys")
sncosmo.register(band, 'csp::y', force=True)
band = sncosmo.get_bandpass("cspjs")
sncosmo.register(band, 'csp::j', force=True)
band = sncosmo.get_bandpass("csphs")
sncosmo.register(band, 'csp::h', force=True)

# ATLAS bands
band = sncosmo.get_bandpass("atlasc")
sncosmo.register(band, 'atlas::c', force=True)
band = sncosmo.get_bandpass("atlaso")
sncosmo.register(band, 'atlas::o', force=True)