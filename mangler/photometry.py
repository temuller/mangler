import numpy as np
import pandas as pd

import sncosmo
from astropy.table import Table
from sncosmo.utils import alias_map
from sncosmo.photdata import PHOTDATA_ALIASES, PHOTDATA_REQUIRED_ALIASES

class Photometry(object):
    """Creates a photometry object to handle observations.
    """
    def __init__(self, data: str | pd.DataFrame | Table):
        """
        Parameters
        ----------
        data: supernova photometry.
        """
        if isinstance(data, str):
            data = pd.read_csv(data)
        elif isinstance(data, Table):
            data = data.to_pandas()
        self.data = Table.from_pandas(data)
        
        # taken from sncosmo
        mapping = alias_map(data.columns, 
                            PHOTDATA_ALIASES,
                            required=PHOTDATA_REQUIRED_ALIASES)

        self.time = np.asarray(data[mapping['time']])
        self.band = data[mapping['band']].values
        self.eff_wave = np.array([sncosmo.get_bandpass(band).wave_eff 
                                  for band in self.band])

        self.flux = np.asarray(data[mapping['flux']])
        self.flux_err = np.asarray(data[mapping['fluxerr']])
        self.zp = np.asarray(data[mapping['zp']])
        self.zpsys = np.asarray(data[mapping['zpsys']])