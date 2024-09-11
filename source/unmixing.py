import numpy as np 
from scipy.optimize import lsq_linear, nnls

import fluorescentneurosurgery.data
from importlib_resources import files
import scipy.io as sio

BASIS_SPECS = sio.loadmat(files(fluorescentneurosurgery.data).joinpath('basisSpectra.mat')) #data is {String variable_name: Numpy Array matrix}

PPIX_634 = BASIS_SPECS['bF'] # this is the PpIX to use as the pure PpIX (confirm this)
PPIX_620 = BASIS_SPECS['bGS']
FLAVO = BASIS_SPECS['flavo']
LIPO = BASIS_SPECS['lipo']
NADH = BASIS_SPECS['nadh']
BASES = {cp: BASIS_SPECS[cp] for cp in {'bF', 'bGS', 'flavo', 'lipo', 'nadh'}}

def unmixing(
    uvdata: np.ndarray, 
    bases: dict,
):
    """ unmix the uvdata with the constraint that all relative concentrations are non-negative 

    Args:
        uvdata (np.ndarray): 
            UV spectra. The shape is (spectrum size, num_samples )
        bases (dict{str: np.ndarray}): 
            bases with their corresponding names 
        
    Returns:
        relative concentrations (np.ndarray) of size (num_bases, num_samples) returned with concentrations in the same order of bases 
        and the ordering (keys) 

        i.e. for each element in unmixed it has ordering keys 
    """  
    keys = bases.keys()
    values = bases.values()
    B = np.column_stack(tuple(values))

    lb = np.zeros(len(keys))
    ub = np.ones(len(keys))*np.inf

    unmixed = [lsq_linear(B,uvdata[:,i], bounds = (lb,ub)).x for i in range(uvdata.shape[1])]
    
    return unmixed, keys


class ConstrainedSpectralUnmixing:
    """ 
    Class for converting unmixing spectra with an initialized set of spectra.
    """
    def __init__(self, specs=('bF', 'bGS', 'flavo', 'lipo', 'nadh')):
        self.DEFAULT_SPECS_MAT = np.column_stack(tuple(BASIS_SPECS[cp] for cp in specs))
        self.specs = specs

    def unmix(self, spectra):
        """ unmix the spectra with the constraint that all relative concentrations are non-negative 

        Args:
            spectra (np.ndarray): 
                UV spectra. The shape is (num_samples, n_spectra)
            
        Returns:
            relative concentrations (np.ndarray) of size (num_bases, num_samples) returned with concentrations in the same order of bases 
            and the ordering (keys) 

            i.e. for each element in unmixed it has ordering keys 
        """  
        B = self.DEFAULT_SPECS_MAT
        def nnls_map(spec):
            return nnls(B, spec)[0] # ignore residuals return contributions

        unmixed = np.apply_along_axis(nnls_map, axis=0, arr=spectra.T)
        
        return unmixed

