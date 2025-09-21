"""Container for storing mixture data."""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .utils import mol2vol

@dataclass
class MixtureData:
    mol_fr: NDArray[np.float64]
    molar_volume: NDArray[np.float64]
    n_electrons: NDArray[np.float64]


    @property
    def vol_fr(self):
        return mol2vol(self.mol_fr, self.molar_volume)
    
    @property
    def vbar(self):
        return self.mol_fr @ self.molar_volume
    
    @property
    def zbar(self):
        return self.mol_fr @ self.n_electrons
    
    @property
    def delta_v(self):
        return self.molar_volume[:-1] - self.molar_volume[-1]
    
    @property
    def delta_z(self):
        return self.n_electrons[:-1] - self.n_electrons[-1]

   