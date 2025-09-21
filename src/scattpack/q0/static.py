"""Calculator for static structure as q => 0."""

import numpy as np
from numpy.typing import NDArray
import pint

from ..mixture_data import MixtureData
from ..constants import r_e, R, k_B, N_A

class Q0StaticCalculator:
        """Calculate static scattering intensities at q₀ from Hessians/energies."""
        def __init__(
                self,
                mixture: MixtureData,
                T: float,
                hessian: NDArray[np.float64] | None = None,
                isothermal_compressiblity: NDArray[np.float64] | None = None,
        ):
            self.mixture = mixture
            self.T = T
            self.hessian = (
                  hessian if hessian is not None 
                  else np.zeros(self.mixture.mol_fr.shape[0])
            )
            self.isothermal_compressiblity = (
                isothermal_compressiblity if isothermal_compressiblity is not None 
                else np.zeros(self.mixture.mol_fr.shape[0])
            )


            # setup unit-registry
            ureg = pint.UnitRegistry()
            self.Q_ = ureg.Quantity
            

        def s0_x(self):
            r"""
            Structure factor as q :math:`\rightarrow` 0 for composition-composition fluctuations.

            Returns
            -------
            np.ndarray
                A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

            Notes
            -----
            The structure factor, :math:`\hat{S}_{ij}^{x}(0)`, is calculated as follows:

            .. math::
                \hat{S}_{ij}^{x}(0) = RT H_{ij}^{-1}

            where:
                - :math:`H_{ij}` is the Hessian of molecules :math:`i,j`
            """
            with np.errstate(divide="ignore", invalid="ignore"):
                return R * self.T / self.hessian
            
        def s0_xp(self) -> NDArray[np.float64]:
            r"""
            Structure factor as q :math:`\rightarrow` 0 for composition-density fluctuations.

            Returns
            -------
            np.ndarray
                2D array of shape ``(n_sys, n_comp-1)``.

            Notes
            -----
            The structure factor, :math:`\hat{S}_{i}^{x\rho}(0)`, is calculated as follows:

            .. math::
                \hat{S}_{i}^{x\rho}(0) = - \sum_{j=1}^{n-1} \left(\frac{V_j - V_n}{\bar{V}}\right) \hat{S}_{ij}^{x}(0)

            where:
                - :math:`V_j` is the molar volume of molecule :math:`j`
                - :math:`\bar{V}` is the molar volume of mixture
            """
            v_ratio = self.mixture.delta_v[np.newaxis, :] / self.mixture.vbar[:, np.newaxis]
            s0_xp_calc = -1 * self.s0_x() * v_ratio[:, :, np.newaxis]
            s0_xp_sum = np.nansum(s0_xp_calc, axis=2)
            return s0_xp_sum
        
        def s0_p(self) -> NDArray[np.float64]:
            r"""
            Structure factor as q :math:`\rightarrow` 0 for density-density fluctuations.

            Returns
            -------
            np.ndarray
                2D array of shape ``(n_sys, n_comp-1)``.

            Notes
            -----
            The structure factor, :math:`\hat{S}^{\rho}(0)`, is calculated as follows:

            .. math::
                \hat{S}^{\rho}(0) = \frac{RT \kappa}{\bar{V}} + \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(\frac{V_i - V_n}{\bar{V}}\right) \left(\frac{V_j - V_n}{\bar{V}}\right) \hat{S}_{ij}^{x}(0)

            where:
                - :math:`V_i` is the molar volume of molecule :math:`i`
                - :math:`\bar{V}` is the molar volume of mixture
                - :math:`\kappa` is the isothermal compressibility
            """
            R_units = float(self.Q_(R, "kJ/mol/K").to("kPa*cm^3/mol/K").magnitude)
            term1 = R_units * self.T * self.isothermal_compressibility / self.mixture.vbar
            v_ratio = self.mixture.delta_v[np.newaxis, :] / self.mixture.vbar[:, np.newaxis]
            term2 = v_ratio[:, :, np.newaxis] * v_ratio[:, np.newaxis, :] * self.s0_x()
            term2_sum = np.nansum(term2, axis=tuple(range(1, term2.ndim)))
            return term1 + term2_sum
        
        def s0_x_e(self) -> NDArray[np.float64]:
            r"""
            Contribution of concentration-concentration structure factor to electron density structure factor.

            Returns
            -------
            np.ndarray
                1D array of shape ``(n_sys)``.

            Notes
            -----
            The contribution of concentration-concentration structure factor to electron density, :math:`\hat{S}^{x,e}(0)`, is calculated as follows:

            .. math::
                \hat{S}^{x,e}(0) = \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(Z_i - Z_n\right) \left(Z_j - Z_n\right) \hat{S}_{ij}^{x}(0)

            where:
                - :math:`Z_i` is the number of electrons in molecule :math:`i`
            """
            s0_x_calc = (
                self.mixture.delta_z[np.newaxis, :, np.newaxis]
                * self.mixture.delta_z[np.newaxis, np.newaxis, :]
                * self.s0_x()
            )
            return np.nansum(s0_x_calc, axis=tuple(range(1, s0_x_calc.ndim)))
        
        def s0_xp_e(self) -> NDArray[np.float64]:
            r"""
            Contribution of concentration-density structure factor to electron density structure factor.

            Returns
            -------
            np.ndarray
                1D array of shape ``(n_sys)``.

            Notes
            -----
            The contribution of concentration-density structure factor to electron density, :math:`\hat{S}^{x\rho,e}(0)`, is calculated as follows:

            .. math::
                \hat{S}^{x\rho,e}(0) = 2 \bar{Z} \sum_{i=1}^{n-1} \left(Z_i - Z_n\right) \hat{S}_{i}^{x\rho}(0)

            where:
                - :math:`Z_i` is the number of electrons in molecule :math:`i`
                - :math:`\bar{Z}` is the number of electrons in the mixture
            """
            s0_xp_calc = self.mixture.delta_z[np.newaxis, :] * self.s0_xp()
            return 2 * self.mixture.zbar * np.nansum(s0_xp_calc, axis=1)
        
        def s0_p_e(self) -> NDArray[np.float64]:
            r"""
            Contribution of density-density structure factor to electron density structure factor.

            Returns
            -------
            np.ndarray
                1D array of shape ``(n_sys)``.

            Notes
            -----
            The contribution of density-density structure factor to electron density, :math:`\hat{S}^{\rho,e}(0)`, is calculated as follows:

            .. math::
                \hat{S}^{\rho,e}(0) = \bar{Z}^2 \hat{S}^{\rho}(0)

            where:
                - :math:`\bar{Z}` is the number of electrons in the mixture
            """
            return self.mixture.zbar**2 * self.s0_p()
        
        def s0_e(self) -> NDArray[np.float64]:
            r"""
            Structure factor of electron density as q :math:`\rightarrow` 0.

            Notes
            -----
            The electron density structure factor, :math:`\hat{S}^e(0)`, is calculated from the sum of the structure factor contributions to electron density.

            .. math::
                \hat{S}^e(0) = \hat{S}^{x,e}(0) + \hat{S}^{x\rho,e}(0) + \hat{S}^{\rho,e}(0)
            """
            return self.s0_x_e() + self.s0_xp_e() + self.s0_p_e()

        def i0(self) -> NDArray[np.float64]:
            r"""
            Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0.

            Returns
            -------
            np.ndarray
                A 1D array with shape ``(n_sys)``

            Notes
            -----
            The scattering intensity at as q :math:`\rightarrow` 0, I(0), is calculated from electron density structure factor (:math:`\hat{S}^e`):

            .. math::
                I(0) = r_e^2 \rho \hat{S}^e(0)

            where:
                - :math:`r_e` is the radius of an electron in cm
            """
            return r_e**2 * (1 / self.mixture.vbar) * N_A * self.s0_e()
