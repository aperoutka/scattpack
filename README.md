# ScattPack: Python toolkit for scattering and structure factors

ScattPack is a contributor‑friendly, extensible library for calculating **scattering intensities** and **structure factors**.  

## Installation

```bash
git clone https://github.com/aperoutka/scattpack.git
cd scattpack
pip install .

## Example

```python
import scattpack as sp

# initialize mixture object from mole fractions, electron numbers, and molar volumes (cm^3/mol)
mixture = sp.MixtureData(mol_fr=mol_fr, n_electrons=n_electrons, molar_volume=molar_volume)
# create structure calculator at a given temperature from hessian (of Gibbs mixing free energy) and isothermal compressibility
structure_calc = sp.q0.Q0StaticCalculator(T=T, hessian=hessian, isothermal_compressibility=isothermal_compressibility)

# calculate X-ray intensity at q->0 
i0 = structure_calc.i0()
```
