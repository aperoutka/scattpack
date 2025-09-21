"""
Physical constants for scattering and thermodynamics.
Values are provided in both SI (J) and kJ-based units.
"""

# --- Boltzmann constant ---
BOLTZMANN_K_J = 1.380649e-23        # J/K
BOLTZMANN_K_KJ = BOLTZMANN_K_J / 1000.0  # kJ/K

# --- Gas constant ---
GAS_CONSTANT_R_J = 8.314462618      # J/(mol·K)
GAS_CONSTANT_R_KJ = GAS_CONSTANT_R_J / 1000.0  # kJ/(mol·K)

# --- Classical electron radius ---
CLASSICAL_E_RADIUS = 2.8179403262e-13  # cm

# --- Avogadro's number ---
AVOGADRO_N_A = 6.02214076e23 # molec/mol

# Aliases for readability
k_B = BOLTZMANN_K_KJ   # kJ/K
R = GAS_CONSTANT_R_KJ  # kJ/(mol·K)
r_e = CLASSICAL_E_RADIUS # cm
N_A = AVOGADRO_N_A # molec / mol