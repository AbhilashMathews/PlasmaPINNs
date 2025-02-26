"""Physical constants and normalization parameters for plasma simulation."""

import numpy as np
from typing import Dict

# Fundamental physical constants
ATOMIC_MASS_UNIT = 1.660539040e-27     # kg
ELECTRON_MASS = 9.10938356e-30         # kg
ELECTRON_CHARGE = 1.60217662e-19       # C
SPEED_OF_LIGHT = 299792458.0           # m/s
BOLTZMANN_CONST = 1.380649e-23         # J/K

# Geometry
MINOR_RADIUS = 0.22             # meters
MAJOR_RADIUS = 0.68             # meters
B_FIELD = 5.0              # Tesla (B0)

# Plasma composition
ION_CHARGE = 1                         # Ionization level (Z)
MASS_RATIO = 3672.3036                 # mi/me
ION_MASS_NUMBER = 2.0                  # m_i/m_proton (mu)
PROTON_MASS = 1.007276 * ATOMIC_MASS_UNIT  # kg
ION_MASS = ION_MASS_NUMBER * PROTON_MASS   # kg

# Initial conditions
ELECTRON_TEMP = 25.0                   # eV (Te0)
ION_TEMP = 25.0                        # eV (Ti0)
PLASMA_DENSITY = 5e19                  # m^-3 (n0)

# Magnetic field
def compute_magnetic_field(radius: float) -> float:
    """Compute magnetic field strength at given radius."""
    center_field = (B_FIELD * MAJOR_RADIUS)/(MAJOR_RADIUS + MINOR_RADIUS)
    return center_field

# Derived plasma parameters
ELECTRON_THERMAL_SPEED = np.sqrt(ELECTRON_CHARGE * ELECTRON_TEMP / ION_MASS)  # m/s (previously cse0)
ION_THERMAL_SPEED = np.sqrt(ELECTRON_CHARGE * ION_TEMP / ION_MASS)           # m/s (previously csi0)

# Time and space normalization
REFERENCE_TIME = np.sqrt((MAJOR_RADIUS * MINOR_RADIUS)/2.0)/ELECTRON_THERMAL_SPEED  # previously tRef
SPACE_FACTOR = 100.0 * MINOR_RADIUS    # cm conversion

# Plasma parameters
ETA = 63.6094                      # Plasma resistivity
TAU_T = 1.0                        # Temperature ratio (Ti/Te)

# Dimensionless parameters
EPS_R = 0.4889                     # Resistivity parameter
EPS_V = 0.3496                     # Viscosity parameter
ALPHA_D = 0.0012                   # Drift parameter
KAPPA_E = 7.6771                   # Electron thermal conductivity
KAPPA_I = 0.2184                   # Ion thermal conductivity
EPS_G = 0.0550                     # Density diffusion coefficient
EPS_GE = 0.0005                    # Temperature diffusion coefficient

# Source terms
N_SRC_A = 20.0                     # Amplitude of density source
ENER_SRC_A = 0.001                 # Amplitude of energy source
X_SRC = -0.15                      # Source location
SIG_SRC = 0.01                     # Source width

# Normalization
DIFF_X_NORM = 50.0                 # Spatial normalization for x-direction
DIFF_Y_NORM = 50.0                 # Spatial normalization for y-direction