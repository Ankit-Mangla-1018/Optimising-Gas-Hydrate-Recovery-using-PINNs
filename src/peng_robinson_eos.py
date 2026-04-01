"""
peng_robinson_eos.py
--------------------
Peng-Robinson Equation of State (PR EOS) for real-gas fugacity calculation.

Reference:
    Peng, D.-Y. & Robinson, D.B. (1976). A new two-constant equation of state.
    Industrial & Engineering Chemistry Fundamentals, 15(1), 59-64.
"""

import numpy as np


R = 8.314  # J/(mol·K)


def pr_parameters(T: float, Tc: float, Pc: float, omega: float):
    """
    Compute PR EOS a(T), b parameters.

    Parameters
    ----------
    T     : Temperature [K]
    Tc    : Critical temperature [K]
    Pc    : Critical pressure [Pa]
    omega : Acentric factor

    Returns
    -------
    a, b  : EOS parameters (SI units)
    """
    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(Tr))) ** 2

    a = 0.45724 * (R ** 2 * Tc ** 2 / Pc) * alpha
    b = 0.07780 * (R * Tc / Pc)
    return a, b


def solve_compressibility(T: float, P: float, a: float, b: float) -> float:
    """
    Solve the cubic PR EOS for the compressibility factor Z (vapour root).

    Z³ - (1-B)Z² + (A-2B-3B²)Z - (AB-B²-B³) = 0

    Returns the largest real root (gas phase).
    """
    A = a * P / (R ** 2 * T ** 2)
    B = b * P / (R * T)

    coeffs = [
        1.0,
        -(1.0 - B),
        (A - 2.0 * B - 3.0 * B ** 2),
        -(A * B - B ** 2 - B ** 3),
    ]
    roots = np.roots(coeffs)

    # Keep only real, positive roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    if not real_roots:
        raise ValueError(f"No valid Z root found at T={T:.2f} K, P={P:.2f} Pa")

    return max(real_roots)  # vapour / supercritical phase


def calculate_fugacity_pr(T: float, P: float,
                           Tc: float, Pc: float, omega: float) -> float:
    """
    Calculate gas-phase fugacity f [Pa] using the PR EOS.

    ln(f/P) = Z - 1 - ln(Z-B) - [A/(2√2·B)] · ln[(Z+2.414B)/(Z-0.414B)]

    Parameters
    ----------
    T, P   : Temperature [K] and pressure [Pa]
    Tc, Pc : Critical properties [K, Pa]
    omega  : Acentric factor

    Returns
    -------
    f : Fugacity in Pa
    """
    a, b = pr_parameters(T, Tc, Pc, omega)
    Z = solve_compressibility(T, P, a, b)

    A = a * P / (R ** 2 * T ** 2)
    B = b * P / (R * T)

    ln_phi = (Z - 1.0
              - np.log(Z - B)
              - (A / (2.0 * np.sqrt(2.0) * B))
              * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))

    return P * np.exp(ln_phi)
