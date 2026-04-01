"""
chen_guo_model.py
-----------------
Standalone implementation of the Chen-Guo thermodynamic model for
predicting methane hydrate equilibrium pressure.

Reference:
    Chen, G.-J. & Guo, T.-M. (1996). Thermodynamic modeling of hydrate
    formation based on new concepts. Fluid Phase Equilibria, 122(1-2), 43-65.
"""

import numpy as np
from .peng_robinson_eos import calculate_fugacity_pr


# ── Constants ──────────────────────────────────────────────────────────────────

R = 8.314  # J/(mol·K)  universal gas constant

# Langmuir constants a, b (small cavity) and c, d (large cavity)
# Units: a, c in K·MPa⁻¹ ; b, d in K
LANGMUIR_PARAMS = {
    "methane": {"a1": 0.0037237, "b1": 2708.8, "c1": 0.018373, "d1": 2737.9},
    "ethane":  {"a1": 0.0,       "b1": 0.0,    "c1": 0.006906, "d1": 3631.6},
}

# Structure I basic parameters (Sloan, 1990 / Table 1 in Chen-Guo)
STRUCTURE_I = {
    "lambda1":  1 / 23,   # cavities / water molecule (small)
    "lambda2":  3 / 23,   # cavities / water molecule (large)
    "Z1":       20,
    "Z2":       24,
    "delta_mu": 1120,     # J/mol
    "delta_h_above_T0": -4297,   # J/mol  (T > T0)
    "delta_h_below_T0":  1714,   # J/mol  (T < T0)
    "delta_Vw_above":    4.6e-6, # m³/mol (T > T0)
    "delta_Vw_below":    3.0e-6, # m³/mol (T < T0)
    "delta_Cp0":        -34.58,  # J/(mol·K) (T > T0)
    "delta_Cp0_below":   3.315,  # J/(mol·K) (T < T0)
    "alpha_prime_above": 0.1890, # J/(mol·K²) (T > T0)
    "alpha_prime_below": 0.0121, # J/(mol·K²) (T < T0)
    "T0": 273.15,                # K  reference temperature
}

# Methane critical properties (for PR EOS)
METHANE = {
    "Tc": 190.56,   # K
    "Pc": 4.599,    # MPa
    "omega": 0.011, # acentric factor
}


# ── Helper functions ────────────────────────────────────────────────────────────

def langmuir_constant(T: float, a: float, b: float) -> float:
    """C_j = (a/T) * exp(b/T)  [MPa⁻¹]"""
    return (a / T) * np.exp(b / T)


def cavity_occupancy(C: float, f_g: float) -> float:
    """θ = C·f_g / (1 + C·f_g)  (Langmuir adsorption)"""
    return (C * f_g) / (1.0 + C * f_g)


def delta_mu_w(T: float, P: float, params: dict, gamma_w: float = 1.0,
               x_w: float = 1.0) -> float:
    """
    Chemical potential difference Δμ_w / RT  (dimensionless).
    Eq. 28 in Chen-Guo (Sloan 1990 method).
    """
    T0 = params["T0"]
    R_gas = R

    if T >= T0:
        delta_mu0 = params["delta_mu"]
        delta_h   = params["delta_h_above_T0"]
        delta_Vw  = params["delta_Vw_above"]
        delta_Cp0 = params["delta_Cp0"]
        alpha_p   = params["alpha_prime_above"]
    else:
        delta_mu0 = params["delta_mu"]
        delta_h   = params["delta_h_below_T0"]
        delta_Vw  = params["delta_Vw_below"]
        delta_Cp0 = params["delta_Cp0_below"]
        alpha_p   = params["alpha_prime_below"]

    # Integral of Δh/RT² dT from T0 to T
    delta_Cp = delta_Cp0 + alpha_p * (T - 273.15)
    h_integral = (delta_h / R_gas) * (1.0 / T0 - 1.0 / T) \
                 + (delta_Cp / R_gas) * (np.log(T / T0) - 1.0 + T0 / T)

    # Pressure integral ∫ΔVw/RT dP  (ΔVw treated as constant)
    P_MPa = P / 1e6  # Pa → MPa
    V_integral = (delta_Vw * P_MPa) / (R_gas * T)

    return delta_mu0 / (R_gas * T0) - h_integral + V_integral - np.log(gamma_w * x_w)


def chen_guo_pressure(T: float, gas: str = "methane",
                      P_init: float = 1e6,
                      max_iter: int = 200,
                      tol: float = 1e-4) -> float:
    """
    Iteratively solve for the equilibrium pressure P [Pa] at temperature T [K]
    using the Chen-Guo model + Peng-Robinson EOS.

    Parameters
    ----------
    T       : Temperature in Kelvin
    gas     : Guest molecule key ('methane' or 'ethane')
    P_init  : Initial pressure guess in Pa
    max_iter: Maximum iteration steps
    tol     : Convergence tolerance on fugacity difference

    Returns
    -------
    P : Equilibrium pressure in Pa
    """
    params = STRUCTURE_I
    lp = LANGMUIR_PARAMS[gas]
    alpha = params["lambda1"] / params["lambda2"]   # structure parameter α

    P = P_init
    Tc = METHANE["Tc"]
    Pc = METHANE["Pc"] * 1e6   # MPa → Pa
    omega = METHANE["omega"]

    for _ in range(max_iter):
        # Step 1: fugacity from PR EOS
        f_eos = calculate_fugacity_pr(T, P, Tc, Pc, omega)

        # Step 2: Langmuir constants
        C1 = langmuir_constant(T, lp["a1"], lp["b1"])  # small cavity
        C2 = langmuir_constant(T, lp["c1"], lp["d1"])  # large cavity

        # Step 3: occupancy of small cavity
        theta1 = cavity_occupancy(C1, f_eos / 1e6)  # convert Pa → MPa for C units

        # Step 4: chemical potential difference
        u_w = delta_mu_w(T, P, params)

        # Step 5: fugacity from Chen-Guo equation
        f_cg = np.exp(u_w) * (1.0 / C2) * (1.0 - theta1) ** alpha

        # Convergence check
        diff = abs(f_eos / 1e6 - f_cg)
        if diff < tol:
            break

        # Secant-style pressure update
        P = P * (f_cg / (f_eos / 1e6))

    return P


def chen_guo_pressure_batch(T_array: np.ndarray, **kwargs) -> np.ndarray:
    """Vectorised wrapper — applies chen_guo_pressure over an array of temperatures."""
    return np.array([chen_guo_pressure(T, **kwargs) for T in T_array])
