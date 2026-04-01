"""
data_utils.py
-------------
Data loading, synthetic data generation, and preprocessing utilities
for the gas hydrate PGNN project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Synthetic data generation ────────────────────────────────────────────────────

# Equilibrium coefficients P [kPa] = exp(a + b/T [K])
EQUILIBRIUM_COEFFS = {
    "methane_LwHV": {"a": 38.980, "b": -8533.80, "T_range": (273.15, 298.15)},
    "methane_IHV":  {"a": 14.717, "b": -1886.79, "T_range": (248.15, 273.15)},
}


def synthetic_equilibrium_data(n_points: int = 500,
                                component: str = "methane_LwHV",
                                noise_std: float = 0.02,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic temperature–pressure equilibrium data.

    Parameters
    ----------
    n_points   : Number of data points
    component  : Key from EQUILIBRIUM_COEFFS
    noise_std  : Fractional Gaussian noise added to P
    seed       : Random seed

    Returns
    -------
    DataFrame with columns ['T_K', 'P_Pa']
    """
    rng = np.random.default_rng(seed)
    coeffs = EQUILIBRIUM_COEFFS[component]
    T_min, T_max = coeffs["T_range"]

    T = rng.uniform(T_min, T_max, n_points)
    P_kPa = np.exp(coeffs["a"] + coeffs["b"] / T)
    P_Pa  = P_kPa * 1e3 * (1.0 + rng.normal(0, noise_std, n_points))

    return pd.DataFrame({"T_K": T, "P_Pa": P_Pa})


# ── File loading ─────────────────────────────────────────────────────────────────

def load_excel_data(filepath: str,
                    T_col: str = "T(K)",
                    P_col: str = "P(Pa)") -> pd.DataFrame:
    """
    Load experimental/synthetic data from the project Excel file.

    Parameters
    ----------
    filepath : Path to .xlsx file
    T_col    : Column name for temperature
    P_col    : Column name for pressure

    Returns
    -------
    Cleaned DataFrame with columns ['T_K', 'P_Pa']
    """
    df = pd.read_excel(filepath)
    df = df[[T_col, P_col]].dropna()
    df.columns = ["T_K", "P_Pa"]
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────────

def prepare_splits(df: pd.DataFrame,
                   feature_col: str = "T_K",
                   target_col: str = "P_Pa",
                   test_size: float = 0.2,
                   seed: int = 42):
    """
    Split and scale data for model training.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled features, raw targets)
    scaler_X : fitted StandardScaler for features
    """
    X = df[[feature_col]].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler_X
