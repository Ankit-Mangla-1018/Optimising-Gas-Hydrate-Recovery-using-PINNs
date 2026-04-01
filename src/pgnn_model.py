"""
pgnn_model.py
-------------
Physics-Guided Neural Network (PGNN) for methane hydrate equilibrium
pressure prediction.

Architecture:
  - ANN path   : multi-layer feedforward net  → P_pre
  - Physics path: Chen-Guo model              → P_phy
  - Loss        : L_total = L_data + λ · L_physics
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score


# ── Model builder ───────────────────────────────────────────────────────────────

def build_ann(n_neurons: int = 64,
              n_hidden: int = 3,
              learning_rate: float = 0.01,
              loss_fn=None) -> keras.Model:
    """
    Build and compile the ANN component of the PGNN.

    Parameters
    ----------
    n_neurons     : Neurons per hidden layer
    n_hidden      : Number of hidden layers
    learning_rate : Adam learning rate
    loss_fn       : Custom loss function (pass hybrid_loss for PGNN);
                    defaults to MSE for plain ANN training.

    Returns
    -------
    Compiled Keras Sequential model
    """
    model_layers = [keras.Input(shape=(1,))]
    for _ in range(n_hidden):
        model_layers.append(layers.Dense(n_neurons))
        model_layers.append(layers.LeakyReLU(negative_slope=0.1))
    model_layers.append(layers.Dense(1, activation="linear"))

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn if loss_fn else "mean_squared_error",
        metrics=["mae", "mse"],
    )
    return model


def make_hybrid_loss(P_phy_train: np.ndarray, lam: float = 0.5):
    """
    Factory that returns a hybrid loss function closing over physics predictions.

    L_total = MSE(y_true, y_pred) + λ · MSE(y_true, P_phy)

    Parameters
    ----------
    P_phy_train : Physics-model predictions aligned with training targets
    lam         : Weight λ for the physics consistency term

    Returns
    -------
    Keras-compatible loss function
    """
    P_phy_tensor = tf.constant(P_phy_train.reshape(-1, 1), dtype=tf.float32)

    def hybrid_loss(y_true, y_pred):
        l_data    = tf.reduce_mean(tf.square(y_true - y_pred))
        l_physics = tf.reduce_mean(tf.square(y_true - P_phy_tensor))
        return l_data + lam * l_physics

    return hybrid_loss


# ── Training helper ─────────────────────────────────────────────────────────────

def train_pgnn(X_train, y_train, X_test, y_test,
               P_phy_train: np.ndarray,
               n_neurons: int = 64,
               n_hidden: int = 3,
               learning_rate: float = 0.01,
               lam: float = 0.5,
               epochs: int = 1000,
               batch_size: int = 32,
               verbose: int = 0):
    """
    Train the full PGNN and return the model + evaluation metrics.

    Parameters
    ----------
    X_train, y_train : Training features and targets
    X_test,  y_test  : Test features and targets
    P_phy_train      : Chen-Guo physics predictions for training samples
    n_neurons        : Neurons per hidden layer
    n_hidden         : Number of hidden layers
    learning_rate    : Adam lr
    lam              : Physics loss weight λ
    epochs           : Training epochs
    batch_size       : Mini-batch size
    verbose          : Keras verbosity (0 = silent)

    Returns
    -------
    model   : Trained Keras model
    metrics : dict with r2, mse, mae on the test set
    history : Keras History object
    """
    loss_fn = make_hybrid_loss(P_phy_train, lam=lam)
    model = build_ann(n_neurons=n_neurons, n_hidden=n_hidden,
                      learning_rate=learning_rate, loss_fn=loss_fn)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "r2":  r2_score(y_test, y_pred),
        "mse": float(np.mean((y_test - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
    }
    return model, metrics, history
