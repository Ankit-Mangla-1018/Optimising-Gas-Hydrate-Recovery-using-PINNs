# Results

This folder stores generated figures and saved model artefacts.

## figures/
Plots produced by `notebooks/Final_PGNN.ipynb`:
- `actual_vs_predicted_pgnn.png` — final PGNN fit
- `actual_vs_predicted_knn.png` — KNN baseline
- `actual_vs_predicted_ann.png` — ANN baseline
- `epochs_vs_r2.png` — R² across epoch counts
- `neurons_sweep.png` — MSE/MAE/R² vs number of neurons
- `hidden_layers_sweep.png` — R² vs hidden layer count
- `learning_rate_sweep.png` — error metrics vs learning rate

## models/
Saved Keras model weights (`.h5` / `.keras`).  
These files are excluded from version control by `.gitignore`.  
To regenerate, run the full notebook.
