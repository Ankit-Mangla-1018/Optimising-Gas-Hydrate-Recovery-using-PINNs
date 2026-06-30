# 🧊 Gas Hydrate Equilibrium Prediction using Physics-Guided Neural Networks (PGNNs)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![IIT Kanpur](https://img.shields.io/badge/IIT%20Kanpur-ME496-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**Undergraduate Research Project (ME496) · Indian Institute of Technology Kanpur**

*Supervisor: Prof. Malay K. Das, Department of Mechanical Engineering*  
*Author: Ankit Mangla (220160)*

</div>

---

## 📖 Abstract

Methane hydrates — ice-like structures trapping methane within water molecules under high pressure and low temperature — represent one of the world's largest untapped energy reserves, estimated to exceed all known fossil fuel deposits combined. Predicting their thermodynamic equilibrium is critical for safe and efficient extraction, yet highly complex due to the interplay of temperature, pressure, salinity, and chemical additives.

This project develops a **Physics-Guided Neural Network (PGNN)** that embeds the **Chen-Guo thermodynamic model** directly into a neural network's loss function. The result is a model that learns from data while remaining grounded in established physical laws — achieving an **R² of 0.981** and **MSE of 1.255**, outperforming all purely data-driven baselines.

---

## 🗂️ Repository Structure

```
gas-hydrate-pgnn/
│
├── 📓 notebooks/
│   └── Final_PGNN.ipynb              # Full pipeline: EDA → KNN → ANN → PGNN
│
├── 📊 data/
│   └── Chen_guo_data.xlsx            # Experimental + synthetic equilibrium dataset
│
├── 📄 reports/
│   ├── Final_Midterm_Report.pdf      # Midterm progress: KNN & ANN baselines
│   └── Final_Endterm_Report.pdf      # Final report: PGNN, Chen-Guo integration & results
│
├── 📁 src/
│   ├── chen_guo_model.py             # Chen-Guo physics model (standalone)
│   ├── peng_robinson_eos.py          # Peng-Robinson equation of state
│   ├── pgnn_model.py                 # PGNN architecture & hybrid loss
│   └── data_utils.py                 # Data loading & preprocessing utilities
│
├── 📁 results/
│   └── figures/                      # Saved plots (actual vs predicted, hyperparameter sweeps)
│
├── 📁 .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI (lint + notebook smoke test)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 Methodology

### 1. Data

Synthetic equilibrium data was generated using the equation of state from *Clathrate Hydrates of Natural Gases* (Sloan & Koh), following:

$$P\ [\text{kPa}] = \exp\!\left(a + \frac{b}{T\ [\text{K}]}\right)$$

| Component | Type | T range (°C) | a | b |
|-----------|------|--------------|--------|-----------|
| Methane | Lw–H–V | 0 to 25 | 38.980 | −8533.80 |
| Methane | I–H–V | −25 to 0 | 14.717 | −1886.79 |

### 2. Models Explored

| Model | R² | RMSE | Notes |
|---|---|---|---|
| Multiple Linear Regression | 0.76 | 3.35 | Baseline |
| K-Nearest Neighbours (k=1) | 0.94 | 1.56 | Optimal k via 5-fold CV |
| Artificial Neural Network | 0.84 | 1.38 | 2 hidden layers, 16 neurons |
| **PGNN (ours)** | **0.981** | **1.11** | 3 hidden layers, 64 neurons |

### 3. PGNN Architecture

```
         Input (T, composition)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  ┌───────────┐     ┌───────────────────┐
  │ Chen-Guo  │     │  ANN              │
  │  Physics  │     │  Dense(64) × 3    │
  │   Model   │     │  LeakyReLU(0.1)   │
  └─────┬─────┘     └────────┬──────────┘
        │                    │
       P_phy               P_pre
        │                    │
        └──────────┬─────────┘
                   ▼
      L_total = L_data + λ · L_physics
```

**Composite Loss Function:**

$$L_{\text{total}} = \underbrace{\|P - P_{\text{pre}}\|^2}_{L_{\text{data}}} + \lambda \cdot \underbrace{\|P - P_{\text{phy}}\|^2}_{L_{\text{physics}}}$$

where λ is a tunable hyperparameter balancing data fidelity against physical consistency.

### 4. Chen-Guo Physics Model

The Chen-Guo model treats hydrate formation as a two-step process. Langmuir constants for small (pentagonal dodecahedra) and large (tetrakaidecahedra) cavities:

$$C_{\text{small}} = \frac{a}{T} \exp\!\left(\frac{b}{T}\right), \qquad C_{\text{large}} = \frac{c}{T} \exp\!\left(\frac{d}{T}\right)$$

Equilibrium fugacity:

$$f_g = \exp\!\left(\frac{\Delta\mu_w}{RT\lambda_2}\right) \cdot \frac{1}{C_2} \cdot (1 - \theta_1)^\alpha$$

Gas-phase fugacity is computed via the **Peng-Robinson Equation of State**:

$$p = \frac{RT}{V_m - b} - \frac{a\alpha}{V_m^2 + 2bV_m - b^2}$$

---

## 📊 Results

### Final Model Performance (64 neurons, 3 hidden layers, lr=0.01)

```
R² Score : 0.981
MSE      : 1.255
```

### Hyperparameter Sweep — Neurons

| Neurons | MSE | MAE | R² |
|---------|------|------|------|
| 4 | 60.71 | 5.86 | 0.07 |
| 8 | 60.68 | 5.84 | 0.07 |
| 16 | 57.39 | 5.82 | 0.12 |
| 32 | 6.10 | 1.59 | 0.90 |
| **64** | **1.25** | **0.75** | **0.98** |
| 128 | 1.52 | 0.81 | 0.97 |

### Hyperparameter Sweep — Hidden Layers (64 neurons)

| Hidden Layers | R² |
|---|---|
| 1 | 0.068 |
| 2 | 0.07 |
| **3** | **0.98** |
| 4 | 0.94 |

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
git clone https://github.com/<ankitmangla1018>/gas-hydrate-pgnn.git
cd gas-hydrate-pgnn
pip install -r requirements.txt
```

### Running the Main Notebook

```bash
jupyter notebook notebooks/Final_PGNN.ipynb
```

> 💡 The notebook was originally developed in **Google Colab**. You can also open it directly there — just upload `data/Chen_guo_data.xlsx` when prompted.

### Using the Source Modules

```python
from src.chen_guo_model import chen_guo_pressure
from src.pgnn_model import build_pgnn, hybrid_loss

# Predict equilibrium pressure from temperature
P_phy = chen_guo_pressure(T=280.0)  # K

# Build and train PGNN
model = build_pgnn(n_neurons=64, n_hidden=3, learning_rate=0.01)
```

---

## 🧪 Running Tests / CI

This repo uses **GitHub Actions** to run a smoke test on every push:

```bash
# Locally lint the notebook
pip install nbqa flake8
nbqa flake8 notebooks/Final_PGNN.ipynb --max-line-length=120

# Execute notebook headlessly (requires nbconvert)
jupyter nbconvert --to notebook --execute notebooks/Final_PGNN.ipynb
```

---

## 🔭 Future Work

- [ ] Extend to **electrolyte-containing systems** (NaCl, KCl, CaCl₂, MgCl₂ and other salts/alcohols/glycols)
- [ ] Predict **kinetics** of gas hydrate dissociation
- [ ] Estimate **percentage of gas extracted** given field conditions
- [ ] **Transfer learning** across different hydrate-forming gases (ethane, CO₂, propane)
- [ ] Deploy as a lightweight **web API** for field engineers

---

## 📚 References

1. Chen, S.-K. et al. — *Prediction of hydrate formation boundaries in pure water and salt/alcohol containing systems based on prior knowledge and AI* 
2. Chen, G.-J. & Guo, T.-M. — *Thermodynamic modeling of hydrate formation based on new concepts*
3. Patel, N.C. & Teja, A.S. — *A new cubic equation of state for fluids and fluid mixtures*
4. Sloan, E.D. & Koh, C.A. — *Clathrate Hydrates of Natural Gases*, 3rd ed.

---

## 📄 License

This project is released under the [MIT License](LICENSE) for academic use.  
Please cite this work if you build upon it.
