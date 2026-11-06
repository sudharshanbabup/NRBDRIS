# Nonreciprocal BD-RIS for Monostatic ISAC

Simulation and optimization code for the paper:
> **Nonreciprocal Beyond Diagonal RIS for Monostatic ISAC: Gain Laws, Contraction Core Optimization, and Size Agnostic Learning**  

---

## 1. Requirements

- Python >= 3.10
- MATLAB >= R2023b (optional, only needed for rendering figures)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Execution

### Option A: Plot Figures from Pre-computed Data (Instant)
All figure datasets from the paper are stored in `matlab/data/`. In MATLAB:
```matlab
cd matlab
matlab_fig_1('data', 'figures')
```
Publication-ready vector PDFs are exported to `matlab/figures/`.

### Option B: Run Specific Experiments
From the `src/` directory:
```bash
cd src

# Beam patterns (Figs. 2 & 3)
python3 exp_misc.py beams

# Theoretical gain laws: SCNR vs CNR (Fig. 4) and offset law (Fig. 5)
python3 exp_theory.py t1
python3 exp_theory.py t2

# Convergence (Fig. 6) and runtime comparison (Fig. 7, Table II)
python3 exp_misc.py conv2
python3 exp_misc.py runtime
python3 timing.py

# Monte Carlo sweeps (Figs. 8–15)
# Usage: python3 mc.py <param> <worker_id> <num_workers>
# Sweep parameters: ctr, P, gam, Q, N, kap
python3 mc.py ctr 0 1

# CoreNet: unsupervised training & cross-size evaluation (Figs. 16 & 17)
STEPS=8000 python3 train_run.py nr
python3 eval_val_ao.py 300
python3 eval_corenet.py 32,48,64,96,128,192,256 100 s30
```

### Option C: Reproduce All Simulations
To re-run all simulations and re-export the figure data:
```bash
bash run_all.sh

cd matlab
python3 export.py    # exports results to data_raw/
python3 style.py     # converts data_raw/ to styled data/
```
Then draw the figures in MATLAB as in Option A.

---

## 3. Repository Structure

- `src/`:
  - `core.py`: Signal model, Halmos unitary dilation lift/reduction, metric calculations.
  - `solvers.py`: Contraction core Minorization-Maximization (MM) and rank-1 tight SDP beamformer.
  - `scenario.py`: Channel models, simulation geometry, and parameter definitions.
  - `theory.py` & `exp_theory.py`: Closed-form gain law implementations and analytical validation.
  - `mc.py`: Multi-worker Monte Carlo parameter sweeps over channel realizations.
  - `l2o.py`, `train_run.py`, `eval_corenet.py`: CoreNet architecture, training loop, and evaluation.
  - `timing.py`: Runtime benchmarks (Table II).
- `matlab/`:
  - `matlab_fig_1.m`: Standalone script to render publication figures.
  - `data/`: Pre-computed JSON data for all 16 paper figures.
  - `export.py` & `style.py`: Pipeline to process raw simulation outputs into figure JSONs.
