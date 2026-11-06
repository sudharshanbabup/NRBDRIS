#!/bin/bash
set -e
# Navigate to src/ if launched from repository root
if [ -d "src" ]; then
  cd src
fi
export OMP_NUM_THREADS=1
mkdir -p results results2

# Section VI.B: theory validation and beam patterns
python3 exp_theory.py t1
python3 exp_theory.py t2
python3 exp_misc.py beams

# Section VI.C: convergence and run time
python3 exp_misc.py conv2
python3 exp_misc.py runtime

# Section VI.D: Monte Carlo sweeps (Figs. 8 to 15)
for e in ctr P Q N kap gam; do python3 mc.py $e 0 1; done
python3 exp_misc.py group 60 0 1

# Section VI.E: CoreNet training, evaluation and Table II
STEPS=8000 python3 train_run.py nr
python3 eval_val_ao.py 300
python3 eval_corenet.py 32,48,64,96,128,192,256 100 s30
python3 timing.py
