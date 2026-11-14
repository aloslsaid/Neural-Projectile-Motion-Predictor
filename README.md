# Neural Projectile Motion Predictor

> A PyTorch regression experiment that learns to predict the horizontal and vertical position of a projectile from its initial conditions and elapsed time.

## Overview
This project explores how a neural network can approximate projectile-motion equations from generated or recorded tabular examples. It accepts initial position, launch speed, launch angle, and elapsed time, then predicts the projectile's X and Y coordinates.

## Input and output features
Inputs:
- `x0` — initial horizontal position.
- `y0` — initial vertical position.
- `initial_speed` — launch speed.
- `angle_degrees` — launch angle.
- `elapsed_time` — time since launch.

Targets:
- `x_position` — horizontal position.
- `y_position` — vertical position.

## Dataset generation
The repository includes a script that generates 100,000 simulated examples using randomized initial conditions and the standard constant-gravity equations:
- Horizontal motion: constant horizontal velocity.
- Vertical motion: initial vertical velocity with gravitational acceleration.
- Samples below ground level are filtered out.

The training script reads `projectile.csv`, scales both input and output arrays, and trains a feed-forward PyTorch network with two predicted coordinates.

## Model and evaluation
The intended architecture contains two hidden layers (216 and 108 units) with ReLU activations and a two-value output. Training uses Adam and mean-squared error. The script reports MAE and RMSE after converting predictions back to the original coordinate scale.

## Installation
Python 3.10+ is recommended:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn torch
```

## Run
The training script is named `rocket postion ai`:
```bash
python "rocket postion ai"
```
Run it from the repository root. It expects the file `projectile.csv`. The generation section in the script writes `projectile_dataset.csv`, so ensure the filename used for training matches the dataset you intend to train on.

## Current source note
The checked-in model class currently uses `init` instead of Python's required `__init__`; the model construction therefore needs a source correction before the training script can run successfully. This documentation-only update does not alter that code.

## Scientific scope
This is a simplified idealized projectile model. It does not include air resistance, wind, spin, changing gravity, or collisions. Its predictions approximate the generated data distribution and should not be treated as a real-world ballistic or engineering calculator.

## Skills demonstrated
Physics-informed synthetic data · PyTorch · Neural-network regression · Feature scaling · Simulation · MAE · RMSE
