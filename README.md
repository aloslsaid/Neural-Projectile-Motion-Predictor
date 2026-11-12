# Neural Projectile Motion Predictor

A PyTorch regression experiment that predicts horizontal and vertical projectile position from initial position, launch speed, launch angle and elapsed time.

## Inputs and outputs

Inputs: `x0`, `y0`, `initial_speed`, `angle_degrees`, `elapsed_time`.

Targets: `x_position`, `y_position`.

The maintained training entry point reads the checked-in `projectile.csv`, validates its schema, standardizes features and targets, trains a feed-forward neural network, selects the best checkpoint using validation loss, and evaluates once on a held-out test set.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

## Outputs

Generated under `artifacts/`:

- `projectile_model.pt`: trained model and column metadata.
- `scalers.joblib`: fitted input and target scalers.
- `test_predictions.csv`: actual and predicted test coordinates.
- Console metrics: MAE, RMSE and R².

The dataset is based on idealized projectile motion. It does not represent drag, wind, spin, variable gravity or real-world launch uncertainty. Check units and generation assumptions before interpreting results.

## Reproducibility and safety

A fixed random seed is used. Do not use this educational model for weapon guidance, safety-critical trajectory planning or real-world targeting.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
