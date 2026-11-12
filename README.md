# Neural Projectile Motion Predictor

A PyTorch feed-forward regression experiment that predicts horizontal and vertical projectile position from initial conditions and elapsed time.

## Repository files
- `rocket postion ai`: Python script containing a synthetic-data generator followed by model training/evaluation (extensionless filename in the current repository).
- `projectile.csv`: dataset consumed by the training section.
- `requirements.txt`: Python dependencies.

## Setup

Python 3.10+ is recommended.

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python "rocket postion ai"
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python "rocket postion ai"
```

## Workflow and data
The first section generates a synthetic projectile dataset and writes `projectile_dataset.csv`. The training section reads the checked-in `projectile.csv`; it does not automatically train on the generated file. The features are initial x/y position, launch speed, launch angle in degrees, and elapsed time. The targets are horizontal and vertical positions.

The model uses train/test splitting and fits input and target scalers on the training split. It reports MAE and RMSE after converting predictions back to the original position units. Results should be reproduced on the current dataset before quoting a performance figure.

## Assumptions and limitations
The generator uses ideal projectile motion with constant gravity and no air resistance, wind, spin, or changing mass. The generated values are synthetic and do not represent real-world ballistic performance. Check units and angle conventions before changing the dataset. This project is for educational simulation only, not trajectory planning or safety-critical use.
