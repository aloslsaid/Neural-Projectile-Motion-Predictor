# Neural Projectile Motion Predictor

An educational project exploring a PyTorch feedforward neural network that predicts a projectile's horizontal and vertical position from initial conditions and elapsed time. The repository contains a CSV dataset and a folder named `rocket postion ai`; inspect that folder for the actual training code and its own requirements before running it.

## Getting started

1. Clone or download the repository.
2. Inspect `rocket postion ai` and the CSV header to confirm the expected inputs and entry point.
3. Create a Python virtual environment.
4. Install the dependencies required by the source code. There is currently no root `requirements.txt`, so do not assume a package list.
5. Run the training/evaluation script using the command documented in that folder, if provided.

The project overview describes synthetic or prepared projectile-motion data, feature scaling, model training, and MAE/RMSE evaluation. Confirm from the code whether the checked-in `projectile.csv` is used or data is generated at runtime. The reported RMSE in older notes should not be treated as independently reproduced by this guide.

## Testing and interpretation

No root test suite or verified test command is currently available. Validate input units and angle conventions, use a held-out test set, and compare predictions with the physics equations used to generate the data. This is a learning project, not a simulator or safety-critical trajectory planner.
