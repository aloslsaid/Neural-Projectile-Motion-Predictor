from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "projectile.csv"
OUTPUT_DIR = ROOT / "artifacts"
FEATURES = ["x0", "y0", "initial_speed", "angle_degrees", "elapsed_time"]
TARGETS = ["x_position", "y_position"]
SEED = 42


class ProjectileNet(nn.Module):
    def __init__(self, input_size: int = 5, output_size: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[FEATURES + TARGETS].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required.")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGETS].to_numpy(dtype=np.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED
    )
    xs, ys = StandardScaler(), StandardScaler()
    X_train = xs.fit_transform(X_train).astype(np.float32)
    X_val = xs.transform(X_val).astype(np.float32)
    X_test_scaled = xs.transform(X_test).astype(np.float32)
    y_train = ys.fit_transform(y_train).astype(np.float32)
    y_val_scaled = ys.transform(y_val).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=512, shuffle=True,
    )
    Xv, yv = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val_scaled).to(device)
    model = ProjectileNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(800):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d} | validation MSE: {val_loss:.6f}")
        if stale >= 60:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()
    predictions = ys.inverse_transform(pred_scaled)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "feature_columns": FEATURES,
        "target_columns": TARGETS,
        "seed": SEED,
        "validation_mse_scaled": best_loss,
    }, OUTPUT_DIR / "projectile_model.pt")
    joblib.dump({"x_scaler": xs, "y_scaler": ys}, OUTPUT_DIR / "scalers.joblib")
    pd.DataFrame({
        "actual_x": y_test[:, 0], "predicted_x": predictions[:, 0],
        "actual_y": y_test[:, 1], "predicted_y": predictions[:, 1],
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"Device: {device} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}")
    print(f"Artifacts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
