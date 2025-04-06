import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import product

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run FNN regression with grid search and analysis.")
parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--resume_from", type=str, default=None, help="Path to saved model (.pt) to resume training")
parser.add_argument("--resume_training", action="store_true", help="Flag to continue training from saved model")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥 Using device: {device}\n")

# Output root path
output_root = "/home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Mean/Reg_Analysis"
os.makedirs(f"{output_root}/models", exist_ok=True)
os.makedirs(f"{output_root}/results", exist_ok=True)
os.makedirs(f"{output_root}/plots", exist_ok=True)


class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(FNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor.to(device))

    y_pred = y_pred_tensor.cpu().numpy().flatten()
    y_true = y_test_tensor.cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Model Evaluation on Test Data:")
    print(f"✅ Test MSE: {mse:.4f}")
    print(f"✅ Test R²: {r2:.4f}")

    return mse, r2, y_true, y_pred


def plot_results(y_true, y_pred, train_losses=None):
    residuals = y_true - y_pred

    if train_losses is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_root}/plots/loss_curve.png")
        plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle='--', color='red')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/residuals_vs_predicted.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', color='red')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/predicted_vs_actual.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/residuals_distribution.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    absolute_errors = np.abs(residuals)
    sorted_errors = np.sort(absolute_errors)
    cum_dist = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    plt.plot(sorted_errors, cum_dist)
    plt.xlabel("Absolute Error")
    plt.ylabel("Cumulative Proportion")
    plt.title("Cumulative Distribution of Absolute Errors")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/cumulative_absolute_errors.png")
    plt.show()

    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/qq_plot_residuals.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=absolute_errors)
    plt.title("Absolute Error vs Actual Value")
    plt.xlabel("Actual Values")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(f"{output_root}/plots/absolute_error_vs_actual.png")
    plt.show()


def run_fnn_pytorch(csv_path, epochs=100, batch_size=16, lr=0.001,
                    hidden_sizes=(128, 64, 32), save_model=False,
                    model_save_path="fnn_model.pth",
                    load_model_path=None, resume_training=False):
    df = pd.read_csv(csv_path)
    print(f"\n🔍 Running PyTorch FNN Regression on FULL dataset (All Cells)")

    features = ["Area_MA", "Perimeter_MA", "Extent_MA", "Solidity_MA",
                "Compactness_MA", "Elongation_MA", "Circularity_MA", "Convexity_MA"]
    target = "X_Centroid_Velocity_MA"

    df = df.dropna(subset=features + [target])
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    X_test_tensor = torch.tensor(X_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = FNN(input_size=X.shape[1], hidden_sizes=hidden_sizes).to(device)
    if resume_training and load_model_path and os.path.isfile(load_model_path):
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        print(f"🔄 Loaded model weights from: {load_model_path}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    if save_model:
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Model saved to {model_save_path}")

    return model, train_losses, X_test_tensor, y_test_tensor


# ===== Run Script =====

if __name__ == "__main__":
    # Customize training here
    model, train_losses, X_test_tensor, y_test_tensor = run_fnn_pytorch(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=8,
        lr=0.00001,
        hidden_sizes=(128, 64, 32, 16, 8),
        save_model=True,
        model_save_path=f"{output_root}/models/fnn_resumed_ep{args.epochs}_lr0.0005_bs12_hl128x64x32x16x8.pt",
        load_model_path=args.resume_from,
        resume_training=args.resume_training
    )

    mse, r2, y_true, y_pred = evaluate_model(model, X_test_tensor, y_test_tensor)
    plot_results(y_true, y_pred, train_losses=train_losses)

    print("\n📌 Final Model Configuration:")
    print(f"LR: 0.0005 | Batch Size: 12 | Hidden: (128, 64, 32, 16, 8) | Epochs: {args.epochs}")
    print(f"Model saved at: {output_root}/models/fnn_resumed_ep{args.epochs}_lr0.0005_bs12_hl128x64x32x16x8.pt")


# CUDA_VISIBLE_DEVICES=0 python FNN_Mean.py \
# --csv_path /home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Mean/Cells_Centroid_Velocity_TrueLabel_MA_Mean_5.csv \
# --epochs 100 \
# --resume_from /home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Mean/Reg_Analysis/models/fnn_lr0.0005_bs12_ep400_hl128x64x32x16x8.pt \
# --resume_training


# CUDA_VISIBLE_DEVICES=1  python FNN_Mean_ReTr.py --csv_path /home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Mean/Cells_Centroid_Velocity_TrueLabel_MA_Mean_5.csv --epochs 500 --resume_from /home/MinaH
# ossain/EmbedTrack/Track_New_Result_Shape/Mean/Reg_Analysis/models/fnn_lr0.0005_bs12_ep400_hl128x64x32x16x8.pt --resume_training