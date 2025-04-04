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
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥 Using device: {device}\n")

# Create timestamped output directory
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_root = f"./Reg_Analysis_{timestamp}"
output_root=f"/home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Median/Reg_Analysis"
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

    print("\n\U0001F4CA Model Evaluation on Test Data:")
    print(f"✅ Test Mean Squared Error (MSE): {mse:.4f}")
    print(f"✅ Test R-squared (R²): {r2:.4f}")

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
                    hidden_sizes=(128, 64, 32), save_model=False, model_save_path="fnn_model.pth"):
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


def grid_search_fnn(csv_path, param_grid, save_best_model=True, output_results_csv=f"{output_root}/results/grid_search_results.csv"):
    results = []
    best_r2 = -float("inf")
    best_config = None
    best_model = None
    best_train_losses = None
    best_X_test_tensor = None
    best_y_test_tensor = None

    all_combinations = list(product(param_grid["lr"], param_grid["batch_size"], param_grid["hidden_sizes"], param_grid["epochs"]))

    for i, (lr, batch_size, hidden_sizes, epochs) in enumerate(all_combinations, 1):
        print(f"\n🔍 [{i}/{len(all_combinations)}] Config: lr={lr}, batch_size={batch_size}, hidden_sizes={hidden_sizes}, epochs={epochs}")

        model, train_losses, X_test_tensor, y_test_tensor = run_fnn_pytorch(
            csv_path=csv_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_sizes=hidden_sizes,
            save_model=False
        )

        mse, r2, y_true, y_pred = evaluate_model(model, X_test_tensor, y_test_tensor)

        results.append({
            "lr": lr,
            "batch_size": batch_size,
            "hidden_sizes": str(hidden_sizes),
            "epochs": epochs,
            "mse": round(mse, 4),
            "r2": round(r2, 4)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_config = {
                "lr": lr,
                "batch_size": batch_size,
                "hidden_sizes": hidden_sizes,
                "epochs": epochs,
                "model_name": f"fnn_lr{lr}_bs{batch_size}_ep{epochs}_hl{'x'.join(map(str, hidden_sizes))}.pt"
            }
            best_model = model
            best_train_losses = train_losses
            best_X_test_tensor = X_test_tensor
            best_y_test_tensor = y_test_tensor

    if save_best_model and best_model is not None:
        model_save_path = f"{output_root}/models/{best_config['model_name']}"
        torch.save(best_model.state_dict(), model_save_path)
        print(f"💾 Best model saved as: {model_save_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_results_csv, index=False)
    print(f"\n✅ Grid Search Summary saved to: {output_results_csv}")
    print(f"🏆 Best R² = {best_r2:.4f} | Best Config = {best_config}")

    return best_model, best_config, results_df, best_train_losses, best_X_test_tensor, best_y_test_tensor


if __name__ == "__main__":
    # param_grid = {
    #     "lr": [0.001, 0.0005],
    #     "batch_size": [16, 32],
    #     "hidden_sizes": [(128, 64, 32), (128, 128), (64, 32)],
    #     "epochs": [100, 200, 300, 400, 500]
    # }

    param_grid = {
    "lr": [0.001, 0.0001,0.0005,0.00001,0.00005,0.000001],
    "batch_size": [8, 12, 16, 24, 32],
    "hidden_sizes": [(1024,512,256,128, 64, 32,16,8),(512,256,128, 64, 32,16,8),(256,128, 64, 32,16,8),(128, 64, 32,16,8), (64, 32,16,8), (32,16,8),(16,8),
                     (256,128,128, 64, 64,16,8), (512,256,256, 128, 32,32,8),(256,128,64, 64, 32,8,8),(256,128,128, 64, 64,32,8),(64, 32,32,16,8),(32, 32,16,8)],
    "epochs": [100,200,300,400,500]
                }

    best_model, best_config, results_df, best_train_losses, best_X_test_tensor, best_y_test_tensor = grid_search_fnn(
        csv_path=args.csv_path,
        param_grid=param_grid,
        save_best_model=True
    )

    mse, r2, y_true, y_pred = evaluate_model(best_model, best_X_test_tensor, best_y_test_tensor)
    plot_results(y_true, y_pred, train_losses=best_train_losses)

    print("\n🏆 Best Model Details:")
    print(best_config)

    if best_model:
        print("\n🧠 Model Architecture:")
        print(best_model)
        print(f"\n💾 Best model is saved as: {output_root}/models/{best_config['model_name']}")
    else:
        print("❌ No best model was found.")



# python3 fnn_regression_pipeline.py --csv_path /path/to/your/data.csv --epochs 300   # For a particular number of epochs. 
# python FNN_Mean.py --csv_path /home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Mean/Cells_Centroid_Velocity_TrueLabel_MA_Mean_5.csv 
# CUDA_VISIBLE_DEVICES=0  python  FNN_Mean.py --csv_path /home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Median/Cells_Centroid_Velocity_TrueLabel_MA_Median_5.csv 