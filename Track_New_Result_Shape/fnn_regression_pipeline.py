import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from itertools import product


import phate
import scprep
import seaborn as sns
import os
import json
from PIL import Image, ImageDraw
import re
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import tifffile as tiff
from tqdm import tqdm
from skimage.exposure import equalize_adapthist
from scipy.stats import stats
import matplotlib.animation as animation
import pandas as pd
import csv
import shutil
from skimage.morphology import dilation, erosion
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import measure
from skimage.measure import regionprops, label
from scipy.spatial import distance
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D  # 3D Plotting
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score

from skimage.morphology import dilation, erosion
from skimage import measure
from scipy.ndimage import center_of_mass
from glob import glob
import random
from skimage.measure import regionprops, label
from scipy.spatial import distance

import imageio.v2 as imageio
from tifffile import imread

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split



# --- CONFIG ---
output_root = "/home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Median/Reg_Analysis_FNN_PCA"  ## Change it
os.makedirs(f"{output_root}/models", exist_ok=True)
os.makedirs(f"{output_root}/results", exist_ok=True)
os.makedirs(f"{output_root}/plots", exist_ok=True)

# --- FNN Model ---
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(FNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Evaluation ---
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy().flatten()
    y_true = y_test_tensor.cpu().numpy().flatten()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\u2705 MSE: {mse:.4f} | R²: {r2:.4f}")
    return mse, r2, y_true, y_pred

# --- Plotting ---
def plot_results(y_true, y_pred, train_losses=None):
    residuals = y_true - y_pred
    if train_losses:
        plt.figure(); plt.plot(train_losses); plt.title("Training Loss"); plt.tight_layout()
        plt.savefig(f"{output_root}/plots/loss_curve.png"); plt.close()
    plt.figure(); sns.scatterplot(x=y_pred, y=residuals); plt.axhline(0, linestyle='--', color='red')
    plt.title("Residuals vs Predicted"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/residuals_vs_predicted.png"); plt.close()
    plt.figure(); sns.scatterplot(x=y_true, y=y_pred); plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title("Predicted vs Actual"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/predicted_vs_actual.png"); plt.close()
    plt.figure(); sns.histplot(residuals, bins=30, kde=True)
    plt.title("Distribution of Residuals"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/residuals_distribution.png"); plt.close()
    plt.figure(); plt.plot(np.sort(np.abs(residuals)), np.linspace(0, 1, len(residuals)))
    plt.title("Cumulative Absolute Errors"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/cumulative_absolute_errors.png"); plt.close()
    plt.figure(); stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/qq_plot_residuals.png"); plt.close()
    plt.figure(); sns.scatterplot(x=y_true, y=np.abs(residuals))
    plt.title("Absolute Error vs Actual"); plt.tight_layout()
    plt.savefig(f"{output_root}/plots/absolute_error_vs_actual.png"); plt.close()

# --- Run FNN ---
def run_fnn(X_train, X_val, X_test, y_train, y_val, y_test, input_size, hidden_sizes, lr, batch_size, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(input_size=input_size, hidden_sizes=hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().to(device)
    X_val_tensor = torch.tensor(X_val).float().to(device)
    y_val_tensor = torch.tensor(y_val).float().to(device)
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_test_tensor = torch.tensor(y_test).float().to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}")

    return model, train_losses, X_test_tensor, y_test_tensor

# --- Grid Search ---
def grid_search_fnn(csv_path, param_grid):
    df = pd.read_csv(csv_path)
    # shape_features = ['Area_MA', 'Perimeter_MA', 'Extent_MA', 'Solidity_MA', 'Compactness_MA', 'Elongation_MA', 'Circularity_MA']
    shape_features = ['Extent_MA',  'Solidity_MA', 'Compactness_MA', 'Elongation_MA','Perimeter_MA', 'Circularity_MA']
    target = 'X_Centroid_Velocity_MA'
    df = df.dropna(subset=shape_features + [target])

    X_shape = df[shape_features].values
    y = df[target].values.reshape(-1, 1)

    X_shape_scaled = StandardScaler().fit_transform(X_shape)
    X_train_s, X_temp_s, y_train, y_temp = train_test_split(X_shape_scaled, y, test_size=0.2, random_state=42)
    X_val_s, X_test_s, y_val, y_test = train_test_split(X_temp_s, y_temp, test_size=0.5, random_state=42)

    # PCA
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    X_test_pca = pca.transform(X_test_s)

    X_train_final = X_train_pca
    X_val_final = X_val_pca
    X_test_final = X_test_pca

    results = []
    best_r2 = -np.inf
    best_model = None
    best_config = None
    best_losses = None
    best_test_tensors = None

    for lr, batch_size, hidden_sizes, epochs in product(param_grid["lr"], param_grid["batch_size"],
                                                        param_grid["hidden_sizes"], param_grid["epochs"]):
        print(f"▶ Training: lr={lr}, batch={batch_size}, hidden={hidden_sizes}, epochs={epochs}")
        model, train_losses, X_test_tensor, y_test_tensor = run_fnn(
            X_train_final, X_val_final, X_test_final,
            y_train, y_val, y_test,
            input_size=X_train_final.shape[1],
            hidden_sizes=hidden_sizes,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs
        )
        mse, r2, y_true, y_pred = evaluate_model(model, X_test_tensor, y_test_tensor)

        results.append({
            "lr": lr, "batch_size": batch_size, "hidden_sizes": str(hidden_sizes),
            "epochs": epochs, "mse": mse, "r2": r2
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_config = {
                "lr": lr, "batch_size": batch_size,
                "hidden_sizes": hidden_sizes, "epochs": epochs
            }
            best_losses = train_losses
            best_test_tensors = (X_test_tensor, y_test_tensor, y_true, y_pred)

    model_path = f"{output_root}/models/fnn_best_model.pt"
    torch.save(best_model.state_dict(), model_path)

    pd.DataFrame(results).to_csv(f"{output_root}/results/grid_search_results.csv", index=False)
    print(f"🏆 Best config: {best_config}, R²={best_r2:.4f}")
    return best_model, best_config, best_losses, best_test_tensors

# --- Entry Point ---
if __name__ == "__main__":
    param_grid = {
        "lr": [0.001,0.0001,0.005,0.0005,0.00001],
        "batch_size": [8,16,24,32],
        "hidden_sizes": [(1024,512,256),(512,256,128),(256,128,64),(128, 64, 32), (64, 32, 16),(32,16,8),(64,32,16,8), (128, 64,32,16), (256,128,64,32),(1024,512,256,128),
                         (512,256,128, 64, 32), (256,128, 64, 32, 16),(1024,512,256,128,64),(1024,512,256,128,64,32,16),(512,256,128,64,32,16,8),(128,256,512,256,128,64,32),
                         (128,256,512,512,256,128,32),(32,64,182,256,128,64,32),(64,128,256,512,256,128,64,32),(64,128,256,512,512,256,128,64,32),
                         (16,32,64,128,256,256,128,64,32,8),(8,16,32,64,128,256,256,128,64,32,8), (16,32,64,128,256,512,256,128,64,32,16)],
        "epochs": [100, 150,200,300]
    }
    csv_path = "/home/MinaHossain/EmbedTrack/Track_New_Result_Shape/Median/Cells_Centroid_Velocity_TrueLabel_MA_Median_5.csv"
    best_model, best_config, best_losses, (X_test_tensor, y_test_tensor, y_true, y_pred) = grid_search_fnn(csv_path, param_grid)
    plot_results(y_true, y_pred, train_losses=best_losses)
