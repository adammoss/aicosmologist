import os
import sys
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.fft import fftn # Use scipy's fftn as numpy's can be slower sometimes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler # Simpler for Z-score

# Suppress specific warnings if necessary, but generally avoid catching errors
# warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# --- Configuration ---
# Paths (relative to root directory)
DATA_DIR = "data"
PARAMS_FILE = os.path.join(DATA_DIR, "latin_hypercube_params.txt")
FIELDS_FILE = os.path.join(DATA_DIR, "latin_hypercube_3D.npy")
OUTPUT_DIR = "out"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
LOG_FILE_TRAIN = os.path.join(OUTPUT_DIR, "train_metrics.txt")
LOG_FILE_VAL = os.path.join(OUTPUT_DIR, "val_metrics.txt")
LOG_FILE_TEST = os.path.join(OUTPUT_DIR, "test_metrics.txt")

# Parameters
PARAM_NAMES = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
N_PARAMS = len(PARAM_NAMES)
N_SIMULATIONS = 2000
BOX_SIZE_MPC_H = 1000.0 # Box size in Mpc/h
GRID_SIZE = 64

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Might make things non-deterministic, but potentially faster
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# --- Training Hyperparameters ---
BATCH_SIZE = 32 # Adjust based on GPU memory
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 200 # Set sufficiently high, rely on early stopping
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
# Patience values
SCHEDULER_PATIENCE = 15
EARLY_STOPPING_PATIENCE = 30 # Should be >= SCHEDULER_PATIENCE

# Physics Feature Config
N_PK_BINS = 20 # Number of log-spaced bins for P(k)
K_MIN_PHYS = 2 * np.pi / BOX_SIZE_MPC_H
K_MAX_PHYS = np.pi * GRID_SIZE / BOX_SIZE_MPC_H
BAO_K_RANGE = (0.05, 0.15) # Approximate k range in h/Mpc for BAO features
N_PDF_BINS = 30 # Bins for density PDF histogram

# --- Wandb Configuration ---
WANDB_PROJECT = "quijote-simulations-3D"
WANDB_RUN_ID = "74e73666" # Specific run ID to resume or create
RESUME_WANDB = True # Try to resume the specific run

# --- Helper Functions ---

def ensure_dir_exists(path):
    """Creates directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_metrics_to_file(filepath, epoch, metrics):
    """Appends metrics to a text file."""
    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode) as f:
        if mode == 'w':
            f.write("Epoch\t" + "\t".join(metrics.keys()) + "\n")
        log_line = f"{epoch}\t" + "\t".join(map(str, metrics.values())) + "\n"
        f.write(log_line)

def save_test_metrics_to_file(filepath, metrics_dict):
    """Saves final test metrics to a text file."""
    with open(filepath, 'w') as f:
        f.write("Parameter\tMSE\tMAE\tR2\n")
        total_mse, total_mae, total_r2 = 0, 0, 0
        count = 0
        for param, metrics in metrics_dict.items():
            f.write(f"{param}\t{metrics['mse']:.6e}\t{metrics['mae']:.6e}\t{metrics['r2']:.6f}\n")
            if isinstance(metrics['mse'], (int, float)): # Avoid adding if it's somehow not a number
                 total_mse += metrics['mse']
                 total_mae += metrics['mae']
                 total_r2 += metrics['r2']
                 count += 1
        if count > 0:
             avg_mse = total_mse / count
             avg_mae = total_mae / count
             avg_r2 = total_r2 / count
             f.write("\nAverage across parameters:\n")
             f.write(f"Average\t{avg_mse:.6e}\t{avg_mae:.6e}\t{avg_r2:.6f}\n")

def calculate_power_spectrum(density_field, k_bins):
    """Calculates the radially binned 3D power spectrum."""
    n = density_field.shape[0]
    if not (density_field.shape[0] == density_field.shape[1] == density_field.shape[2]):
        raise ValueError("Density field must be cubic")

    # FFT and Power
    fft_field = fftn(density_field)
    power = np.abs(fft_field)**2
    power /= (n**6) # Normalization: <delta_k delta_k^*> = P(k) * (2pi)^3 / V * delta_dirac(k-k'). often V/(N^6) scaling used. Let's use V/N^6, V=(1000)^3

    # k-space coordinates
    k_vals = np.fft.fftfreq(n, d=BOX_SIZE_MPC_H / n / (2 * np.pi)) # k in h/Mpc
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2).flatten()

    # Ignore k=0 mode
    k_magnitude = k_magnitude[1:]
    power = power.flatten()[1:]

    # Binning
    pk_means, _, _ = sp_stats.binned_statistic(
        k_magnitude, power, statistic='mean', bins=k_bins
    )
    # Replace NaN bins (if any) with 0 or interpolation? Let's use 0 for simplicity.
    pk_means = np.nan_to_num(pk_means, nan=0.0)
    return pk_means, k_bins

def calculate_physics_features(density_field_raw, log1p_density_field, k_bins, bao_k_range, n_pdf_bins):
    """Extracts P(k) and PDF-based features."""
    features = []

    # 1. Power Spectrum Features (using raw density field)
    # Normalise density field: delta = rho / mean(rho) - 1
    mean_density = np.mean(density_field_raw)
    if mean_density <= 0: # Avoid division by zero or negative density
         # Handle this case - maybe return zeros or raise error? For Quijote, should be positive.
         # Let's assume mean_density is positive.
         delta_field = (density_field_raw / mean_density) - 1
    else:
         delta_field = (density_field_raw / mean_density) - 1

    pk_means, k_bin_edges = calculate_power_spectrum(delta_field, k_bins)
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    features.extend(pk_means) # Add binned P(k) values

    # BAO-related features (ratios/slopes in the BAO range)
    bao_mask = (k_bin_centers > bao_k_range[0]) & (k_bin_centers < bao_k_range[1])
    pk_bao = pk_means[bao_mask]
    if len(pk_bao) >= 3:
        # Simple features: Ratio of adjacent bins, maybe a slope fit
        ratios = pk_bao[1:] / pk_bao[:-1]
        features.extend(np.nan_to_num(ratios, nan=1.0)) # Use 1.0 if division by zero
        # Fit a simple slope to log(P(k)) vs log(k) in BAO range
        if np.all(pk_bao > 0) and np.all(k_bin_centers[bao_mask] > 0):
             log_pk = np.log(pk_bao)
             log_k = np.log(k_bin_centers[bao_mask])
             slope, _, _, _, _ = sp_stats.linregress(log_k, log_pk)
             features.append(np.nan_to_num(slope, nan=0.0))
        else:
             features.append(0.0) # Append 0 if cannot compute slope
    else:
        # Not enough bins in BAO range, append placeholders (e.g., zeros)
        # Need to ensure feature vector dimension is consistent
        # Let's decide on a fixed number of expected BAO features (e.g., 3)
        num_expected_bao_derived = 3 # Example: 2 ratios, 1 slope
        actual_bao_derived = (len(pk_bao) - 1 if len(pk_bao) > 1 else 0) + (1 if len(pk_bao) >= 3 else 0)
        features.extend([0.0] * (num_expected_bao_derived - actual_bao_derived))


    # Shape/Amplitude feature (e.g., slope at larger scales, amplitude at k=0.1)
    # Example: slope between first few bins
    if len(pk_means) >= 3 and np.all(pk_means[:3] > 0) and np.all(k_bin_centers[:3] > 0):
        log_pk_lowk = np.log(pk_means[:3])
        log_k_lowk = np.log(k_bin_centers[:3])
        slope_lowk, _, _, _, _ = sp_stats.linregress(log_k_lowk, log_pk_lowk)
        features.append(np.nan_to_num(slope_lowk, nan=0.0))
    else:
        features.append(0.0)
    # Amplitude near k=0.1 h/Mpc
    target_k = 0.1
    closest_k_idx = np.argmin(np.abs(k_bin_centers - target_k))
    features.append(pk_means[closest_k_idx])


    # 2. Void/PDF Statistics (using log1p transformed field)
    hist, bin_edges = np.histogram(log1p_density_field.flatten(), bins=n_pdf_bins, density=True)
    features.extend(hist) # Add histogram values as features

    # Moments/Percentiles (focus on underdense)
    percentiles = np.percentile(log1p_density_field.flatten(), [5, 10, 25]) # Low percentiles represent underdense regions
    variance = np.var(log1p_density_field.flatten())
    skewness = sp_stats.skew(log1p_density_field.flatten())
    kurt = sp_stats.kurtosis(log1p_density_field.flatten())
    features.extend(percentiles)
    features.append(variance)
    features.append(skewness)
    features.append(kurt)

    return np.array(features, dtype=np.float32)

# --- Dataset Definition ---
class CosmologyDataset(Dataset):
    def __init__(self, fields, params, physics_features,
                 field_scaler, physics_scaler, params_scaler,
                 apply_augmentation=False):
        self.fields = fields
        self.params = params
        self.physics_features = physics_features
        self.field_scaler = field_scaler
        self.physics_scaler = physics_scaler
        self.params_scaler = params_scaler
        self.apply_augmentation = apply_augmentation

        # Pre-apply log1p transform to fields
        self.log_fields = np.log1p(self.fields)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        log_field = self.log_fields[idx].copy() # Ensure we work with a copy for augmentation
        phys_features = self.physics_features[idx].copy()
        target_params = self.params[idx].copy()

        # Augmentation (only for training)
        if self.apply_augmentation:
            # Random Rotations (0, 90, 180, 270 degrees along axes 1, 2)
            # Note: k specifies number of 90 degree rotations. axes specifies plane.
            if np.random.rand() > 0.5:
                 k_rot = np.random.randint(0, 4)
                 axes_rot = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
                 log_field = np.rot90(log_field, k=k_rot, axes=axes_rot)
            # Random Flips (along axes 0, 1, 2)
            if np.random.rand() > 0.5:
                 axis_flip = np.random.randint(0, 3)
                 log_field = np.flip(log_field, axis=axis_flip)

        # Add channel dimension: (64, 64, 64) -> (1, 64, 64, 64)
        log_field = np.expand_dims(log_field, axis=0)

        # Convert to tensor BEFORE normalization (Scalers expect numpy arrays typically)
        # But our scaler will be applied on tensors for consistency with transforms
        log_field_tensor = torch.from_numpy(log_field.copy()).float() # Use .copy() if augmentation modified it
        phys_features_tensor = torch.from_numpy(phys_features).float()
        target_params_tensor = torch.from_numpy(target_params).float()

        # Apply Z-score Normalization (using pre-fitted scalers)
        # Field normalization - needs reshaping for scaler
        # Scaler expects (n_samples, n_features) or similar. We apply per-pixel/channel.
        # Assuming field_scaler stores mean and std (scalar or per channel)
        # We calculated mean/std across all pixels of training fields
        mean_field, std_field = self.field_scaler['mean'], self.field_scaler['std']
        log_field_tensor = (log_field_tensor - mean_field) / std_field

        # Physics features normalization
        mean_phys, std_phys = self.physics_scaler['mean'], self.physics_scaler['std']
        phys_features_tensor = (phys_features_tensor - mean_phys) / std_phys

        # Target parameters normalization
        mean_params, std_params = self.params_scaler['mean'], self.params_scaler['std']
        target_params_tensor = (target_params_tensor - mean_params) / std_params

        return log_field_tensor, phys_features_tensor, target_params_tensor

# --- Model Architecture ---

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True), # Use ReLU or GELU
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# Basic 3D Residual Block with SE
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResBlock3D, self).__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True) # Or GELU
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if self.use_se:
            self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# Physics-Augmented Attentive 3D ResNet
class PhysicsResNet(nn.Module):
    def __init__(self, block, num_blocks, num_phys_features, num_params=N_PARAMS, use_se=True):
        super(PhysicsResNet, self).__init__()
        self.in_channels = 64
        self.use_se = use_se

        # Initial convolution (1 input channel for density field)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True) # Or GELU

        # Residual layers (like ResNet-18 structure)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Calculate CNN feature dimension after pooling
        cnn_feature_dim = 512 # Based on the last layer's output channels

        # MLP Head for regression
        self.fc = nn.Sequential(
            nn.Linear(cnn_feature_dim + num_phys_features, 256),
            nn.GELU(), # Using GELU as suggested in plan
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_params)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_se=self.use_se))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x_field, x_phys):
        # CNN path for density field
        out = self.relu(self.bn1(self.conv1(x_field)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        cnn_features = torch.flatten(out, 1)

        # Fusion: Concatenate CNN features and physics features
        combined_features = torch.cat((cnn_features, x_phys), dim=1)

        # MLP Head
        output = self.fc(combined_features)
        return output

def ResNet18_3D_Physics(num_phys_features, num_params=N_PARAMS, use_se=True):
    # Corresponds to ResNet-18 structure: [2, 2, 2, 2] blocks
    return PhysicsResNet(ResBlock3D, [2, 2, 2, 2], num_phys_features, num_params, use_se)

# --- Training and Evaluation Functions ---

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (fields, physics, params) in enumerate(dataloader):
        fields, physics, params = fields.to(device), physics.to(device), params.to(device)

        optimizer.zero_grad()

        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(fields, physics)
            loss = criterion(outputs, params)

        if torch.isnan(loss):
             print(f"ERROR: Loss is NaN at batch {batch_idx}! Exiting.")
             # Optionally save state or debug info here
             raise ValueError("NaN loss detected during training")

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Basic progress output (replace tqdm)
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
             elapsed = time.time() - start_time
             print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
             start_time = time.time() # Reset timer for next chunk

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fields, physics, params in dataloader:
            fields, physics, params = fields.to(device), physics.to(device), params.to(device)

            # No autocast needed for evaluation if not optimizing
            outputs = model(fields, physics)
            loss = criterion(outputs, params)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, params_scaler, device, param_names, output_dir):
    model.eval()
    all_preds_normalized = []
    all_targets_normalized = []

    with torch.no_grad():
        for fields, physics, params_norm in dataloader:
            fields, physics = fields.to(device), physics.to(device)
            params_norm = params_norm.to(device) # Keep targets normalized on device initially

            preds_norm = model(fields, physics)

            all_preds_normalized.append(preds_norm.cpu().numpy())
            all_targets_normalized.append(params_norm.cpu().numpy())

    all_preds_normalized = np.concatenate(all_preds_normalized, axis=0)
    all_targets_normalized = np.concatenate(all_targets_normalized, axis=0)

    # Inverse transform using the *training* parameter scaler
    # Scaler expects (n_samples, n_features)
    mean_params = torch.tensor(params_scaler['mean']).float().numpy() # Ensure numpy
    std_params = torch.tensor(params_scaler['std']).float().numpy()
    all_preds = all_preds_normalized * std_params + mean_params
    all_targets = all_targets_normalized * std_params + mean_params


    # Calculate metrics per parameter
    metrics = {}
    print("\n--- Evaluation Results ---")
    for i, name in enumerate(param_names):
        preds_param = all_preds[:, i]
        targets_param = all_targets[:, i]

        mse = mean_squared_error(targets_param, preds_param)
        mae = mean_absolute_error(targets_param, preds_param)
        r2 = r2_score(targets_param, preds_param)

        metrics[name] = {'mse': mse, 'mae': mae, 'r2': r2}
        print(f"{name}:")
        print(f"  MSE: {mse:.6e}")
        print(f"  MAE: {mae:.6e}")
        print(f"  R2:  {r2:.6f}")

    # Save metrics to file
    save_test_metrics_to_file(LOG_FILE_TEST, metrics)
    print(f"Test metrics saved to {LOG_FILE_TEST}")

    # Create and save plots
    ensure_dir_exists(output_dir)
    for i, name in enumerate(param_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets[:, i], all_preds[:, i], alpha=0.5, s=10)
        # Add identity line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.title(f"Predicted vs. True {name} (Test Set)")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"pred_vs_true_{name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    return metrics


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Physics-Augmented 3D ResNet.")
    parser.add_argument('--check', action='store_true', help='Perform data loading and sanity checks.')
    parser.add_argument('--train', action='store_true', help='Run training.')
    parser.add_argument('--eval', action='store_true', help='Run evaluation on the test set.')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if exists.')

    args = parser.parse_args()

    if not (args.check or args.train or args.eval):
        print("Please specify a mode: --check, --train, or --eval")
        sys.exit(1)

    # Ensure output directory exists for all modes
    ensure_dir_exists(OUTPUT_DIR)

    # Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    print("Loading data...")
    params_all = pd.read_csv(PARAMS_FILE, sep=' ', header=None, comment='#',
                             names=PARAM_NAMES).values.astype(np.float32)
    fields_all = np.load(FIELDS_FILE).astype(np.float32) # Shape (2000, 64, 64, 64)
    print(f"Parameters shape: {params_all.shape}")
    print(f"Density fields shape: {fields_all.shape}")

    # Train/Val/Test Split (fixed random state for reproducibility)
    indices = np.arange(N_SIMULATIONS)
    train_indices, test_indices = train_test_split(indices, test_size=TEST_SPLIT, random_state=SEED)
    train_indices, val_indices = train_test_split(train_indices, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), random_state=SEED)

    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # Apply split
    fields_train, fields_val, fields_test = fields_all[train_indices], fields_all[val_indices], fields_all[test_indices]
    params_train, params_val, params_test = params_all[train_indices], params_all[val_indices], params_all[test_indices]

    # --- Physics Feature Extraction (Pre-computation) ---
    print("Calculating physics features for all samples...")
    k_bins_log = np.logspace(np.log10(K_MIN_PHYS), np.log10(K_MAX_PHYS), N_PK_BINS + 1)
    physics_features_all = []
    log1p_fields_all = np.log1p(fields_all) # Calculate log1p once for PDF features

    start_phys_time = time.time()
    for i in range(N_SIMULATIONS):
        if (i + 1) % 200 == 0:
             print(f"  Processing sample {i+1}/{N_SIMULATIONS}...")
        features = calculate_physics_features(fields_all[i], log1p_fields_all[i],
                                             k_bins_log, BAO_K_RANGE, N_PDF_BINS)
        physics_features_all.append(features)
    physics_features_all = np.array(physics_features_all, dtype=np.float32)
    print(f"Physics features shape: {physics_features_all.shape}")
    print(f"Physics feature calculation took {time.time() - start_phys_time:.2f}s")
    num_phys_features = physics_features_all.shape[1]

    # Apply split to physics features
    phys_train, phys_val, phys_test = physics_features_all[train_indices], physics_features_all[val_indices], physics_features_all[test_indices]

    # --- Normalization (Fit on Training Data Only) ---
    print("Calculating normalization statistics from training data...")

    # Field normalization (pixel-wise mean/std over training log1p fields)
    # Need log1p fields for training set
    log1p_fields_train = np.log1p(fields_train)
    field_mean = np.mean(log1p_fields_train).astype(np.float32)
    field_std = np.std(log1p_fields_train).astype(np.float32)
    if field_std < 1e-9: field_std = 1.0 # Avoid division by zero if std is tiny
    field_scaler = {'mean': field_mean, 'std': field_std}
    print(f"Field Scaler: Mean={field_mean:.4f}, Std={field_std:.4f}")

    # Physics features normalization (feature-wise mean/std over training physics features)
    physics_scaler = StandardScaler()
    physics_scaler.fit(phys_train)
    physics_scaler_dict = {'mean': torch.from_numpy(physics_scaler.mean_.astype(np.float32)).float(),
                           'std': torch.from_numpy(np.sqrt(physics_scaler.var_).astype(np.float32)).float()}
    # Ensure std is not zero
    physics_scaler_dict['std'][physics_scaler_dict['std'] < 1e-9] = 1.0
    print(f"Physics Scaler: Mean shape={physics_scaler_dict['mean'].shape}, Std shape={physics_scaler_dict['std'].shape}")


    # Parameters normalization (parameter-wise mean/std over training parameters)
    params_scaler = StandardScaler()
    params_scaler.fit(params_train)
    params_scaler_dict = {'mean': torch.from_numpy(params_scaler.mean_.astype(np.float32)).float(),
                          'std': torch.from_numpy(np.sqrt(params_scaler.var_).astype(np.float32)).float()}
    # Ensure std is not zero
    params_scaler_dict['std'][params_scaler_dict['std'] < 1e-9] = 1.0
    print(f"Params Scaler: Mean={params_scaler_dict['mean'].numpy()}, Std={params_scaler_dict['std'].numpy()}")


    # --- Check Mode ---
    if args.check:
        print("\n--- Running Check Mode ---")
        print("Checking dataset instantiation and sample retrieval...")
        temp_dataset = CosmologyDataset(fields_train[:10], params_train[:10], phys_train[:10],
                                        field_scaler, physics_scaler_dict, params_scaler_dict,
                                        apply_augmentation=True)
        sample_field, sample_phys, sample_params = temp_dataset[0]
        print(f"Sample field shape: {sample_field.shape}, dtype: {sample_field.dtype}")
        print(f"Sample physics shape: {sample_phys.shape}, dtype: {sample_phys.dtype}")
        print(f"Sample params shape: {sample_params.shape}, dtype: {sample_params.dtype}")
        print("Checking normalization (first sample):")
        print(f"  Normalized field mean: {sample_field.mean():.4f}, std: {sample_field.std():.4f}") # Should be approx 0, 1 if single sample represents distribution well
        print(f"  Normalized physics mean: {sample_phys.mean():.4f}, std: {sample_phys.std():.4f}")
        print(f"  Normalized params: {sample_params.numpy()}")
        print("Check complete.")
        sys.exit(0)


    # --- Create Datasets and DataLoaders ---
    print("Creating Datasets and DataLoaders...")
    train_dataset = CosmologyDataset(fields_train, params_train, phys_train,
                                     field_scaler, physics_scaler_dict, params_scaler_dict,
                                     apply_augmentation=True)
    val_dataset = CosmologyDataset(fields_val, params_val, phys_val,
                                   field_scaler, physics_scaler_dict, params_scaler_dict,
                                   apply_augmentation=False)
    test_dataset = CosmologyDataset(fields_test, params_test, phys_test,
                                    field_scaler, physics_scaler_dict, params_scaler_dict,
                                    apply_augmentation=False)

    # Consider num_workers > 0 if I/O bound, but can cause issues on some systems
    num_workers = 2 if device == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if device=='cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if device=='cuda' else False)
    print("DataLoaders created.")

    # --- Model Initialization ---
    print("Initializing model...")
    # Make sure num_phys_features is correctly determined
    model = ResNet18_3D_Physics(num_phys_features=num_phys_features, num_params=N_PARAMS, use_se=True)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Loss Function and Optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=SCHEDULER_PATIENCE, verbose=True)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    if device == 'cuda': print("Using mixed precision training.")

    # --- Wandb Initialization ---
    print("Initializing Wandb...")
    try:
        import wandb
        # Check if API key is available
        if wandb.api.api_key is None:
             print("Warning: Wandb API key not found. Run 'wandb login' or set WANDB_API_KEY environment variable.")
             # Decide whether to proceed without logging or exit
             use_wandb = False
        else:
             use_wandb = True
             # Determine if resuming based on run ID existence
             api = wandb.Api()
             try:
                 # Check if the run exists
                 _ = api.run(f"{WANDB_PROJECT}/{WANDB_RUN_ID}")
                 resume_status = "must" if RESUME_WANDB else None
                 print(f"Wandb run {WANDB_RUN_ID} exists. Setting resume='{resume_status}'.")
             except wandb.errors.CommError:
                 # Run does not exist, create it anew
                 print(f"Wandb run {WANDB_RUN_ID} does not exist. Creating new run.")
                 resume_status = None # Cannot resume if it doesn't exist

             wandb.init(
                 project=WANDB_PROJECT,
                 id=WANDB_RUN_ID, # Use the specific ID
                 resume=resume_status, # 'must' requires it to exist, None creates if not found
                 config={
                     "learning_rate": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY,
                     "batch_size": BATCH_SIZE,
                     "epochs": N_EPOCHS,
                     "scheduler_patience": SCHEDULER_PATIENCE,
                     "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                     "seed": SEED,
                     "validation_split": VALIDATION_SPLIT,
                     "test_split": TEST_SPLIT,
                     "architecture": "PhysicsResNet18_3D_SE",
                     "num_phys_features": num_phys_features,
                     "n_pk_bins": N_PK_BINS,
                     "n_pdf_bins": N_PDF_BINS,
                     "augmentation": True,
                     "normalization": "ZScore",
                 }
             )
             # Save the code itself to wandb
             wandb.save(os.path.basename(__file__))

    except ImportError:
        print("Wandb not installed. Skipping Wandb logging.")
        use_wandb = False
    except Exception as e:
        print(f"Error initializing Wandb: {e}. Skipping Wandb logging.")
        use_wandb = False

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if args.resume or args.train: # Load if resuming or starting training (might resume implicitly)
         if os.path.exists(CHECKPOINT_PATH):
             print(f"Loading checkpoint from {CHECKPOINT_PATH}")
             try:
                 checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
                 model.load_state_dict(checkpoint['model_state_dict'])
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 start_epoch = checkpoint['epoch'] + 1
                 best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Use get for backward compatibility
                 epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                 # Load scheduler state if saved
                 if 'scheduler_state_dict' in checkpoint:
                       scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 # Load GradScaler state if saved and using cuda
                 if scaler and 'scaler_state_dict' in checkpoint:
                      scaler.load_state_dict(checkpoint['scaler_state_dict'])

                 print(f"Resumed from epoch {start_epoch}. Best val loss so far: {best_val_loss:.6f}")
             except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                 start_epoch = 0
                 best_val_loss = float('inf')
                 epochs_no_improve = 0
                 # Re-initialize optimizer and scheduler if loading failed
                 optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                 scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=SCHEDULER_PATIENCE, verbose=True)
                 if scaler: scaler = torch.cuda.amp.GradScaler()

         else:
              print("No checkpoint found. Starting training from scratch.")


    # --- Training Loop ---
    if args.train:
        print("\n--- Starting Training ---")
        train_metrics_log = {}
        val_metrics_log = {}

        for epoch in range(start_epoch, N_EPOCHS):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")

            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss = validate_epoch(model, val_loader, criterion, device)

            epoch_duration = time.time() - epoch_start_time

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Duration:   {epoch_duration:.2f}s")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Logging
            train_metrics_log = {'train_loss': train_loss, 'lr': optimizer.param_groups[0]['lr']}
            val_metrics_log = {'val_loss': val_loss}
            if use_wandb:
                wandb.log({**train_metrics_log, **val_metrics_log}, step=epoch)

            # Append metrics to local files
            save_metrics_to_file(LOG_FILE_TRAIN, epoch, train_metrics_log)
            save_metrics_to_file(LOG_FILE_VAL, epoch, val_metrics_log)


            # Learning Rate Scheduling
            scheduler.step(val_loss)

            # Checkpoint Saving and Early Stopping
            is_best = val_loss < best_val_loss
            if is_best:
                print(f"Validation loss improved ({best_val_loss:.6f} --> {val_loss:.6f}). Saving best model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                epochs_no_improve = 0
                if use_wandb:
                    wandb.summary['best_val_loss'] = best_val_loss
                    # Also save as artifact potentially
                    # wandb.save(BEST_MODEL_PATH) # Saves current version, might overwrite
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve for {epochs_no_improve} epochs.")

            # Always save the latest checkpoint for resuming
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
            }
            if scaler:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, CHECKPOINT_PATH)
            #print(f"Checkpoint saved to {CHECKPOINT_PATH}") # Can be verbose

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print("Training finished.")
        # Load the best model for potential evaluation step afterwards
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation.")
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        else:
            print("Warning: Best model file not found. Evaluation will use the last state.")


    # --- Evaluation ---
    if args.eval:
        print("\n--- Starting Evaluation on Test Set ---")
        if not args.train: # If only evaluating, ensure the best model is loaded
             if os.path.exists(BEST_MODEL_PATH):
                 print(f"Loading best model from {BEST_MODEL_PATH}")
                 try:
                      model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
                 except Exception as e:
                      print(f"Error loading best model state_dict: {e}. Evaluation might fail or use initial weights.")
                      # Decide if to proceed or exit
                      # sys.exit(1) # Exit if best model is crucial and missing/corrupt
             else:
                 print("Warning: Best model file not found. Evaluating with potentially untrained model.")
                 # Decide if to proceed or exit
                 # sys.exit(1) # Exit if evaluation requires a trained model

        test_metrics = evaluate(model, test_loader, criterion, params_scaler_dict, device, PARAM_NAMES, OUTPUT_DIR)

        # Log final test metrics to wandb if enabled
        if use_wandb:
            wandb_test_metrics = {}
            for param, metrics in test_metrics.items():
                for metric_name, value in metrics.items():
                    wandb_test_metrics[f"test_{param}_{metric_name}"] = value
            # Calculate and log average test metrics
            avg_mse = np.mean([m['mse'] for m in test_metrics.values()])
            avg_mae = np.mean([m['mae'] for m in test_metrics.values()])
            avg_r2 = np.mean([m['r2'] for m in test_metrics.values()])
            wandb_test_metrics["test_avg_mse"] = avg_mse
            wandb_test_metrics["test_avg_mae"] = avg_mae
            wandb_test_metrics["test_avg_r2"] = avg_r2

            wandb.log(wandb_test_metrics)
            wandb.summary.update(wandb_test_metrics) # Add to summary as well

        print("Evaluation finished.")

    # --- Finish Wandb Run ---
    if use_wandb and (args.train or args.eval):
        print("Finishing Wandb run...")
        wandb.finish()

    print("Script execution completed.")