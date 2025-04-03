import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import timm
import wandb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

# Configuration
CONFIG = {
    "SEED": 42,
    "TRAIN_IMAGE_DIR": "data/images_training/",
    "TEST_IMAGE_DIR": "data/images_test/",
    "LABEL_FILE": "data/training_solutions.csv",
    "OUTPUT_DIR": "out",
    "MODEL_NAME": "convnext_base", # Using ConvNeXt as Model A (CNN)
    "CROP_SIZE": 224, # Standard for ConvNeXt-Base
    "MULTI_RES_RANGE": [256, 456], # Random resize shorter edge to this range before crop
    "BATCH_SIZE": 32, # Adjust based on GPU memory
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "NUM_EPOCHS": 50, # Set sufficiently high, rely on early stopping
    "PATIENCE": 7, # Early stopping patience
    "TTA_RESOLUTIONS": [256, 384, 456], # Resolutions for TTA
    "TTA_ROTATIONS": [0, 90, 180, 270],
    "TTA_FLIPS": ["none", "horizontal", "vertical"], # Added vertical flip to TTA plan
    "WANDB_PROJECT": "galaxy-zoo",
    "WANDB_RUN_ID": "b57f2df5",
    "VAL_SPLIT": 0.15, # Validation set size
    "NUM_TARGETS": 37,
}

# Ensure output directory exists
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

BEST_MODEL_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "best_model.pth")
CHECKPOINT_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "checkpoint.pth")
PREDICTIONS_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "predictions.csv")
TRAIN_METRICS_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "train_metrics.txt")
VAL_METRICS_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "val_metrics.txt")
PLOT_LOSS_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "loss_curve.png")
PLOT_RMSE_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "rmse_curve.png")

# --- Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG["SEED"])

def worker_init_fn(worker_id):
    """Ensure dataloader workers have unique seeds."""
    np.random.seed(CONFIG["SEED"] + worker_id)
    random.seed(CONFIG["SEED"] + worker_id)

# --- Custom Transform for Multi-Resolution Training ---
class RandomResizeBeforeCrop:
    def __init__(self, resize_range, interpolation=transforms.InterpolationMode.BILINEAR):
        self.resize_range = resize_range
        self.interpolation = interpolation

    def __call__(self, img):
        target_size = random.randint(self.resize_range[0], self.resize_range[1])
        # Resize shorter edge to target_size, maintaining aspect ratio
        resized_img = transforms.functional.resize(img, target_size, self.interpolation)
        return resized_img

# --- Dataset ---
class GalaxyDataset(Dataset):
    def __init__(self, image_dir, df_labels, image_ids, transform=None, is_test=False):
        self.image_dir = image_dir
        self.df_labels = df_labels
        self.image_ids = image_ids
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            self.labels = {row['GalaxyID']: row.iloc[1:].values.astype(np.float32)
                           for _, row in df_labels.iterrows()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            # Ensure image is loaded correctly, even if truncated
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and potentially flag this ID
            img = Image.new('RGB', (CONFIG["CROP_SIZE"], CONFIG["CROP_SIZE"]), color = 'red')
            # If training/val, we still need a label size match
            if not self.is_test:
                 label = torch.zeros(CONFIG["NUM_TARGETS"], dtype=torch.float32)
            else:
                 label = torch.zeros(1) # Placeholder for test

        if not self.is_test:
            label = self.labels[image_id]
            label = torch.from_numpy(label)

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, image_id
        else:
            return img, label

# --- Transforms ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pad to Crop Size Function (used after random resize if image < crop size)
def pad_if_needed(img, min_size):
    w, h = img.size
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        # Calculate padding (left, top, right, bottom)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
    return img


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([
         transforms.RandomRotation((0, 0), interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.RandomRotation((90, 90), interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.RandomRotation((180, 180), interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.RandomRotation((270, 270), interpolation=transforms.InterpolationMode.BILINEAR)
    ]),
    # --- Multi-Resolution Step ---
    RandomResizeBeforeCrop(CONFIG["MULTI_RES_RANGE"]),
    # Pad if the randomly resized image is smaller than crop size in any dimension
    transforms.Lambda(lambda img: pad_if_needed(img, CONFIG["CROP_SIZE"])),
    transforms.RandomCrop(CONFIG["CROP_SIZE"]),
    # --- End Multi-Resolution Specific ---
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation/Test transform (Resize slightly larger than crop, then center crop)
# Intermediate size for val/test resize before center cropping
_val_test_intermediate_size = int(CONFIG["CROP_SIZE"] * 1.14) # Approx 256 for 224 crop

val_transform = transforms.Compose([
    transforms.Resize(_val_test_intermediate_size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(CONFIG["CROP_SIZE"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# TTA specific transform generation
def get_tta_transform(resolution, rotation, flip):
    transforms_list = []
    # 1. Resize
    transforms_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))
    # 2. Pad if needed before rotation/flip/crop (if resize < crop_size)
    transforms_list.append(transforms.Lambda(lambda img: pad_if_needed(img, CONFIG["CROP_SIZE"])))
    # 3. Rotation
    if rotation != 0:
        transforms_list.append(transforms.functional.rotate) # Use functional rotate with angle param later
    # 4. Flip
    if flip == "horizontal":
        transforms_list.append(transforms.functional.hflip)
    elif flip == "vertical":
        transforms_list.append(transforms.functional.vflip)
    # 5. Center Crop
    transforms_list.append(transforms.CenterCrop(CONFIG["CROP_SIZE"]))
    # 6. ToTensor & Normalize
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    # Need a wrapper function because functional transforms need img argument
    def apply_transforms(img):
        x = img
        for t in transforms_list:
            if t == transforms.functional.rotate:
                 x = transforms.functional.rotate(x, angle=rotation, interpolation=transforms.InterpolationMode.BILINEAR)
            elif t == transforms.functional.hflip:
                 x = transforms.functional.hflip(x)
            elif t == transforms.functional.vflip:
                 x = transforms.functional.vflip(x)
            else: # Apply Compose-style transforms
                 x = t(x)
        return x

    return apply_transforms


# --- Model ---
def get_model(model_name, pretrained=True, num_classes=CONFIG["NUM_TARGETS"]):
    model = timm.create_model(model_name, pretrained=pretrained)

    # Replace the classifier head
    if hasattr(model, 'head') and hasattr(model.head, 'fc'): # Specific check for ConvNeXt structure
         num_ftrs = model.head.fc.in_features
         model.head.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier'): # Common for EfficientNet, etc.
         num_ftrs = model.classifier.in_features
         model.classifier = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'head'): # Common for ViT
         num_ftrs = model.head.in_features
         model.head = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'fc'): # Common for ResNet
         num_ftrs = model.fc.in_features
         model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise NotImplementedError(f"Classifier replacement logic not defined for model type: {model_name}")

    return model

# --- Loss Function ---
# Using MSE as per plan (simpler than BetaNLL for this implementation)
criterion = nn.MSELoss()

def calculate_rmse(outputs, targets):
    """Calculates RMSE from MSE loss."""
    mse = criterion(outputs, targets)
    # Check for NaN loss immediately
    if torch.isnan(mse):
        raise ValueError("NaN loss detected during RMSE calculation!")
    return torch.sqrt(mse)

# --- Training Function ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, device):
    wandb.watch(model, log="all", log_freq=100)

    scaler = GradScaler()
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    history = {'train_loss': [], 'train_rmse': [], 'val_loss': [], 'val_rmse': []}

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint['best_val_rmse']
        history = checkpoint['history']
        # Load GradScaler state if saved (optional but good practice)
        if 'scaler_state_dict' in checkpoint:
           scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed from Epoch {start_epoch}, Best Val RMSE: {best_val_rmse:.6f}")
    else:
        print("No checkpoint found, starting training from scratch.")


    for epoch in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        model.train()
        train_loss_epoch = 0.0
        train_rmse_epoch = 0.0
        processed_samples = 0

        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast(): # Mixed precision context
                outputs = model(images)
                # Clamp output before loss calculation? Usually done before *metric* calculation.
                # Let's calculate loss on raw outputs for gradient flow.
                loss = criterion(outputs, targets)

            # Check for NaN loss *before* backward pass
            if torch.isnan(loss):
                raise ValueError(f"NaN loss detected during training epoch {epoch}, batch {i}!")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Clamp for metric calculation
            outputs_clamped = torch.clamp(outputs.detach(), 0.0, 1.0)
            rmse = calculate_rmse(outputs_clamped, targets)

            batch_size = images.size(0)
            train_loss_epoch += loss.item() * batch_size
            train_rmse_epoch += rmse.item() * batch_size
            processed_samples += batch_size

            if (i + 1) % 100 == 0:
                 print(f"Epoch [{epoch+1}/{CONFIG['NUM_EPOCHS']}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.6f}, Batch RMSE: {rmse.item():.6f}")

        train_loss_avg = train_loss_epoch / processed_samples
        train_rmse_avg = train_rmse_epoch / processed_samples
        history['train_loss'].append(train_loss_avg)
        history['train_rmse'].append(train_rmse_avg)

        # --- Validation ---
        val_loss, val_rmse = validate_model(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        print(f"Epoch [{epoch+1}/{CONFIG['NUM_EPOCHS']}] Train Loss: {train_loss_avg:.6f}, Train RMSE: {train_rmse_avg:.6f} | Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.6f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "train_rmse": train_rmse_avg,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "learning_rate": optimizer.param_groups[0]['lr'] # Log learning rate
        })

        # --- Checkpoint and Early Stopping ---
        scheduler.step(val_rmse) # ReduceLROnPlateau monitors validation RMSE

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            print(f"New best model found! Val RMSE: {best_val_rmse:.6f}. Saving model to {BEST_MODEL_PATH}")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation RMSE did not improve for {epochs_no_improve} epoch(s). Best: {best_val_rmse:.6f}")

        # Save checkpoint regularly
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'history': history,
            'scaler_state_dict': scaler.state_dict() # Save scaler state
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        # print(f"Checkpoint saved to {CHECKPOINT_PATH}") # Optional: uncomment for verbosity

        if epochs_no_improve >= CONFIG["PATIENCE"]:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation RMSE: {best_val_rmse:.6f}")
    # Load the best model weights at the end
    print(f"Loading best model from {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    return model, history

# --- Validation Function ---
def validate_model(model, val_loader, device):
    model.eval()
    val_loss_epoch = 0.0
    val_rmse_epoch = 0.0
    processed_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            with autocast(): # Use autocast even in validation if model was trained with it
                outputs = model(images)
                # Clamp output *before* calculating metrics
                outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
                loss = criterion(outputs_clamped, targets) # Use clamped for consistency with RMSE calc

            # Check for NaN loss
            if torch.isnan(loss):
                 print(f"Warning: NaN loss detected during validation!") # Don't raise error, just warn
                 # Handle appropriately, maybe skip batch metric calculation or return inf
                 return float('inf'), float('inf')

            rmse = calculate_rmse(outputs_clamped, targets)

            batch_size = images.size(0)
            val_loss_epoch += loss.item() * batch_size
            val_rmse_epoch += rmse.item() * batch_size
            processed_samples += batch_size

    val_loss_avg = val_loss_epoch / processed_samples
    val_rmse_avg = val_rmse_epoch / processed_samples
    return val_loss_avg, val_rmse_avg


# --- Evaluation Function (using TTA on Validation Set) ---
def evaluate_model(model, val_loader, device):
    print("\nStarting evaluation on validation set with TTA...")
    model.eval()
    all_targets = []
    all_tta_preds = [] # Store final TTA-averaged preds per image

    original_images = {} # Store original PIL images by index for TTA
    targets_map = {}     # Store targets by index

    print("Collecting original images and targets...")
    with torch.no_grad():
        # First pass: collect raw images and targets
        temp_dataset = val_loader.dataset
        temp_dataset.transform = None # Temporarily remove transform to get PIL image
        for idx in range(len(temp_dataset)):
            pil_img, target = temp_dataset[idx]
            original_images[idx] = pil_img
            targets_map[idx] = target.numpy() # Store as numpy array

    print(f"Collected {len(original_images)} images.")

    # TTA loop
    num_images = len(original_images)
    for idx in range(num_images):
        if (idx + 1) % 100 == 0:
             print(f"Processing TTA for image {idx+1}/{num_images}")
        
        pil_img = original_images[idx]
        target_np = targets_map[idx]
        
        tta_predictions_for_image = []

        for res in CONFIG["TTA_RESOLUTIONS"]:
            for rot in CONFIG["TTA_ROTATIONS"]:
                for flip in CONFIG["TTA_FLIPS"]:
                    # Get the specific transform function for this TTA combination
                    transform_func = get_tta_transform(res, rot, flip)
                    # Apply transform to the original PIL image
                    img_tensor = transform_func(pil_img)
                    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim and move to device

                    with torch.no_grad():
                         with autocast():
                             output = model(img_tensor)
                    
                    # Clamp prediction and move to CPU
                    pred_cpu = torch.clamp(output.detach(), 0.0, 1.0).cpu().squeeze(0)
                    tta_predictions_for_image.append(pred_cpu)

        # Average TTA predictions for this image
        if tta_predictions_for_image:
            mean_prediction = torch.stack(tta_predictions_for_image).mean(dim=0)
            final_pred = torch.clamp(mean_prediction, 0.0, 1.0) # Final clamp after averaging
            all_tta_preds.append(final_pred.numpy())
            all_targets.append(target_np)
        else:
            print(f"Warning: No TTA predictions generated for image index {idx}")


    if not all_tta_preds:
        print("Error: No TTA predictions were collected during evaluation.")
        return

    # Calculate final metrics
    all_tta_preds = np.array(all_tta_preds)
    all_targets = np.array(all_targets)

    final_mse = np.mean((all_tta_preds - all_targets)**2)
    final_rmse = np.sqrt(final_mse)

    print("\n--- Evaluation Results (TTA on Validation Set) ---")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Final RMSE: {final_rmse:.6f}")
    print("---------------------------------------------------")

    # Save metrics to file
    with open(VAL_METRICS_PATH, 'w') as f:
        f.write(f"Validation Set Evaluation (with TTA)\n")
        f.write(f"Final MSE: {final_mse:.6f}\n")
        f.write(f"Final RMSE: {final_rmse:.6f}\n")
    print(f"Validation metrics saved to {VAL_METRICS_PATH}")

    # --- Plotting ---
    # Load training history if available (saved during training)
    if os.path.exists(CHECKPOINT_PATH):
        try:
             checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu') # Load to CPU for safety
             history = checkpoint.get('history')

             if history:
                 epochs = range(1, len(history['train_loss']) + 1)

                 # Save training metrics
                 with open(TRAIN_METRICS_PATH, 'w') as f:
                     f.write("Epoch,Train Loss,Train RMSE,Val Loss,Val RMSE\n")
                     for i in range(len(epochs)):
                         f.write(f"{epochs[i]},{history['train_loss'][i]:.6f},{history['train_rmse'][i]:.6f},{history['val_loss'][i]:.6f},{history['val_rmse'][i]:.6f}\n")
                 print(f"Training metrics saved to {TRAIN_METRICS_PATH}")


                 # Plot Loss
                 plt.figure(figsize=(10, 5))
                 plt.plot(epochs, history['train_loss'], label='Training Loss')
                 plt.plot(epochs, history['val_loss'], label='Validation Loss')
                 plt.title('Training and Validation Loss')
                 plt.xlabel('Epochs')
                 plt.ylabel('Loss (MSE)')
                 plt.legend()
                 plt.grid(True)
                 plt.savefig(PLOT_LOSS_PATH)
                 plt.close()
                 print(f"Loss curve saved to {PLOT_LOSS_PATH}")

                 # Plot RMSE
                 plt.figure(figsize=(10, 5))
                 plt.plot(epochs, history['train_rmse'], label='Training RMSE')
                 plt.plot(epochs, history['val_rmse'], label='Validation RMSE')
                 plt.title('Training and Validation RMSE')
                 plt.xlabel('Epochs')
                 plt.ylabel('RMSE')
                 plt.legend()
                 plt.grid(True)
                 plt.savefig(PLOT_RMSE_PATH)
                 plt.close()
                 print(f"RMSE curve saved to {PLOT_RMSE_PATH}")

             else:
                 print("Could not find training history in checkpoint for plotting.")

        except Exception as e:
            print(f"Could not load or process checkpoint for plotting: {e}")

    # Additional Plots (Example: Scatter plot of prediction vs actual for first target)
    if len(all_tta_preds) > 0:
         plt.figure(figsize=(8, 8))
         plt.scatter(all_targets[:, 0], all_tta_preds[:, 0], alpha=0.3, s=10)
         plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Identity line
         plt.title('Prediction vs Actual (Target 0)')
         plt.xlabel('Actual Probability')
         plt.ylabel('Predicted Probability (TTA)')
         plt.grid(True)
         plt.xlim(0, 1)
         plt.ylim(0, 1)
         plt.gca().set_aspect('equal', adjustable='box')
         plot_scatter_path = os.path.join(CONFIG["OUTPUT_DIR"], "pred_vs_actual_target0.png")
         plt.savefig(plot_scatter_path)
         plt.close()
         print(f"Prediction vs Actual scatter plot saved to {plot_scatter_path}")


# --- Test Prediction Function (using TTA) ---
def predict_test(model, test_loader, device):
    print("\nStarting prediction on test set with TTA...")
    model.eval()
    results = {}

    original_images = {} # Store original PIL images by index for TTA

    print("Collecting original test images...")
    with torch.no_grad():
        # First pass: collect raw images
        temp_dataset = test_loader.dataset
        temp_dataset.transform = None # Temporarily remove transform to get PIL image
        for idx in range(len(temp_dataset)):
            pil_img, image_id = temp_dataset[idx]
            original_images[image_id] = pil_img # Store by image_id

    print(f"Collected {len(original_images)} test images.")

    # TTA loop for test set
    test_image_ids = list(original_images.keys())
    num_images = len(test_image_ids)
    
    for i, image_id in enumerate(test_image_ids):
        if (i + 1) % 500 == 0:
             print(f"Processing TTA for test image {i+1}/{num_images}")
        
        pil_img = original_images[image_id]
        tta_predictions_for_image = []

        for res in CONFIG["TTA_RESOLUTIONS"]:
            for rot in CONFIG["TTA_ROTATIONS"]:
                for flip in CONFIG["TTA_FLIPS"]:
                    # Get the specific transform function
                    transform_func = get_tta_transform(res, rot, flip)
                    # Apply transform
                    img_tensor = transform_func(pil_img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)

                    with torch.no_grad():
                        with autocast():
                            output = model(img_tensor)
                            
                    pred_cpu = torch.clamp(output.detach(), 0.0, 1.0).cpu().squeeze(0)
                    tta_predictions_for_image.append(pred_cpu)

        # Average TTA predictions
        if tta_predictions_for_image:
            mean_prediction = torch.stack(tta_predictions_for_image).mean(dim=0)
            final_pred = torch.clamp(mean_prediction, 0.0, 1.0) # Final clamp
            results[image_id] = final_pred.numpy()
        else:
             print(f"Warning: No TTA predictions generated for test image {image_id}")
             # Handle missing prediction - e.g., predict zeros or average? Let's use zeros.
             results[image_id] = np.zeros(CONFIG["NUM_TARGETS"], dtype=np.float32)


    # Format for submission
    print("Formatting predictions for submission...")
    # Get column names from the original training solutions file
    df_solution_example = pd.read_csv(CONFIG["LABEL_FILE"], nrows=1)
    column_names = df_solution_example.columns.tolist()

    submission_list = []
    # Ensure order matches the test loader's original order if possible, or sort by ID
    # Using the order from test_image_ids which came from original_images keys
    for image_id in test_image_ids:
        preds = results.get(image_id) # Get the numpy array
        if preds is not None:
            row = [image_id] + list(preds)
            submission_list.append(row)
        else:
            print(f"Error: Missing prediction for GalaxyID {image_id} in final results dict.")


    submission_df = pd.DataFrame(submission_list, columns=column_names)

    # Save submission file
    submission_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Galaxy Zoo Challenge - Plan 3 Implementation")
    parser.add_argument('--check', action='store_true', help='Run data loading check mode.')
    parser.add_argument('--train', action='store_true', help='Run training mode.')
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode on validation set.')
    parser.add_argument('--test', action='store_true', help='Run prediction mode on test set.')
    args = parser.parse_args()

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # --- Load Data ---
    print("Loading label data...")
    df_labels = pd.read_csv(CONFIG["LABEL_FILE"])
    all_image_ids = df_labels['GalaxyID'].tolist()
    print(f"Loaded labels for {len(all_image_ids)} images.")

    # --- Train/Validation Split ---
    print("Splitting data into training and validation sets...")
    train_ids, val_ids = train_test_split(
        all_image_ids,
        test_size=CONFIG["VAL_SPLIT"],
        random_state=CONFIG["SEED"]
    )
    print(f"Training set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")

    # --- Mode: Check Data ---
    if args.check:
        print("\n--- Check Mode ---")
        print("Checking training dataset...")
        check_train_dataset = GalaxyDataset(CONFIG["TRAIN_IMAGE_DIR"], df_labels, train_ids[:10], transform=train_transform)
        check_train_loader = DataLoader(check_train_dataset, batch_size=4, num_workers=0) # Use 0 workers for check
        img_batch, lbl_batch = next(iter(check_train_loader))
        print(f"Train Image Batch Shape: {img_batch.shape}")
        print(f"Train Label Batch Shape: {lbl_batch.shape}")
        print(f"Sample Label Tensor: {lbl_batch[0][:5]}...") # Print first 5 values of first label

        print("\nChecking validation dataset...")
        check_val_dataset = GalaxyDataset(CONFIG["TRAIN_IMAGE_DIR"], df_labels, val_ids[:10], transform=val_transform)
        check_val_loader = DataLoader(check_val_dataset, batch_size=4, num_workers=0)
        img_batch, lbl_batch = next(iter(check_val_loader))
        print(f"Val Image Batch Shape: {img_batch.shape}")
        print(f"Val Label Batch Shape: {lbl_batch.shape}")

        print("\nChecking test dataset...")
        # Need to list test image files to get IDs
        test_image_files = [f for f in os.listdir(CONFIG["TEST_IMAGE_DIR"]) if f.endswith('.jpg')]
        test_image_ids = [int(f.split('.')[0]) for f in test_image_files]
        check_test_dataset = GalaxyDataset(CONFIG["TEST_IMAGE_DIR"], None, test_image_ids[:10], transform=val_transform, is_test=True)
        check_test_loader = DataLoader(check_test_dataset, batch_size=4, num_workers=0)
        img_batch, id_batch = next(iter(check_test_loader))
        print(f"Test Image Batch Shape: {img_batch.shape}")
        print(f"Test ID Batch: {id_batch}")

        print("\nData check complete.")


    # --- Mode: Train ---
    elif args.train:
        print("\n--- Training Mode ---")

        # Setup WandB
        print("Initializing WandB...")
        try:
            wandb.init(
                project=CONFIG["WANDB_PROJECT"],
                id=CONFIG["WANDB_RUN_ID"],
                config=CONFIG,
                resume="allow" # Allow resuming if run ID exists
            )
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"Error initializing WandB: {e}. Check API key and login status.")
            # Decide whether to proceed without WandB or exit
            # For this script, we'll proceed without WandB logging if init fails
            wandb.init(mode="disabled") # Disable wandb if init fails
            print("Proceeding without WandB logging.")


        # Datasets and DataLoaders
        train_dataset = GalaxyDataset(CONFIG["TRAIN_IMAGE_DIR"], df_labels, train_ids, transform=train_transform)
        val_dataset = GalaxyDataset(CONFIG["TRAIN_IMAGE_DIR"], df_labels, val_ids, transform=val_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=True,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["BATCH_SIZE"] * 2, # Often use larger batch size for validation
            shuffle=False,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # Model, Optimizer, Scheduler
        model = get_model(CONFIG["MODEL_NAME"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=CONFIG["PATIENCE"] // 2, verbose=True) # Reduce LR faster than stopping

        # Train
        print("Starting training loop...")
        model, history = train_model(model, train_loader, val_loader, optimizer, scheduler, device)

        # Finish WandB run
        wandb.finish()
        print("Training complete.")

    # --- Mode: Evaluate ---
    elif args.eval:
        print("\n--- Evaluation Mode ---")
        # Load best model
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Best model file not found at {BEST_MODEL_PATH}. Please train the model first.")

        print(f"Loading best model from {BEST_MODEL_PATH}")
        model = get_model(CONFIG["MODEL_NAME"], pretrained=False).to(device) # Load architecture, set pretrained=False
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

        # Create validation dataset *without standard transform* for TTA
        # Note: The evaluate_model function will handle applying TTA transforms internally
        # We still need a loader to iterate through the validation IDs
        val_dataset_for_eval = GalaxyDataset(CONFIG["TRAIN_IMAGE_DIR"], df_labels, val_ids, transform=None) # No default transform here
        val_loader_for_eval = DataLoader( # Minimal loader just to get indices/data access
             val_dataset_for_eval,
             batch_size=1, # Process one image at a time for TTA evaluation logic
             shuffle=False,
             num_workers=CONFIG["NUM_WORKERS"], # Can use workers to load PIL images faster
             pin_memory=False # Not needed for PIL images
        )


        # Run evaluation with TTA
        evaluate_model(model, val_loader_for_eval, device) # Pass the loader designed for evaluation
        print("Evaluation complete.")


    # --- Mode: Test ---
    elif args.test:
        print("\n--- Test Prediction Mode ---")
        # Load best model
        if not os.path.exists(BEST_MODEL_PATH):
             raise FileNotFoundError(f"Best model file not found at {BEST_MODEL_PATH}. Please train the model first.")

        print(f"Loading best model from {BEST_MODEL_PATH}")
        model = get_model(CONFIG["MODEL_NAME"], pretrained=False).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

        # Create test dataset (again, without default transform initially)
        test_image_files = [f for f in os.listdir(CONFIG["TEST_IMAGE_DIR"]) if f.endswith('.jpg')]
        test_image_ids = sorted([int(f.split('.')[0]) for f in test_image_files]) # Sort IDs for consistent order
        print(f"Found {len(test_image_ids)} test images.")

        test_dataset = GalaxyDataset(CONFIG["TEST_IMAGE_DIR"], None, test_image_ids, transform=None, is_test=True)
        test_loader = DataLoader( # Minimal loader for accessing test images by ID
            test_dataset,
            batch_size=1, # Process one image at a time for TTA prediction logic
            shuffle=False,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=False
        )

        # Run prediction with TTA
        predict_test(model, test_loader, device)
        print("Test prediction complete.")


    # --- No Mode Specified ---
    else:
        print("Please specify a mode: --check, --train, --eval, or --test")
