# check.py - GPU-OPTIMIZED VERSION FOR ENHANCED FORGERY DETECTION
# Specifically optimized for NVIDIA GPU training with CUDA
# Features: Mixed Precision (AMP), Gradient Accumulation, CUDA optimizations
# Date: October 2025

import json
import random
import time
from pathlib import Path

import torch
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader
from PIL import UnidentifiedImageError
from torch.cuda.amp import autocast, GradScaler

# ========================================================================================
# GPU OPTIMIZATION CHECK
# ========================================================================================

def check_gpu_availability():
    """Check GPU availability and print detailed information"""
    print("="*80)
    print("🔍 GPU AVAILABILITY CHECK")
    print("="*80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Please install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False

    print(f"✅ CUDA is available!")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    print()

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"🖥️  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   Available Memory: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB")
        print(f"   Multi-Processor Count: {props.multi_processor_count}")
        print()

    return True

# ========================================================================================
# REPRODUCIBILITY
# ========================================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set True for speed if input size is fixed

# ========================================================================================
# SAFE IMAGE LOADER
# ========================================================================================

def safe_loader(path):
    """Safely load images, handling corrupted files"""
    try:
        img = default_loader(path)
        return img.convert('RGB')
    except (UnidentifiedImageError, OSError, Exception):
        return None

# ========================================================================================
# CUSTOM AUGMENTATION CLASSES
# ========================================================================================

class AddGaussianNoise:
    """Add Gaussian noise to images during training"""
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class AddSaltPepperNoise:
    """Add salt and pepper noise to images during training"""
    def __init__(self, prob=0.02):
        self.prob = prob

    def __call__(self, tensor):
        noise_tensor = torch.rand(tensor.size())
        salt = noise_tensor < self.prob / 2
        pepper = noise_tensor > 1 - self.prob / 2
        tensor[salt] = 1.0
        tensor[pepper] = 0.0
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"

# ========================================================================================
# ENHANCED CNN MODEL ARCHITECTURE (GPU-OPTIMIZED)
# ========================================================================================

class EnhancedCNN(nn.Module):
    """
    Enhanced CNN architecture for image forgery detection
    GPU-optimized with efficient memory usage and CUDA operations
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # Feature Extraction Backbone
        self.fe = nn.Sequential(
            # Block 1: 64 filters
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Block 2: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Block 3: 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),

            # Block 4: 512 filters
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=0.3)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.fe(x)
        x = self.classifier(x)
        return x

# ========================================================================================
# FOCAL LOSS FOR HARD EXAMPLE MINING
# ========================================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ========================================================================================
# DATA TRANSFORMS
# ========================================================================================

def get_train_transforms(img_size=224):
    """Enhanced training augmentations"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.RandomApply([AddGaussianNoise(mean=0., std=0.05)], p=0.3),
        T.RandomApply([AddSaltPepperNoise(prob=0.02)], p=0.3),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=224):
    """Validation transforms"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ========================================================================================
# GPU-OPTIMIZED TRAINING FUNCTIONS WITH MIXED PRECISION
# ========================================================================================

def train_one_epoch_gpu(model, loader, optimizer, criterion, device, scaler, 
                        grad_accumulation_steps=1):
    """
    GPU-optimized training with mixed precision (AMP) and gradient accumulation

    Args:
        scaler: GradScaler for automatic mixed precision
        grad_accumulation_steps: Accumulate gradients over multiple batches
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accumulation_steps  # Normalize loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights after accumulating gradients
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Statistics (no gradient tracking needed)
        with torch.no_grad():
            running_loss += loss.item() * grad_accumulation_steps
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_gpu(model, loader, criterion, device):
    """GPU-optimized validation with mixed precision"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Mixed precision inference
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def compute_confusion_matrix(model, loader, device, num_classes):
    """Compute confusion matrix and per-class metrics"""
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # Calculate per-class metrics
    precision = []
    recall = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precision.append(prec)
        recall.append(rec)

    return confusion_matrix, precision, recall

# ========================================================================================
# MAIN TRAINING SCRIPT (GPU-OPTIMIZED)
# ========================================================================================

def main():
    """Main GPU-optimized training pipeline"""

    print("="*80)
    print("🚀 GPU-OPTIMIZED IMAGE FORGERY DETECTION - TRAINING PIPELINE")
    print("="*80)
    print()

    # Check GPU availability
    if not check_gpu_availability():
        print("Exiting: GPU not available")
        return

    # Set random seed
    set_seed(42)

    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune for best performance

    # ==================================================================================
    # CONFIGURATION (GPU-OPTIMIZED)
    # ==================================================================================

    DATA_PATH = Path("data/patches")
    OUTPUT_PATH = Path("output/pre_trained_cnn")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (optimized for GPU)
    IMG_SIZE = 224
    BATCH_SIZE = 64          # Increased for GPU (can go higher: 128, 256)
    EPOCHS = 50
    LEARNING_RATE = 0.0002   # Higher LR with larger batch size
    PATIENCE = 10
    GRAD_ACCUMULATION = 2    # Effective batch size = 64 * 2 = 128
    NUM_WORKERS = 4          # Parallel data loading (adjust based on CPU cores)

    # Device configuration
    device = torch.device("cuda")
    torch.cuda.empty_cache()  # Clear GPU cache

    print("⚙️  GPU Training Configuration:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Gradient Accumulation Steps: {GRAD_ACCUMULATION}")
    print(f"   Effective Batch Size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Mixed Precision: Enabled (AMP)")
    print(f"   cuDNN Benchmark: Enabled")
    print(f"   Data Loading Workers: {NUM_WORKERS}")
    print()

    # ==================================================================================
    # LOAD DATASET
    # ==================================================================================

    print("📁 Loading dataset...")
    train_tfm = get_train_transforms(IMG_SIZE)
    val_tfm = get_val_transforms(IMG_SIZE)

    full_dataset = tv.datasets.ImageFolder(
        str(DATA_PATH), 
        transform=None, 
        loader=safe_loader
    )

    print("   Filtering corrupted images...")
    valid_samples = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for i, (path, label) in enumerate(full_dataset.samples):
    # Check file extension
      if path.lower().endswith(supported_formats):
        # Try to load it
        if safe_loader(path) is not None:
            valid_samples.append((path, label))


    full_dataset.samples = valid_samples
    full_dataset.imgs = valid_samples

    print(f"   Total valid images: {len(full_dataset)}")
    print(f"   Classes: {full_dataset.classes}")
    print(f"   Class distribution:", end=" ")
    for i, cls in enumerate(full_dataset.classes):
        count = sum(1 for _, label in full_dataset.samples if label == i)
        print(f"{cls}={count}", end="  ")
    print("\n")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_tfm
    val_dataset.dataset.transform = val_tfm

    # Create data loaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,      # Faster GPU transfer
        persistent_workers=True if NUM_WORKERS > 0 else False  # Keep workers alive
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    print(f"📊 Dataset split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print()

    # ==================================================================================
    # INITIALIZE MODEL (GPU)
    # ==================================================================================

    print("🧠 Initializing model on GPU...")
    num_classes = len(full_dataset.classes)
    model = EnhancedCNN(num_classes=num_classes).to(device)

    # Optional: Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"   Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Architecture: EnhancedCNN")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")
    print()

    # ==================================================================================
    # LOSS, OPTIMIZER, SCHEDULER, MIXED PRECISION
    # ==================================================================================

    print("⚙️  Configuring GPU training...")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"   Loss function: Focal Loss")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4
    )
    print(f"   Optimizer: Adam (lr={LEARNING_RATE})")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=5,
        min_lr=1e-6
    )
    print(f"   LR Scheduler: ReduceLROnPlateau")

    # Mixed Precision Scaler
    scaler = GradScaler()
    print(f"   Mixed Precision: GradScaler enabled")
    print(f"   Early stopping patience: {PATIENCE} epochs")
    print()

    # ==================================================================================
    # TRAINING LOOP (GPU-OPTIMIZED)
    # ==================================================================================

    print("="*80)
    print("🚀 STARTING GPU TRAINING")
    print("="*80)
    print()

    best_val_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train with GPU optimizations
        train_loss, train_acc = train_one_epoch_gpu(
            model, train_loader, optimizer, criterion, device, scaler, GRAD_ACCUMULATION
        )

        # Validate
        val_loss, val_acc = validate_gpu(model, val_loader, criterion, device)

        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # GPU memory stats
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:6.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"GPU Mem: {gpu_memory_allocated:.2f}GB | "
              f"Time: {epoch_time:.1f}s")

        if current_lr < old_lr:
            print(f"   📉 Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            cm, precision, recall = compute_confusion_matrix(
                model, val_loader, device, num_classes
            )

            # Save model (handle DataParallel)
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            checkpoint = {
                'epoch': epoch + 1,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict(),
                'classes': full_dataset.classes,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'img_size': IMG_SIZE,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'confusion_matrix': cm.tolist(),
                'precision': precision,
                'recall': recall
            }

            torch.save(checkpoint, OUTPUT_PATH / "enhanced_cnn_best.pt")

            print(f"   ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
            print(f"      Precision: {[f'{p:.3f}' for p in precision]}")
            print(f"      Recall: {[f'{r:.3f}' for r in recall]}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break

        print()

        # Clear GPU cache periodically
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # ==================================================================================
    # TRAINING COMPLETE
    # ==================================================================================

    total_time = time.time() - start_time

    print("="*80)
    print("✨ GPU TRAINING COMPLETE")
    print("="*80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {total_time/len(training_history['train_loss']):.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {OUTPUT_PATH / 'enhanced_cnn_best.pt'}")
    print()

    # Save training history
    with open(OUTPUT_PATH / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save metadata
    metadata = {
        'checkpoint_path': str(OUTPUT_PATH / "enhanced_cnn_best.pt"),
        'classes': full_dataset.classes,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'img_size': IMG_SIZE,
        'best_val_acc': best_val_acc,
        'architecture': 'EnhancedCNN',
        'total_params': total_params,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'grad_accumulation': GRAD_ACCUMULATION,
            'mixed_precision': True,
            'num_workers': NUM_WORKERS
        }
    }

    with open(OUTPUT_PATH / "inference_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_PATH / "classes.json", 'w') as f:
        json.dump(full_dataset.classes, f)

    print("📄 Saved metadata files:")
    print(f"   - enhanced_cnn_best.pt")
    print(f"   - inference_meta.json")
    print(f"   - classes.json")
    print(f"   - training_history.json")
    print()

    # Final GPU stats
    print("📊 Final GPU Statistics:")
    print(f"   Peak memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    print(f"   Peak memory reserved: {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")
    print()
    print("="*80)
    print("🎉 Ready for inference! Use prediction.py or ui.py to test the model.")
    print("="*80)


if __name__ == "__main__":
    main()
