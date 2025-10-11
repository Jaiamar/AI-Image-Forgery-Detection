# prediction.py - Updated for PyTorch Model

import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torch import nn

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=0.3)
        )
        
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
        x = self.fe(x)
        x = self.classifier(x)
        return x

def predict_result(image_path):
    """Predict if image is authentic or forged"""
    try:
        # Handle tuple input from UI
        if isinstance(image_path, tuple):
            image_path = image_path[0]
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint_path = "output/pre_trained_cnn/enhanced_cnn_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        num_classes = len(checkpoint['classes'])
        model = EnhancedCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        
        # Get parameters
        mean = checkpoint.get('mean', [0.485, 0.456, 0.406])
        std = checkpoint.get('std', [0.229, 0.224, 0.225])
        img_size = checkpoint.get('img_size', 224)
        classes = checkpoint['classes']
        
        # Transform
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get label
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Map to standard labels
        if predicted_class in ['fake_images', '1', 'forged', 'Forged']:
            label = "Forged"
        else:
            label = "Authentic"
        
        # Format confidence
        confidence_str = f"{confidence_score * 100:.2f}"
        
        return (label, confidence_str)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return ("Error", "0.00")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        label, conf = predict_result(sys.argv[1])
        print(f"Result: {label} (Confidence: {conf}%)")
