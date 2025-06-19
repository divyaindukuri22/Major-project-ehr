import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for classification
image_transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the MobileNet model class (same as the one used during training)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the trained classification model
classification_model = MobileNetModel(num_classes=2)
classification_model.load_state_dict(torch.load("mobilenet.pt"))
classification_model = classification_model.to(device)
classification_model.eval()

# Load the segmentation model
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
segmentation_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=1,  # Input channels for grayscale images
    classes=1,      # Output classes
    activation=None
)
segmentation_model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
segmentation_model.to(device)
segmentation_model.eval()

def predict_image(image_path):
    # Load and preprocess the image for classification
    image = Image.open(image_path).convert('RGB')
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        output = classification_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

def preprocess_image_for_segmentation(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to match the model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Add channel dimension (1 channel for grayscale)
    return torch.tensor(image).to(device)

def predict_segmentation(image_path):
    input_image = preprocess_image_for_segmentation(image_path)
    
    with torch.no_grad():  # Disable gradient calculation
        output_logits = segmentation_model(input_image)
        
    # Apply sigmoid to get probabilities and threshold
    output_prob = torch.sigmoid(output_logits)
    prediction = (output_prob > 0.5).float()  # Convert to binary mask
    
    return prediction.squeeze().cpu().numpy()  # Move back to CPU and remove unnecessary dimensions
