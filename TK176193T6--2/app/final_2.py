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

# Image transformations for classification and relevancy
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

# Load the trained classification model for tumor/no-tumor detection
classification_model = MobileNetModel(num_classes=2)
classification_model.load_state_dict(torch.load("mobilenet.pt"))
classification_model = classification_model.to(device)
classification_model.eval()

# Load the trained relevancy detection model
relevancy_model = MobileNetModel(num_classes=2)
relevancy_model.load_state_dict(torch.load("mobilenet_irrelevent.pt"))
relevancy_model = relevancy_model.to(device)
relevancy_model.eval()

# Load the segmentation model for tumor detection
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

# Predict image relevance
def predict_relevance(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = relevancy_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Tumor detection in relevant images
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = classification_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Preprocess image for segmentation
def preprocess_image_for_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Add channel dimension (1 channel for grayscale)
    return torch.tensor(image).to(device)

# Predict tumor segmentation
def predict_segmentation(image_path):
    input_image = preprocess_image_for_segmentation(image_path)

    with torch.no_grad():  # Disable gradient calculation
        output_logits = segmentation_model(input_image)
    
    output_prob = torch.sigmoid(output_logits)
    prediction = (output_prob > 0.5).float()  # Convert to binary mask

    return prediction.squeeze().cpu().numpy()  # Move back to CPU and remove unnecessary dimensions

# # Example usage
# image_path = r'relevent and irrelevent\irrelevent\001_2jP9N_ipAo8.jpg'

# # Step 1: Check if the image is relevant
# relevance_prediction = predict_relevance(image_path)
# if relevance_prediction == 1:  # Irrelevant image
#     print("Image is irrelevant. No further analysis.")
# else:  # Relevant image, proceed with tumor prediction
#     print("Image is relevant. Proceeding with tumor detection...")
    
#     # Step 2: Classify the image for tumor presence
#     tumor_prediction = predict_image(image_path)
#     if tumor_prediction == 1:  # Tumor detected
#         print("Tumor detected. Proceeding to segmentation...")
#         predicted_mask = predict_segmentation(image_path)

#         # Visualize the input image and predicted mask
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
#         plt.title('Input Image')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(predicted_mask, cmap='gray')
#         plt.title('Predicted Mask')
#         plt.axis('off')

#         plt.show()
#     else:
#         print("No tumor detected. No segmentation performed.")
