import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

import os
import gdown

MODEL_PATH = "model/saved_model.pth"
DRIVE_URL = "https://drive.google.com/uc?id=1CkqoV-O_pN0hQApX4xnlQ2k8LZwse48O"

def get_model():
    """Download the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?export=download&id=1CkqoV-O_pN0hQApX4xnlQ2k8LZwse48O",
            MODEL_PATH,
            quiet=False,
            use_cookies=False
        )
    return MODEL_PATH
trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

# load the pretrained ResNet Model

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    if trained_model is None:
        trained_model = CarClassifierResNet()
        state_dict = torch.load(get_model(), map_location=device)  # ✅ downloads if missing
        trained_model.load_state_dict(state_dict)
        trained_model.to(device)
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]

