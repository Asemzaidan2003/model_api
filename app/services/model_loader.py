import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models ,transforms
from app.utils.preprocess import load_image_from_url
import joblib

warnings.filterwarnings("ignore")

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ready labels and their number of classes
num_classes = {
    'gender': 5,
    'masterCategory': 3,
    'subCategory': 18,
    'articleType': 72,
    'baseColour': 47,
    'season': 5,
    'usage': 8
}

saved_models_dir = "saved_models"
ATTRIBUTES = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']

# Load models and encoders
models_dict = {}
encoders = {}
ready_labels = []
for label_name in num_classes.keys():
    model_path = os.path.join(saved_models_dir, f'model_{label_name}.pth')
    encoder_path = os.path.join(saved_models_dir, f'label_encoder_{label_name}.pkl')
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        ready_labels.append(label_name)
    else:
        print(f"❌ Model or encoder not found for {label_name}. Please check your 'saved_models' directory.")
        print(f"Model path: {model_path}")
        print(f"Encoder path: {encoder_path}")

if not ready_labels:
    print("❌ No ready labels found! Please check your 'saved_models' directory.")
    exit()

print(f"✅ Ready labels for prediction: {ready_labels}")

class SingleOutputResNet(nn.Module):
    def __init__(self, num_classes):
        super(SingleOutputResNet, self).__init__()
        base_model = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def preprocess_image(image):
    print("Preprocessing image")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(device)


def predict_single_label(image, label_name):
    print(f"Predicting label: {label_name}")
    model_path = os.path.join(saved_models_dir, f'model_{label_name}.pth')
    encoder_path = os.path.join(saved_models_dir, f'label_encoder_{label_name}.pkl')

    model = SingleOutputResNet(num_classes[label_name])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()

    label_encoder = joblib.load(encoder_path)

    # Move the image to the same device as the model
    image = image.to(device)

    with torch.no_grad():
        output = model(image)[0]
        probs = F.softmax(output, dim=0)

    pred_index = torch.argmax(probs).item()

    try:
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        print(f"✅ {label_name}: {pred_label} (Confidence: {probs[pred_index]:.4f})")
    except Exception as e:
        print(f"❌ Error decoding label for {label_name}: {e}")
        pred_label = "Unknown"

    return pred_label

def predict_image_from_url(url):
    print("Predicting from URL")
    image = load_image_from_url(url)
    if image is None:
        raise ValueError("Invalid image URL")
    predictions = {}
    for label_name in ready_labels:
        pred_label = predict_single_label(image, label_name)
        predictions[label_name] = pred_label
    return predictions
