from PIL import Image
import requests
import io
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def load_image_from_upload(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error loading image from file: {e}")
        return None
