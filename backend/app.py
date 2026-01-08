from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODEL (MATCHES TRAINING) ----------
class MyCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        return self.fc_layers(x)

# ---------- LOAD CLASSES ----------
DATASET_PATH = r"C:\plant_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
classes = datasets.ImageFolder(DATASET_PATH).classes

model = MyCNNModel(len(classes))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plantdoc_cnn.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- ROUTES ----------
@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    img = Image.open(image).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1).item()

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "disease": classes[pred],
        "info": "This disease affects plant health. Early treatment helps reduce damage.",
        "image": img_b64
    })

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"]
    return jsonify({
        "reply": f"🤖 Dummy response for: {msg}"
    })

if __name__ == "__main__":
    app.run(debug=True)
