from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- IMAGE MODEL ----------------
class MyCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.gap(self.conv(x)))

# ---------------- LOAD IMAGE MODEL ----------------
DATASET_PATH = r"C:\plant_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
classes = datasets.ImageFolder(DATASET_PATH).classes

image_model = MyCNNModel(len(classes))
image_model.load_state_dict(torch.load("plantdoc_cnn.pth", map_location=device))
image_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- LOAD TEXT MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained("plant_disease_text_model")
text_model = AutoModelForSequenceClassification.from_pretrained(
    "plant_disease_text_model"
).to(device).eval()

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image is required"}), 400

    img = Image.open(request.files["image"]).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_pred = image_model(img_tensor).argmax(1).item()

    image_disease = classes[img_pred]

    # Optional text
    text = request.form.get("text", "").strip()
    text_disease = None

    if text:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = text_model(**inputs).logits
        text_disease = label_encoder.inverse_transform([logits.argmax(1).item()])[0]

    final_disease = image_disease
    note = "Prediction based on image"

    if text_disease:
        note = "Image and text agree" if text_disease == image_disease else "Image prioritized over text"

    return jsonify({
        "final_disease": final_disease,
        "note": note
    })

if __name__ == "__main__":
    app.run(debug=True)