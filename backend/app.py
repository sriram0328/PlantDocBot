from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- IMAGE MODEL ----------------
class CNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n)
        )

    def forward(self, x):
        return self.net(x)

# EXACT class order used during training (38 classes)
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

image_model = CNN(len(CLASSES))
image_model.load_state_dict(torch.load("plantdoc_cnn.pth", map_location=device))
image_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- TEXT MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained("plant_disease_text_model")
text_model = AutoModelForSequenceClassification.from_pretrained(
    "plant_disease_text_model"
).to(device).eval()

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- DATASET (LOAD ONCE) ----------------
df = pd.read_parquet("plant_disease_captions.parquet")

disease_info = (
    df.groupby("caption")["captions"]
      .apply(lambda x: sum(x, []))
      .to_dict()
)

# ---------------- ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image required"}), 400

    # ---- IMAGE PREDICTION ----
    img = Image.open(request.files["image"]).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        idx = image_model(x).argmax(1).item()

    image_disease = CLASSES[idx]

    # ---- TEXT VALIDATION ----
    note = "Prediction based on image"
    text = request.form.get("text", "").strip()

    if text:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = text_model(**inputs).logits

        text_disease = label_encoder.inverse_transform(
            [logits.argmax(1).item()]
        )[0]

        if text_disease == image_disease:
            note = "Prediction validated using symptom description"
        else:
            note = "Image prioritized; symptom description partially mismatched"

    # ---- INFORMATION FROM DATASET ----
    info = disease_info.get(
        image_disease,
        ["No description available for this disease."]
    )[:3]

    return jsonify({
        "final_disease": image_disease,
        "info": info,
        "note": note
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))