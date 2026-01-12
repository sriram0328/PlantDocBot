from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------- APP SETUP --------------------
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- IMAGE PREPROCESS --------------------
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def normalize(label: str) -> str:
    return str(label).replace("___", " ").replace("_", " ").lower().strip()

# -------------------- CHAT INTENT --------------------
def detect_intent(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["treat", "treatment", "cure", "remedy"]):
        return "treatment"
    if any(w in msg for w in ["prevent", "prevention", "avoid"]):
        return "prevention"
    if any(w in msg for w in ["cause", "why"]):
        return "cause"
    if any(w in msg for w in ["spread", "transmit"]):
        return "spread"
    return "general"

# -------------------- CNN MODEL --------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        return self.fc_layers(x)


# -------------------- CLASSES --------------------
CLASSES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# -------------------- LOAD IMAGE MODEL --------------------
image_model = CNN(len(CLASSES))
image_model.load_state_dict(torch.load("plantdoc_cnn.pth", map_location=device))
image_model.to(device).eval()

# -------------------- LOAD TEXT MODEL --------------------
tokenizer = AutoTokenizer.from_pretrained("plant_disease_text_model")
text_model = AutoModelForSequenceClassification.from_pretrained(
    "plant_disease_text_model"
).to(device).eval()

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

TEXT_SUPPORTED = set(normalize(c) for c in label_encoder.classes_)

# -------------------- DISEASE KNOWLEDGE BASE --------------------
DISEASE_INFO = {
    normalize("Apple___Black_rot"): {
        "description": (
            "Apple black rot is a fungal disease caused by Botryosphaeria obtusa. "
            "It produces dark circular leaf spots, sunken black lesions on fruits, "
            "and cankers on branches. Mummified fruits and infected wood act as "
            "primary sources of infection."
        ),
        "care": [
            "Prune out dead and diseased branches.",
            "Remove mummified fruits and cankers.",
            "Apply appropriate fungicides during the growing season."
        ]
    },
}

# Auto-fill remaining diseases with safe scientific summaries
for c in CLASSES:
    key = normalize(c)
    if key not in DISEASE_INFO:
        DISEASE_INFO[key] = {
            "description": (
                f"{c.replace('___',' - ').replace('_',' ')} is a plant health condition "
                "that affects normal growth and productivity. It is identified based "
                "on visible symptoms on leaves, stems, or fruits and requires proper "
                "management to prevent spread."
            ),
            "care": [
                "Remove affected plant parts when possible.",
                "Maintain good air circulation and sanitation.",
                "Monitor plants regularly for symptom progression."
            ]
        }

# -------------------- PREDICT --------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img = Image.open(request.files["image"]).convert("RGB")
    x = img_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(image_model(x), dim=1)
        conf, idx = probs.max(1)

    confidence = conf.item()
    raw_label = CLASSES[idx.item()]
    norm_label = normalize(raw_label)

    if confidence < 0.6:
        return jsonify({
            "final_disease": "Uncertain Diagnosis",
            "confidence": round(confidence, 2),
            "description": "Image confidence too low for a reliable diagnosis.",
            "text_used": False
        })

    note = "Visual diagnosis completed."
    text_used = False

    text = request.form.get("text", "").strip()
    if text and norm_label in TEXT_SUPPORTED:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            t_idx = text_model(**inputs).logits.argmax(1).item()
        text_label = label_encoder.inverse_transform([t_idx])[0]
        if normalize(text_label) == norm_label:
            note = "Diagnosis confirmed by symptom description."
        text_used = True

    return jsonify({
        "final_disease": raw_label.replace("___", " - ").replace("_", " "),
        "raw_label": raw_label,
        "confidence": round(confidence, 2),
        "description": DISEASE_INFO[norm_label]["description"],
        "care": DISEASE_INFO[norm_label]["care"],
        "note": note,
        "text_used": text_used
    })

# -------------------- CHAT --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid request"}), 400

    disease = data.get("disease", "")
    message = data.get("message", "")

    if not disease or not message:
        return jsonify({"error": "Missing disease or message"}), 400

    norm_disease = normalize(disease)

    if norm_disease not in DISEASE_INFO:
        return jsonify({
            "reply": "I don’t have detailed information for this disease. Please consult an agricultural expert."
        })

    intent = detect_intent(message)
    care = DISEASE_INFO[norm_disease]["care"]

    if intent == "treatment":
        reply = f"Recommended treatment:\n- {care[0]}"

    elif intent == "prevention":
        tips = care[1:3] if len(care) > 2 else care
        reply = "Prevention measures:\n" + "\n".join(f"- {t}" for t in tips)

    elif intent == "cause":
        reply = (
            "This disease is caused by a pathogen such as a fungus, bacterium, or virus. "
            "Environmental conditions like high humidity, leaf wetness, and poor air "
            "circulation often promote its development."
        )

    elif intent == "spread":
        reply = (
            "The disease can spread through water splash, wind, insects, contaminated tools, "
            "or infected plant material. Removing infected parts and maintaining sanitation "
            "helps limit spread."
        )

    else:
        reply = (
            "You can ask about treatment, prevention, causes, or how this disease spreads."
        )

    return jsonify({"reply": reply})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
