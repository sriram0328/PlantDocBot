import json

# Load extracted training classes
with open("training_classes.json", "r") as f:
    training_classes = json.load(f)

# 🔴 COPY your CLASSES list from app.py EXACTLY
INFERENCE_CLASSES = [
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

print("=" * 60)
print("CLASS ORDER COMPARISON")
print("=" * 60)

print(f"Training classes:  {len(training_classes)}")
print(f"Inference classes: {len(INFERENCE_CLASSES)}\n")

mismatch = False

for i, (t, inf) in enumerate(zip(training_classes, INFERENCE_CLASSES)):
    if t != inf:
        mismatch = True
        print(f"❌ Index {i}:")
        print(f"   TRAIN : {t}")
        print(f"   INFER : {inf}\n")

print("Only in training:", set(training_classes) - set(INFERENCE_CLASSES))
print("Only in inference:", set(INFERENCE_CLASSES) - set(training_classes))

if not mismatch:
    print("\n✅ CLASS ORDER MATCHES PERFECTLY")
else:
    print("\n🔥 CLASS ORDER MISMATCH — THIS ALONE CAN CAUSE ~1% ACCURACY")
