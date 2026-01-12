"""
Generate fresh disease information database.

This script creates a CLEAN, VERIFIED care-recommendation dataset
that MUST align 1-to-1 with the CNN image classes.
"""

import pandas as pd

# -------------------- SOURCE OF TRUTH --------------------
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

# -------------------- DISEASE INFO --------------------
DISEASE_INFO = {
    'Apple___Apple_scab': [
        "Remove and destroy fallen leaves to reduce fungal spores.",
        "Apply fungicides during wet spring weather.",
        "Choose scab-resistant apple varieties for future plantings."
    ],
    'Apple___Black_rot': [
        "Prune out dead and diseased branches.",
        "Remove mummified fruits and cankers.",
        "Apply appropriate fungicides during the growing season."
    ],
    'Apple___Cedar_apple_rust': [
        "Remove nearby cedar trees if possible.",
        "Apply fungicides starting at pink bud stage.",
        "Plant rust-resistant apple varieties."
    ],
    'Apple___healthy': [
        "Continue regular monitoring for disease symptoms.",
        "Maintain good sanitation practices.",
        "Ensure proper nutrition and watering."
    ],
    'Blueberry___healthy': [
        "Maintain soil pH between 4.5–5.5.",
        "Provide consistent moisture.",
        "Monitor for pests and diseases."
    ],
    'Cherry_(including_sour)___Powdery_mildew': [
        "Improve air circulation by pruning.",
        "Apply sulfur-based fungicides.",
        "Avoid overhead watering."
    ],
    'Cherry_(including_sour)___healthy': [
        "Prune for airflow.",
        "Monitor regularly.",
        "Maintain proper nutrition."
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        "Use resistant hybrids.",
        "Rotate crops.",
        "Apply fungicides if severe."
    ],
    'Corn_(maize)___Common_rust_': [
        "Plant resistant hybrids.",
        "Apply fungicides if early.",
        "Remove volunteer corn."
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        "Use resistant hybrids.",
        "Rotate crops.",
        "Bury infected residue."
    ],
    'Corn_(maize)___healthy': [
        "Monitor crop health.",
        "Maintain soil fertility.",
        "Ensure proper spacing."
    ],
    'Grape___Black_rot': [
        "Remove mummified berries.",
        "Apply fungicides from bloom to harvest.",
        "Prune for airflow."
    ],
    'Grape___Esca_(Black_Measles)': [
        "Remove severely infected vines.",
        "Avoid pruning during wet conditions.",
        "No effective chemical control available."
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        "Improve canopy airflow.",
        "Apply copper fungicides.",
        "Remove infected debris."
    ],
    'Grape___healthy': [
        "Maintain canopy management.",
        "Monitor vines regularly.",
        "Ensure proper nutrition."
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        "Remove infected trees immediately.",
        "Control psyllid vectors.",
        "Use certified nursery stock."
    ],
    'Peach___Bacterial_spot': [
        "Plant resistant varieties.",
        "Apply copper sprays during dormancy.",
        "Avoid overhead irrigation."
    ],
    'Peach___healthy': [
        "Prune annually.",
        "Monitor pests and diseases.",
        "Thin fruit when necessary."
    ],
    'Pepper,_bell___Bacterial_spot': [
        "Use clean seeds and transplants.",
        "Avoid overhead irrigation.",
        "Apply copper sprays preventively."
    ],
    'Pepper,_bell___healthy': [
        "Maintain soil moisture.",
        "Provide proper spacing.",
        "Monitor plant health."
    ],
    'Potato___Early_blight': [
        "Remove infected leaves.",
        "Apply fungicides early.",
        "Rotate crops."
    ],
    'Potato___Late_blight': [
        "Apply preventive fungicides.",
        "Destroy infected plants immediately.",
        "Use certified seed potatoes."
    ],
    'Potato___healthy': [
        "Hill soil around plants.",
        "Monitor regularly.",
        "Maintain proper irrigation."
    ],
    'Raspberry___healthy': [
        "Prune old canes.",
        "Improve air circulation.",
        "Monitor for disease."
    ],
    'Soybean___healthy': [
        "Scout fields regularly.",
        "Maintain soil fertility.",
        "Use crop rotation."
    ],
    'Squash___Powdery_mildew': [
        "Apply sulfur-based fungicides.",
        "Improve airflow.",
        "Plant resistant varieties."
    ],
    'Strawberry___Leaf_scorch': [
        "Remove infected leaves.",
        "Apply fungicides during runner stage.",
        "Renovate beds after harvest."
    ],
    'Strawberry___healthy': [
        "Remove old leaves.",
        "Control weeds.",
        "Ensure good spacing."
    ],
    'Tomato___Bacterial_spot': [
        "Use clean transplants.",
        "Apply copper sprays.",
        "Avoid working with wet plants."
    ],
    'Tomato___Early_blight': [
        "Remove infected leaves.",
        "Apply fungicides early.",
        "Use mulch."
    ],
    'Tomato___Late_blight': [
        "Apply preventive fungicides.",
        "Remove infected plants immediately.",
        "Avoid overhead watering."
    ],
    'Tomato___Leaf_Mold': [
        "Increase ventilation.",
        "Space plants adequately.",
        "Apply fungicides if needed."
    ],
    'Tomato___Septoria_leaf_spot': [
        "Remove infected leaves.",
        "Apply fungicides.",
        "Avoid overhead irrigation."
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Wash mites off plants.",
        "Use insecticidal soap or neem oil.",
        "Introduce predatory mites."
    ],
    'Tomato___Target_Spot': [
        "Remove infected debris.",
        "Apply labeled fungicides.",
        "Reduce leaf wetness."
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Control whiteflies.",
        "Remove infected plants.",
        "Use resistant varieties."
    ],
    'Tomato___Tomato_mosaic_virus': [
        "Use resistant varieties.",
        "Sanitize tools.",
        "Remove infected plants."
    ],
    'Tomato___healthy': [
        "Monitor plant health.",
        "Maintain consistent watering.",
        "Prune for airflow."
    ]
}

# -------------------- VALIDATION --------------------
missing = set(CLASSES) - set(DISEASE_INFO.keys())
if missing:
    raise RuntimeError(f"Missing disease info for: {missing}")

# -------------------- BUILD DATAFRAME --------------------
rows = []
for disease, tips in DISEASE_INFO.items():
    for tip in tips:
        rows.append({
            "disease_label": disease,
            "captions": tip
        })

df = pd.DataFrame(rows)
df.to_parquet("plant_disease_captions.parquet", index=False)

print("✓ plant_disease_captions.parquet generated successfully")
print(f"✓ Diseases covered: {df['disease_label'].nunique()}")
print(f"✓ Total entries: {len(df)}")
