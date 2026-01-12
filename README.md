# PlantDocBot : AI Plant Disease Diagnosis via Chat and Image Upload

PlantDocBot is a multimodal AI system for plant disease diagnosis that combines **image-based classification**, **text-based symptom validation**, and **dataset-driven guidance** through a retrieval-based chatbot.
The system is designed to assist farmers, gardeners, and students by providing reliable disease identification from plant leaf images, with optional symptom validation.

---

## Overview

PlantDocBot processes:
- **Plant leaf images** for primary disease detection
- **Optional symptom descriptions** for validation
- **Curated dataset captions** for explanations and guidance

The image model is the authoritative predictor, while the text model acts as a validator.

---

## Objectives

- Detect plant diseases from leaf images
- Validate predictions using symptom descriptions (optional)
- Retrieve disease-specific information from trusted datasets
- Provide interactive, dataset-backed guidance via chatbot

---

## Key Features

- Image-based plant disease classification (38 classes)
- Optional text-based symptom validation
- Dataset-backed disease explanations
- Retrieval-based chatbot for plant care guidance
- Web-based interface for image upload and interaction

---

## System Architecture


- Image model: primary diagnosis
- Text model: validation only
- Chatbot: retrieval-based (non-generative)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                                  │
│                     (React Frontend - index.html)                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌───────────────┐         ┌──────────────┐
        │ Image Upload  │         │  Text Input  │
        │   (Leaf Photo)│         │  (Symptoms)  │
        └───────┬───────┘         └──────┬───────┘
                │                        │
                ▼                        ▼
        ┌───────────────┐         ┌──────────────────┐
        │   CNN Model   │         │   NLP Model      │
        │               │         │                  │
        │               │         │                  │
                        │         │                  │
        └───────┬───────┘         └──────┬───────────┘
                │                        │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────┐
                │ Disease Prediction │
                │  (Validation &     │
                │  Confidence Score) │
                └─────────┬──────────┘
                          │
                          ├─────────────────┐
                          ▼                 ▼
                ┌─────────────────┐   ┌────────────────┐
                │ Dataset Lookup  │   │   Chatbot      │
                │   (captions.    │   │   Interface    │
                │    parquet)     │   │  (Interactive  │
                │                 │   │   Guidance)    │
                └────────┬────────┘   └────────┬───────┘
                         │                     │
                         ▼                     ▼
                ┌─────────────────┐   ┌────────────────┐
                │  Explanation    │   │   Treatment    │
                │  • Disease Info │   │   Guidance     │
                │  • Symptoms     │   │  • Cures       │
                │  • Causes       │   │  • Prevention  │
                └─────────────────┘   └────────────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Response to User   │
                         │  (Frontend Display) │
                         └─────────────────────┘
```
---

## 🗂️ Datasets Used

- **PlantVillage Dataset**  
  Used for training the image-based disease classification model (38 classes).

- **PlantVillage Image–Text Pairs (Hugging Face)**  
  Used to retrieve disease descriptions and guidance.

---

## 🤖 Models

### Image Model
- Custom CNN trained on 38 PlantVillage disease classes
- Input: Plant leaf image
- Output: Disease label

### Text Model
- Transformer-based classifier (DistilBERT)
- Input: Symptom description (optional)
- Role: Validation of image-based prediction

### Chatbot
- Retrieval-based
- Uses dataset captions for:
  - Symptoms
  - Treatment suggestions
  - Prevention tips
- No generative language model used

---

## 📁 Project Structure

```
PlantDocBot/
├── backend/
│   ├── app.py                              
│   ├── requirements.txt                     
│   ├── plantdoc_cnn.pth                    
│   ├── label_encoder.pkl                   
│   ├── plant_disease_captions.parquet      
│   └── plant_disease_text_model/           
│       ├── model.safetensors              
│       ├── config.json                      
│       ├── tokenizer.json                   
│       ├── tokenizer_config.json            
│       ├── special_tokens_map.json          
│       └── vocab.txt                        
│
├── frontend/
│   ├── index.html                          
│   └── src/
│       ├── main.jsx                         
│       └── App.jsx                          
│
├── PlantDoc.ipynb                          
├── .gitignore
└── LICENSE
```
---
### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/springboardmentor123g/PlantDocBotProject.git
cd PlantDocBotProject
git checkout Intern-SriramRamanadham
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
```

Backend will be available at `http://localhost:5000`

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

---

##  Model Weights

Due to size limitations, trained model weights are **not stored in this repository**.

They are **automatically downloaded on first run** using a scripted setup.

---

##  Deployment

- **Backend**: Flask + PyTorch (Render)
- **Frontend**: React (Vite) static site
- API endpoints:
  - `/predict` — disease detection
  - `/chat` — dataset-driven guidance

---


## 📧 Contact

For any inquiries or feedback, please reach out to [Sriram Ramanadham](sriramramanadham355@gmail.com).

---
