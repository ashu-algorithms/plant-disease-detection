"""
predict.py - Inference and Disease Diagnosis Module

Loads the trained CNN model (models/plant_model.keras) and performs
image preprocessing, prediction, confidence scoring, and diagnostic lookup.
Can be imported in app.py or run directly from the command line.
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Ensure Keras runs with the PyTorch backend on Windows for reliable execution
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")

IMG_HEIGHT = 128
IMG_WIDTH = 128

# Cached model and classes
_MODEL = None
_CLASSES = None

# Comprehensive Agronomic Knowledge Base
DISEASE_INFO = {
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn - Common Rust",
        "crop": "Corn / Maize (Zea mays)",
        "status": "Diseased",
        "pathogen": "Fungus (Puccinia sorghi)",
        "severity": "Moderate to High",
        "symptoms": (
            "Golden-brown to cinnamon-brown elongated pustules appear on both upper and lower leaf surfaces. "
            "Pustules rupture epidermal tissue, releasing powdery rust-colored spores that spread via wind and rain."
        ),
        "treatment": (
            "1. Apply foliar fungicides (such as azoxystrobin, pyraclostrobin, or propiconazole) if infection reaches threshold prior to tasseling.\n"
            "2. Utilize resistant or tolerant corn hybrid seed varieties for subsequent seasons.\n"
            "3. Rotate fields with non-host crops like soybeans or legumes to reduce fungal survival."
        ),
        "prevention": (
            "Plant early in the season to avoid peak spore dispersal. Ensure optimal plant density to encourage foliage drying."
        )
    },
    "Corn_(maize)___healthy": {
        "display_name": "Corn - Healthy",
        "crop": "Corn / Maize (Zea mays)",
        "status": "Healthy",
        "pathogen": "None (No Disease Detected)",
        "severity": "None",
        "symptoms": (
            "Foliage demonstrates uniform vibrant green pigmentation, smooth leaf blades, and vigorous vegetative growth with no visible chlorosis or lesions."
        ),
        "treatment": (
            "No chemical or therapeutic treatment required. Continue standard balanced nitrogen-phosphorus-potassium (NPK) fertilization."
        ),
        "prevention": (
            "Maintain soil organic matter, monitor soil moisture levels, and perform regular field scouting."
        )
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper - Bacterial Spot",
        "crop": "Bell Pepper (Capsicum annuum)",
        "status": "Diseased",
        "pathogen": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "severity": "High",
        "symptoms": (
            "Small, circular or irregular water-soaked spots on leaves that turn brown to purplish-black with chlorotic yellow halos. "
            "Leaves may turn yellow and drop prematurely, causing sunscald on developing fruit."
        ),
        "treatment": (
            "1. Spray fixed copper bactericides combined with mancozeb to slow bacterial spread.\n"
            "2. Promptly remove and safely destroy heavily infected leaves and plant debris.\n"
            "3. Avoid overhead sprinkler irrigation; switch to drip or furrow watering to keep foliage dry."
        ),
        "prevention": (
            "Use certified disease-free seeds and transplants. Practice a 2 to 3-year crop rotation avoiding solanaceous crops."
        )
    },
    "Pepper,_bell___healthy": {
        "display_name": "Bell Pepper - Healthy",
        "crop": "Bell Pepper (Capsicum annuum)",
        "status": "Healthy",
        "pathogen": "None (No Disease Detected)",
        "severity": "None",
        "symptoms": (
            "Glossy, intact dark green leaves with crisp margins and active vegetative and floral development without leaf spotting or curling."
        ),
        "treatment": (
            "No disease control measures necessary. Maintain standard micronutrient feeding (calcium and magnesium)."
        ),
        "prevention": (
            "Provide consistent irrigation to prevent blossom-end rot. Mulch soil surface to conserve moisture."
        )
    },
    "Potato___Early_blight": {
        "display_name": "Potato - Early Blight",
        "crop": "Potato (Solanum tuberosum)",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate",
        "symptoms": (
            "Target-board concentric ring lesions with dark brown necrotic centers, typically starting on older bottom leaves. "
            "Surrounding tissue turns yellow, eventually drying up and drooping."
        ),
        "treatment": (
            "1. Apply protectant fungicides such as chlorothalonil, mancozeb, or azoxystrobin at early symptom onset.\n"
            "2. Prune heavily diseased lower canopy leaves to reduce inoculum load.\n"
            "3. Ensure balanced nitrogen nutrition; stressed or nutrient-deficient plants are significantly more susceptible."
        ),
        "prevention": (
            "Implement a minimum 3-year crop rotation. Ensure proper hill spacing to promote rapid leaf drying."
        )
    },
    "Potato___Late_blight": {
        "display_name": "Potato - Late Blight",
        "crop": "Potato (Solanum tuberosum)",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Severe / Critical",
        "symptoms": (
            "Large, irregular water-soaked pale to dark lesions that rapidly expand and turn black. "
            "In humid or wet weather, a delicate white fuzzy fungal growth appears on leaf undersides. Can destroy an entire crop in days."
        ),
        "treatment": (
            "1. Apply systemic targeted fungicides (e.g. cymoxanil, mandipropamid, fluopicolide, or metalaxyl).\n"
            "2. Immediately cull and bury/burn infected plants. Do not compost late blight residues.\n"
            "3. Harvest tubers only in dry weather and ensure tubers are cured before storage."
        ),
        "prevention": (
            "Plant only certified disease-free seed tubers. Avoid excessive foliage wetness and destroy volunteer potato plants."
        )
    },
    "Potato___healthy": {
        "display_name": "Potato - Healthy",
        "crop": "Potato (Solanum tuberosum)",
        "status": "Healthy",
        "pathogen": "None (No Disease Detected)",
        "severity": "None",
        "symptoms": (
            "Robust compound leaves displaying deep green coloration, firm turgor pressure, and absence of necrotic spotting or wilting."
        ),
        "treatment": (
            "No pathogen intervention required. Maintain steady soil hilling and moisture."
        ),
        "prevention": (
            "Continue standard monitoring for Colorado potato beetle and potato aphids."
        )
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato - Early Blight",
        "crop": "Tomato (Solanum lycopersicum)",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate to High",
        "symptoms": (
            "Dark brown concentric circular spots ('bulls-eye' pattern) on older lower leaves. "
            "Yellow halo borders develop around lesions, causing leaves to yellow, wither, and drop prematurely."
        ),
        "treatment": (
            "1. Spray copper soap or copper octanoate, or synthetic fungicides like chlorothalonil or difenoconazole.\n"
            "2. Prune lower branches (bottom 10-12 inches) to prevent fungal spores from splashing up from soil.\n"
            "3. Stake or cage tomato plants to elevate foliage off ground level."
        ),
        "prevention": (
            "Apply straw or plastic mulch around plant bases. Water only at the soil line using drip hoses."
        )
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato - Late Blight",
        "crop": "Tomato (Solanum lycopersicum)",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Severe / Critical",
        "symptoms": (
            "Rapidly spreading greasy, olive-green to dark brown water-soaked lesions on leaves and stems. "
            "White fungal bloom appears on lesion undersides in humid conditions. Leaves rapidly collapse and turn brown."
        ),
        "treatment": (
            "1. Apply systemic translaminar fungicides (e.g. cymoxanil, mefenoxam, or copper hydroxide).\n"
            "2. Destroy infected plants immediately; bag them before disposal to prevent airborne spore escape.\n"
            "3. Keep greenhouse or field humidity below 80% with ventilation."
        ),
        "prevention": (
            "Select late-blight resistant cultivars (e.g. Mountain Magic, Defiant). Maintain wide row spacing for good air circulation."
        )
    },
    "Tomato___healthy": {
        "display_name": "Tomato - Healthy",
        "crop": "Tomato (Solanum lycopersicum)",
        "status": "Healthy",
        "pathogen": "None (No Disease Detected)",
        "severity": "None",
        "symptoms": (
            "Serrated, vibrant green leaves with strong vascular leaf veins, sturdy petioles, and robust flowering/fruit clusters."
        ),
        "treatment": (
            "No chemical intervention needed. Ensure regular potassium and calcium uptake."
        ),
        "prevention": (
            "Mulch regularly, prune suckers for optimal sunlight penetration, and maintain consistent soil hydration."
        )
    }
}

def load_prediction_model(model_path=None):
    """Loads and caches the trained Keras CNN model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Please train the model first by running: py train.py"
        )
    
    print(f"[INFO] Loading trained CNN model from {path}...")
    _MODEL = keras.models.load_model(path)
    return _MODEL

def load_class_names(classes_path=None):
    """Loads and caches the class names mapping."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    path = classes_path or CLASSES_PATH
    if not os.path.exists(path):
        # Fallback default classes
        return list(DISEASE_INFO.keys())
    
    with open(path, "r", encoding="utf-8") as f:
        _CLASSES = json.load(f)
    return _CLASSES

def preprocess_image(image_input):
    """
    Preprocesses an input image (PIL Image, file path, or file stream)
    for model inference: converts to RGB, resizes to (128, 128), and expands dims.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        # File-like object (e.g. Streamlit UploadedFile)
        img = Image.open(image_input)

    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    # Shape: (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch

def predict_disease(image_input, top_k=3):
    """
    Predicts plant disease from an input leaf image.
    
    Returns:
        dict containing:
            - predicted_class: raw string key
            - display_name: formatted readable name
            - confidence: float percentage (0 - 100)
            - is_healthy: boolean
            - top_predictions: list of tuples [(class_name, display_name, score%)]
            - details: dict of symptoms, treatments, etc.
    """
    model = load_prediction_model()
    class_names = load_class_names()
    
    pil_img, img_batch = preprocess_image(image_input)
    
    # Model forward pass
    raw_preds = model.predict(img_batch, verbose=0)[0]
    # Ensure probabilities sum to 1
    probabilities = np.array(raw_preds, dtype=np.float64)
    if probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
        
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    best_idx = top_indices[0]
    predicted_class = class_names[best_idx]
    confidence_score = float(probabilities[best_idx]) * 100.0
    
    # Gather top-k predictions
    top_list = []
    for idx in top_indices:
        c_name = class_names[idx]
        info = DISEASE_INFO.get(c_name, {})
        disp_name = info.get("display_name", c_name.replace("___", " - ").replace("_", " "))
        score = float(probabilities[idx]) * 100.0
        top_list.append({
            "class_name": c_name,
            "display_name": disp_name,
            "confidence": round(score, 2)
        })
        
    info = DISEASE_INFO.get(predicted_class, {
        "display_name": predicted_class.replace("___", " - ").replace("_", " "),
        "crop": predicted_class.split("___")[0].replace("_", " "),
        "status": "Healthy" if "healthy" in predicted_class.lower() else "Diseased",
        "pathogen": "None" if "healthy" in predicted_class.lower() else "Plant Pathogen",
        "severity": "None" if "healthy" in predicted_class.lower() else "Moderate",
        "symptoms": "Visual leaf abnormality or spotting detected.",
        "treatment": "Inspect leaf specimen and apply appropriate botanical fungicide or bactericide.",
        "prevention": "Maintain clean field conditions and sanitize garden tools."
    })
    
    is_healthy = "healthy" in predicted_class.lower()

    return {
        "predicted_class": predicted_class,
        "display_name": info["display_name"],
        "confidence": round(confidence_score, 2),
        "is_healthy": is_healthy,
        "crop": info.get("crop", "Unknown"),
        "pathogen": info.get("pathogen", "N/A"),
        "severity": info.get("severity", "N/A"),
        "symptoms": info.get("symptoms", "N/A"),
        "treatment": info.get("treatment", "N/A"),
        "prevention": info.get("prevention", "N/A"),
        "top_predictions": top_list
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <path_to_leaf_image>")
        # If test sample exists, predict on it
        samples_dir = os.path.join(BASE_DIR, "samples")
        if os.path.exists(samples_dir) and os.listdir(samples_dir):
            sample_img = os.path.join(samples_dir, os.listdir(samples_dir)[0])
            print(f"\n[DEMO] Testing with sample image: {sample_img}")
            res = predict_disease(sample_img)
            print("=" * 60)
            print(f"Prediction: {res['display_name']}")
            print(f"Confidence: {res['confidence']}%")
            print(f"Status:     {'HEALTHY' if res['is_healthy'] else 'DISEASED'}")
            print(f"Pathogen:   {res['pathogen']}")
            print(f"Treatment:\n{res['treatment']}")
            print("=" * 60)
        sys.exit(0)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
        
    result = predict_disease(image_path)
    print("=" * 60)
    print(f"  PLANT DISEASE DIAGNOSIS RESULT")
    print("=" * 60)
    print(f"  Diagnosis:    {result['display_name']}")
    print(f"  Confidence:   {result['confidence']}%")
    print(f"  Condition:    {'HEALTHY' if result['is_healthy'] else 'DISEASED'}")
    print(f"  Crop:         {result['crop']}")
    print(f"  Pathogen:     {result['pathogen']}")
    print(f"  Severity:     {result['severity']}")
    print("-" * 60)
    print("  SYMPTOMS:")
    print(f"  {result['symptoms']}")
    print("-" * 60)
    print("  RECOMMENDED TREATMENT:")
    print(f"  {result['treatment']}")
    print("-" * 60)
    print("  PREVENTION:")
    print(f"  {result['prevention']}")
    print("=" * 60)
