from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()

# Mount static files (for CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the saved Random Forest model
model = joblib.load("random_forest_model.pkl")

# Use templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Define the symptoms
symptoms = [
    "Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills", 
    "Joint Pain", "Stomach Pain", "Acidity", "Ulcers On Tongue", "Muscle Wasting", "Vomiting", 
    "Burning Micturition", "Spotting urination", "Fatigue", "Weight Gain", "Anxiety", 
    "Cold Hands And Feets", "Mood Swings", "Weight Loss", "Restlessness", "Lethargy", 
    "Patches In Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes", 
    "Breathlessness", "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", 
    "Dark Urine", "Nausea", "Loss Of Appetite", "Pain Behind The Eyes", "Back Pain", "Constipation", 
    "Abdominal Pain", "Diarrhoea", "Mild Fever", "Yellow Urine", "Yellowing Of Eyes", 
    "Acute Liver Failure", "Fluid Overload", "Swelling Of Stomach", "Swelled Lymph Nodes", 
    "Malaise", "Blurred And Distorted Vision", "Phlegm", "Throat Irritation", "Redness Of Eyes", 
    "Sinus Pressure", "Runny Nose", "Congestion", "Chest Pain", "Weakness In Limbs", "Fast Heart Rate", 
    "Pain During Bowel Movements", "Pain In Anal Region", "Bloody Stool", "Irritation In Anus", 
    "Neck Pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen Legs", "Swollen Blood Vessels", 
    "Puffy Face And Eyes", "Enlarged Thyroid", "Brittle Nails", "Swollen Extremities", 
    "Excessive Hunger", "Extra Marital Contacts", "Drying And Tingling Lips", "Slurred Speech", 
    "Knee Pain", "Hip Joint Pain", "Muscle Weakness", "Stiff Neck", "Swelling Joints", 
    "Movement Stiffness", "Spinning Movements", "Loss Of Balance", "Unsteadiness", 
    "Weakness Of One Body Side", "Loss Of Smell", "Bladder Discomfort", "Foul Smell Of urine", 
    "Continuous Feel Of Urine", "Passage Of Gases", "Internal Itching", "Toxic Look (typhos)", 
    "Depression", "Irritability", "Muscle Pain", "Altered Sensorium", "Red Spots Over Body", 
    "Belly Pain", "Abnormal Menstruation", "Dischromic Patches", "Watering From Eyes", 
    "Increased Appetite", "Polyuria", "Family History", "Mucoid Sputum", "Rusty Sputum", 
    "Lack Of Concentration", "Visual Disturbances", "Receiving Blood Transfusion", 
    "Receiving Unsterile Injections", "Coma", "Stomach Bleeding", "Distention Of Abdomen", 
    "History Of Alcohol Consumption", "Fluid Overload.1", "Blood In Sputum", "Prominent Veins On Calf", 
    "Palpitations", "Painful Walking", "Pus Filled Pimples", "Blackheads", "Scurring", 
    "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails", "Inflammatory Nails", 
    "Blister", "Red Sore Around Nose", "Yellow Crust Ooze"
]

# Define the disease names corresponding to the model's output
diseases = [
    "Paroymsal Positional Vertigo", "AIDS", "Acne", "Alcoholic hepatitis", "Allergy", "Arthritis",
    "Bronchial Asthma", "Cervical spondylosis", "Chicken pox", "Chronic cholestasis", 
    "Common Cold", "Dengue", "Diabetes", "Dimorphic hemorrhoids (piles)", "Drug Reaction", 
    "Fungal infection", "GERD", "Gastroenteritis", "Heart attack", "Hepatitis B", 
    "Hepatitis C", "Hepatitis D", "Hepatitis E", "Hypertension", "Hyperthyroidism", 
    "Hypoglycemia", "Hypothyroidism", "Impetigo", "Jaundice", "Malaria", "Migraine", 
    "Osteoarthritis", "Paralysis (brain hemorrhage)", "Peptic ulcer disease", "Pneumonia", 
    "Psoriasis", "Tuberculosis", "Typhoid", "Urinary tract infection", "Varicose veins", 
    "Hepatitis A"
]

# Route to render the form
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "symptoms": symptoms})

# Route to handle form submission and make a prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict_disease(request: Request, selected_symptoms: list = Form(...)):
    input_data = [0] * len(symptoms)
    
    # Map selected symptoms to the model input
    for symptom in selected_symptoms:
        index = symptoms.index(symptom)
        input_data[index] = 1
    
    # Reshape input and predict
    input_data = np.array(input_data).reshape(1, -1)
    prediction_index = model.predict(input_data)[0]
    
    # Map the index to the disease name
    predicted_disease = diseases[prediction_index]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "symptoms": symptoms,
        "prediction": predicted_disease
    })