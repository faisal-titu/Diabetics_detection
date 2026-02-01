import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load trained pipeline model
MODEL_PATH = "diabetes_model.pkl"

with open(MODEL_PATH, "rb") as f:
    final_model = pickle.load(f)

print("Model loaded successfully")

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, dpf, age):

    # Create DataFrame matching training columns
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    pred = final_model.predict(input_data)[0]
    probs = final_model.predict_proba(input_data)[0]

    # Colored prediction text
    if pred == 1:
        label = "<h3 style='color:red;'>Diabetes Detected</h3>"
    else:
        label = "<h3 style='color:blue;'>No Diabetes Detected</h3>"

    confidence = f"{max(probs)*100:.2f}%"

    prob_df = pd.DataFrame({
        "Class": ["No Diabetes", "Diabetes"],
        "Probability": [probs[0], probs[1]]
    })

    return label, confidence, prob_df


interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Slider(0,17,step=1,label="Pregnancies",value=3),
        gr.Slider(0,200,step=1,label="Glucose (mg/dL)",value=120),
        gr.Slider(0,122,step=1,label="Blood Pressure (mm Hg)",value=70),
        gr.Slider(0,99,step=1,label="Skin Thickness (mm)",value=20),
        gr.Slider(0,846,step=1,label="Insulin (mu U/ml)",value=79),
        gr.Slider(0,67,step=0.1,label="BMI",value=27),
        gr.Slider(0,2.5,step=0.001,label="Diabetes Pedigree Function",value=0.5),
        gr.Slider(21,81,step=1,label="Age",value=35),
    ],

    outputs=[
        gr.HTML(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Dataframe(label="Class Probabilities")
    ],

    title="Diabetes Prediction System",
    description="""
    Enter medical parameters to predict diabetes risk.

    """,

    examples=[
        [1,85,66,29,0,26.6,0.351,31],
        [6,148,72,35,0,33.6,0.627,50],
        [3,120,70,20,79,27,0.5,35]
    ],

    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch()
