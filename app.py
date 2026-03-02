import gradio as gr
import pandas as pd
import numpy as np
import pickle
from model_utils import RenameDPF  # noqa: F401 — required so pickle can resolve the class

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "diabetes_model.pkl"
with open(MODEL_PATH, "rb") as f:
    final_model = pickle.load(f)
print("Model loaded successfully")

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* ── Dark app background ── */
.gradio-container {
    background: linear-gradient(135deg, #0d0e1c 0%, #12132a 100%) !important;
    min-height: 100vh;
}

/* Hide default title bar */
.main > .prose h1 { display: none; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #3730a3 0%, #6d28d9 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 2.25rem 2.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 40px rgba(109,40,217,0.45);
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(165,120,255,0.2);
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.07) 0%, transparent 60%),
                radial-gradient(circle at 80% 50%, rgba(255,255,255,0.05) 0%, transparent 60%);
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #a5b4fc !important;
    margin-bottom: 0.75rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #2d2f55 !important;
}

/* ── Card panels ── */
.input-panel, .output-panel {
    background: #1a1c30 !important;
    border-radius: 18px !important;
    border: 1px solid #2d2f55 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
    padding: 1.5rem !important;
}

/* ── Slider track color ── */
input[type=range]::-webkit-slider-thumb { background: #818cf8 !important; }
input[type=range] { accent-color: #818cf8 !important; }

/* ── Number inputs beside sliders ── */
.gr-number input { border-radius: 8px !important; }

/* ── Predict button ── */
#predict-btn {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    color: white !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}
#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(124,58,237,0.65) !important;
}
#predict-btn:active { transform: translateY(0) !important; }

/* ── Clear button ── */
#clear-btn {
    background: #1e2038 !important;
    border: 2px solid #2d2f55 !important;
    border-radius: 12px !important;
    padding: 0.85rem 0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}
#clear-btn:hover {
    border-color: #4f46e5 !important;
    background: #252745 !important;
    color: #c7d2fe !important;
}

/* ── Slider label styling ── */
.gr-form label > span { font-weight: 600 !important; color: #cbd5e1 !important; font-size: 0.85rem !important; }

/* ── Info box ── */
.info-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #7dd3fc;
    line-height: 1.6;
}

/* ── Examples section ── */
.gr-samples-table tr:hover { background: #1e2040 !important; cursor: pointer; }
.gr-samples-table th { font-weight: 600 !important; color: #a5b4fc !important; font-size: 0.8rem !important; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    padding: 1.25rem 0 0.5rem;
    line-height: 1.7;
}
"""

# ── Prediction logic ─────────────────────────────────────────────────────────
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, dpf, age):

    input_data = pd.DataFrame({
        "Pregnancies":             [pregnancies],
        "Glucose":                 [glucose],
        "BloodPressure":           [blood_pressure],
        "SkinThickness":           [skin_thickness],
        "Insulin":                 [insulin],
        "BMI":                     [bmi],
        "DiabetesPedigreeFunction":[dpf],
        "Age":                     [age],
    })

    pred  = final_model.predict(input_data)[0]
    probs = final_model.predict_proba(input_data)[0]

    no_prob  = float(probs[0]) * 100
    yes_prob = float(probs[1]) * 100
    confidence = max(no_prob, yes_prob)

    # Risk tier
    if yes_prob >= 65:
        risk_label, risk_badge_color, risk_text_color = "High Risk",    "#7f1d1d", "#fca5a5"
    elif yes_prob >= 35:
        risk_label, risk_badge_color, risk_text_color = "Moderate Risk", "#78350f", "#fde68a"
    else:
        risk_label, risk_badge_color, risk_text_color = "Low Risk",      "#14532d", "#86efac"

    # ── Result card (dark) ──
    if pred == 1:
        icon       = "⚠️"
        title      = "Diabetes Detected"
        title_clr  = "#f87171"
        card_bg    = "linear-gradient(135deg,#2d0a0a,#3f0e0e)"
        card_bdr   = "#7f1d1d"
        bar_bg     = "linear-gradient(90deg,#f87171,#ef4444)"
        bar_pct    = yes_prob
        tip        = "⚕️ Please consult a licensed healthcare professional for a proper diagnosis and treatment plan."
    else:
        icon       = "✅"
        title      = "No Diabetes Detected"
        title_clr  = "#4ade80"
        card_bg    = "linear-gradient(135deg,#052e16,#0a3d1f)"
        card_bdr   = "#166534"
        bar_bg     = "linear-gradient(90deg,#4ade80,#22c55e)"
        bar_pct    = no_prob
        tip        = "✅ Maintain a healthy diet, regular exercise, and annual check-ups."

    result_html = f"""
    <div style="background:{card_bg};border:2px solid {card_bdr};border-radius:18px;
                padding:2rem;text-align:center;animation:fadeIn .4s ease;">
      <div style="font-size:3.2rem;line-height:1;margin-bottom:.6rem;">{icon}</div>
      <h2 style="margin:0 0 .6rem;font-size:1.55rem;font-weight:800;color:{title_clr};">{title}</h2>
      <span style="display:inline-block;background:{risk_badge_color};color:{risk_text_color};
                   padding:.25rem .9rem;border-radius:20px;font-weight:700;font-size:.82rem;
                   margin-bottom:1.25rem;">{risk_label}</span>
      <div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.08);border-radius:14px;
                  padding:1.1rem 1.4rem;box-shadow:0 2px 8px rgba(0,0,0,.3);">
        <div style="font-size:.75rem;font-weight:600;letter-spacing:.8px;
                    text-transform:uppercase;color:#94a3b8;margin-bottom:.3rem;">Confidence</div>
        <div style="font-size:2.4rem;font-weight:800;color:{title_clr};line-height:1.1;">
          {confidence:.1f}<span style="font-size:1.2rem">%</span>
        </div>
        <div style="background:rgba(255,255,255,.1);border-radius:20px;height:10px;overflow:hidden;margin:.8rem 0 .3rem;">
          <div style="width:{bar_pct:.1f}%;background:{bar_bg};height:100%;
                      border-radius:20px;transition:width .7s cubic-bezier(.4,0,.2,1);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.72rem;color:#64748b;">
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
      </div>
      <p style="margin:1rem 0 0;font-size:.78rem;color:#64748b;line-height:1.6;">{tip}</p>
    </div>
    """

    # ── Probability breakdown (dark) ──
    prob_html = f"""
    <div style="background:#1a1c30;border-radius:14px;padding:1.25rem 1.5rem;
                border:1px solid #2d2f55;box-shadow:0 2px 12px rgba(0,0,0,.4);">
      <div style="font-size:.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                  color:#a5b4fc;border-bottom:2px solid #2d2f55;padding-bottom:.5rem;margin-bottom:1rem;">
        Probability Breakdown
      </div>

      <div style="margin-bottom:1rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
          <span style="font-weight:600;color:#4ade80;font-size:.88rem;">🟢 &nbsp;No Diabetes</span>
          <span style="font-weight:800;color:#4ade80;font-size:.95rem;">{no_prob:.1f}%</span>
        </div>
        <div style="background:rgba(255,255,255,.08);border-radius:20px;height:14px;overflow:hidden;">
          <div style="width:{no_prob:.1f}%;background:linear-gradient(90deg,#4ade80,#22c55e);
                      height:100%;border-radius:20px;"></div>
        </div>
      </div>

      <div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
          <span style="font-weight:600;color:#f87171;font-size:.88rem;">🔴 &nbsp;Diabetes</span>
          <span style="font-weight:800;color:#f87171;font-size:.95rem;">{yes_prob:.1f}%</span>
        </div>
        <div style="background:rgba(255,255,255,.08);border-radius:20px;height:14px;overflow:hidden;">
          <div style="width:{yes_prob:.1f}%;background:linear-gradient(90deg,#f87171,#ef4444);
                      height:100%;border-radius:20px;"></div>
        </div>
      </div>

      <div style="margin-top:1.1rem;padding:.8rem 1rem;background:#12132a;border-radius:10px;
                  border:1px dashed #2d2f55;font-size:.78rem;color:#64748b;line-height:1.6;">
        <b style='color:#94a3b8'>Model:</b> <span style='color:#818cf8'>Calibrated Soft-Voting Ensemble</span> &nbsp;|&nbsp;
        <b style='color:#94a3b8'>AUC:</b> <span style='color:#818cf8'>0.9306</span> &nbsp;|&nbsp;
        <b style='color:#94a3b8'>Accuracy:</b> <span style='color:#818cf8'>90.1%</span>
      </div>
    </div>
    """

    return result_html, prob_html


def clear_inputs():
    return 3, 120, 70, 20, 79, 27.0, 0.500, 35, "", ""


# ── Build UI ──────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
).set(
    body_background_fill="transparent",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    block_background_fill="#1a1c30",
    block_border_color="#2d2f55",
    block_shadow="0 4px 24px rgba(0,0,0,0.4)",
    block_label_text_color="#cbd5e1",
    block_label_background_fill="#12132a",
    block_label_text_size="sm",
    block_title_text_color="#e2e8f0",
    input_background_fill="#12132a",
    input_border_color="#2d2f55",
    input_border_color_focus="#6366f1",
    input_placeholder_color="#475569",
    slider_color="#818cf8",
    table_even_background_fill="#1a1c30",
    table_odd_background_fill="#14162a",
    table_border_color="#2d2f55",
    button_primary_background_fill="linear-gradient(90deg,#4f46e5,#7c3aed)",
    button_primary_text_color="white",
    button_secondary_background_fill="#1e2038",
    button_secondary_border_color="#2d2f55",
    button_secondary_text_color="#94a3b8",
    border_color_primary="#2d2f55",
    color_accent="#818cf8",
    color_accent_soft="#1e2040",
    background_fill_primary="#0d0e1c",
    background_fill_secondary="#12132a",
)

with gr.Blocks(title="Diabetes Risk Predictor", theme=theme, css=CUSTOM_CSS) as demo:

    # ── Hero ──
    gr.HTML("""
    <div class="hero-header">
      <div style="font-size:2.6rem;margin-bottom:.3rem;">🩺</div>
      <h1 style="font-size:1.9rem;font-weight:800;margin:0 0 .4rem;letter-spacing:-.5px;">
        Diabetes Risk Predictor
      </h1>
      <p style="opacity:.87;font-size:.97rem;margin:0;max-width:520px;margin-inline:auto;line-height:1.6;">
        AI-powered assessment using a Calibrated Soft-Voting Ensemble trained on the
        Pima Indians Diabetes dataset &nbsp;·&nbsp; <b>90.1% accuracy</b>
      </p>
    </div>
    """)

    # ── Main layout ──
    with gr.Row(equal_height=False):

        # ── Left column – inputs ──
        with gr.Column(scale=5, min_width=340):
            gr.HTML('<div class="section-label">📋 &nbsp;Patient Parameters</div>')

            with gr.Group(elem_classes="input-panel"):

                gr.HTML('<div style="font-size:.78rem;font-weight:600;color:#a5b4fc;'
                        'text-transform:uppercase;letter-spacing:.8px;margin-bottom:.6rem;">'
                        '👤 Personal Info</div>')
                with gr.Row():
                    age         = gr.Slider(21, 81,  step=1,   value=35,   label="Age (years)")
                    pregnancies = gr.Slider(0,  17,  step=1,   value=3,    label="Pregnancies")

                gr.HTML('<div style="font-size:.78rem;font-weight:600;color:#a5b4fc;'
                        'text-transform:uppercase;letter-spacing:.8px;margin:.9rem 0 .6rem;">'
                        '🩸 Blood Metrics</div>')
                with gr.Row():
                    glucose        = gr.Slider(0, 200, step=1,   value=120,  label="Glucose (mg/dL)")
                    blood_pressure = gr.Slider(0, 122, step=1,   value=70,   label="Blood Pressure (mmHg)")

                with gr.Row():
                    insulin = gr.Slider(0, 846, step=1,   value=79,   label="Insulin (µU/mL)")
                    dpf     = gr.Slider(0,  2.5, step=0.001, value=0.500, label="Diabetes Pedigree Function")

                gr.HTML('<div style="font-size:.78rem;font-weight:600;color:#a5b4fc;'
                        'text-transform:uppercase;letter-spacing:.8px;margin:.9rem 0 .6rem;">'
                        '📏 Body Measurements</div>')
                with gr.Row():
                    bmi             = gr.Slider(0, 67, step=0.1, value=27.0, label="BMI (kg/m²)")
                    skin_thickness  = gr.Slider(0, 99, step=1,   value=20,   label="Skin Thickness (mm)")

            # Info banner
            gr.HTML("""
            <div class="info-box" style="margin-top:.75rem;">
              <b>Normal reference ranges (approximate):</b><br>
              Glucose: 70–99 &nbsp;·&nbsp; Blood Pressure: 60–80 &nbsp;·&nbsp;
              BMI: 18.5–24.9 &nbsp;·&nbsp; Insulin: 2–25 &nbsp;·&nbsp; DPF: 0.08–2.5
            </div>
            """)

            # Buttons
            with gr.Row():
                clear_btn   = gr.Button("↺  Clear", elem_id="clear-btn")
                predict_btn = gr.Button("🔍  Predict", elem_id="predict-btn", variant="primary")

        # ── Right column – outputs ──
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<div class="section-label">📊 &nbsp;Prediction Result</div>')

            result_html = gr.HTML(
                value="""
                <div style="background:#12132a;border:2px dashed #2d2f55;border-radius:18px;
                            padding:3rem 2rem;text-align:center;color:#475569;">
                  <div style="font-size:2.8rem;margin-bottom:.75rem;">🔬</div>
                  <p style="font-weight:600;font-size:1rem;margin:0 0 .3rem;color:#64748b;">
                    Ready to Analyze
                  </p>
                  <p style="font-size:.83rem;margin:0;">
                    Adjust parameters and click <b style='color:#818cf8'>Predict</b>
                  </p>
                </div>""",
                elem_classes="output-panel",
            )

            prob_html = gr.HTML(
                value="""
                <div style="background:#1a1c30;border:1px solid #2d2f55;border-radius:14px;
                            padding:2rem;text-align:center;">
                  <div style="font-size:1.8rem;margin-bottom:.5rem;">📈</div>
                  <p style="font-size:.83rem;margin:0;color:#475569;">
                    Probability breakdown will appear here
                  </p>
                </div>""",
            )

    # ── Examples ──
    gr.HTML('<div class="section-label" style="margin-top:1rem;">💡 &nbsp;Quick Examples</div>')
    gr.Examples(
        examples=[
            [1, 85, 66, 29, 0,    26.6, 0.351, 31],   # likely no diabetes
            [6, 148, 72, 35, 0,   33.6, 0.627, 50],   # likely diabetes
            [3, 120, 70, 20, 79,  27.0, 0.500, 35],   # borderline
            [8, 183, 64, 0,  0,   23.3, 0.672, 32],   # high glucose
            [0, 137, 40, 35, 168, 43.1, 2.288, 33],   # high insulin/DPF
        ],
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age],
        outputs=[result_html, prob_html],
        fn=predict_diabetes,
        cache_examples=False,
        label=None,
    )

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
      <b>Disclaimer:</b> This tool is for educational purposes only and is <em>not</em> a medical device.<br>
      Always consult a qualified healthcare professional for diagnosis and treatment.
      <br><br>
      Built with ❤️ using <b>PyCaret · scikit-learn · Gradio</b>
    </div>
    """)

    # ── Wire up events ──
    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age],
        outputs=[result_html, prob_html],
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                 insulin, bmi, dpf, age, result_html, prob_html],
    )


if __name__ == "__main__":
    demo.launch()

