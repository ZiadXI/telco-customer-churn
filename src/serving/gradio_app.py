import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

def predict_churn(
    gender, senior_citizen, partner, dependents, tenure, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges, total_charges
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code != 200:
            return f'<div style="color: #ef4444; font-weight: bold;">❌ API Error ({response.status_code}): {response.text}</div>'
        
        result = response.json()
        
        proba = result["churn_probability"]
        prediction = result["churn_prediction"]
        probability_percent = proba * 100

        color = "#ef4444" if prediction == "Yes" else "#10b981" # Red vs Green
        status_text = "HIGH RISK" if prediction == "Yes" else "LOW RISK"
        icon = "⚠️" if prediction == "Yes" else "✅"
        
        html_result = f"""
        <div style="background-color: #1f2937; padding: 30px; border-radius: 12px; border: 2px solid {color}; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
            <div style="color: {color}; font-size: 24px; font-weight: bold; margin-bottom: 10px;">{icon} {status_text}</div>
            <div style="font-size: 72px; font-weight: 800; color: white; line-height: 1;">
                {probability_percent:.1f}%
            </div>
            <div style="color: #9ca3af; font-size: 14px; margin-top: 5px;">CHURN PROBABILITY</div>
            <hr style="border-color: #374151; margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; color: white;">
                <span>Prediction:</span>
                <span style="font-weight: bold; color: {color}">{prediction}</span>
            </div>
        </div>
        """
        return html_result

    except requests.exceptions.ConnectionError:
        return '<div style="color: #ef4444; font-weight: bold;">❌ Connection Error: Is the API running?</div>'
    except Exception as e:
        return f'<div style="color: red;">System Error: {str(e)}</div>'

nuclear_css = """
:root, .gradio-container {
    --body-background-fill: #0b0f19 !important;
    --body-text-color: #ffffff !important;
    --background-fill-primary: #1f2937 !important;
    --background-fill-secondary: #0b0f19 !important;
    --block-background-fill: #111827 !important;
    --block-border-color: #374151 !important;
    --block-label-text-color: #ffffff !important;
    --input-background-fill: #1f2937 !important;
    --input-text-color: #ffffff !important;
    --button-primary-background-fill: #2563eb !important;
    --button-primary-text_color: #ffffff !important;
}

label, span.block-label, span.label, .block-title, .form label, .gradio-container .label-wrap span {
    color: #ffffff !important;
    font-weight: bold !important;
    opacity: 1 !important;
    background-color: transparent !important;
}

.gradio-container select, .gradio-container input, .gradio-container textarea {
    color: white !important;
    background-color: #1f2937 !important;
}

ul.options li.item {
    background-color: #1f2937 !important;
    color: white !important;
}

.gradio-container input[type=radio] {
    background-color: #374151 !important; 
    border-color: #4b5563 !important;
}

.gradio-container .panel {
    background: #111827 !important;
    border: 1px solid #374151 !important;
}
"""

with gr.Blocks(theme=gr.themes.Default(), css=nuclear_css, title="Telco Churn AI") as demo:
    
    with gr.Row():
        gr.Markdown(
            """
            #  Telco Customer Churn Prediction (API Connected)
            ### Configure customer profile to assess retention risk.
            """
        )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("###  Demographics")
                with gr.Row():
                    gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                    senior = gr.Radio([0, 1], label="Senior Citizen", value=0)
                with gr.Row():
                    partner = gr.Radio(["Yes", "No"], label="Partner", value="No")
                    dependents = gr.Radio(["Yes", "No"], label="Dependents", value="No")

            with gr.Group():
                gr.Markdown("###  Services")
                with gr.Row():
                    tenure = gr.Slider(0, 72, label="Tenure (Months)", step=1, value=1)
                    phone = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
                
                with gr.Row():
                    internet = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                    lines = gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")

                with gr.Accordion("Advanced Service Options", open=False):
                    with gr.Row():
                        sec = gr.Radio(["Yes", "No", "No internet service"], label="Online Security", value="No")
                        backup = gr.Radio(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                    with gr.Row():
                        dev = gr.Radio(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                        tech = gr.Radio(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                    with gr.Row():
                        tv = gr.Radio(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                        mov = gr.Radio(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

            with gr.Group():
                gr.Markdown("###  Billing Info")
                with gr.Row():
                    contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                    paperless = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
                
                payment = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
                                      label="Payment Method", value="Electronic check")
                
                with gr.Row():
                    monthly = gr.Number(label="Monthly Charges", value=99.0)
                    total = gr.Number(label="Total Charges", value=99.0)

            btn_predict = gr.Button("Analyze Risk ", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("###  Analysis Result")
            output_html = gr.HTML(label="Result")
            gr.Markdown("> **High Risk:** > 50% | **Low Risk:** < 50%")

    input_list = [
        gender, senior, partner, dependents, tenure, phone, lines, internet,
        sec, backup, dev, tech, tv, mov, contract, paperless, payment, monthly, total
    ]
    
    btn_predict.click(fn=predict_churn, inputs=input_list, outputs=output_html)

if __name__ == "__main__":
    demo.launch()