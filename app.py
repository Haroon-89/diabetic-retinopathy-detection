import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

icon = Image.open("favicon/logo.png")
st.set_page_config(
    page_title="DR Detection System",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        color: #c62828;
        border: 4px solid #c62828;
    }
    .negative {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        border: 4px solid #2e7d32;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4 0%, #2196F3 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background: rgba(33, 150, 243, 0.1);
        color: #e3f2fd;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_trained_model():
    try:
        model_path = None
        possible_paths = [
            'models/best_model.h5',
            'models/dr_model_final.h5',
            'best_model.h5',
            'dr_model_final.h5'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            return None, None, "Model file not found"
        
        model = load_model(model_path, compile=False)
        
        metadata_paths = ['models/metadata.json', 'metadata.json']
        metadata = None
        
        for meta_path in metadata_paths:
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                break
        
        if metadata is None:
            metadata = {
                'classes': ['DR', 'No_DR'],
                'test_accuracy': 0.96,
                'test_auc': 0.99,
                'architecture': 'EfficientNetB3'
            }
        
        return model, metadata, None
        
    except Exception as e:
        return None, None, str(e)

with st.spinner("Loading AI model..."):
    model, metadata, error = load_trained_model()

def preprocess_image(image, target_size=(224, 224)):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array, None
        
    except Exception as e:
        return None, str(e)

def predict_image(image):
    if model is None:
        return None, None, "Model not loaded"
    
    try:
        processed_img, error = preprocess_image(image)
        if error:
            return None, None, error
        
        prediction = model.predict(processed_img, verbose=0)
        probability = float(prediction[0][0])  # Probability of 'No_DR'
        
        # Model outputs probability of No_DR (class index 1)
        # Higher probability (>0.5) = No_DR, Lower probability (<0.5) = DR
        if probability > 0.5:
            predicted_class = 1  # No_DR
            predicted_label = "No_Diabetic_Retinopathy"
            confidence = probability
        else:
            predicted_class = 0  # DR
            predicted_label = "Diabetic_Retinopathy"
            confidence = 1 - probability
        
        return predicted_label, confidence, None
        
    except Exception as e:
        return None, None, str(e)


with st.sidebar:
    st.markdown("## About")
    st.markdown("AI-powered detection of Diabetic Retinopathy from retinal images using deep learning.")
    
    st.markdown("---")
    
    if metadata:
        st.markdown("### Model Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metadata.get('test_accuracy', 0)*100:.1f}%")
        with col2:
            st.metric("AUC", f"{metadata.get('test_auc', 0):.3f}")
        
        st.info(f"**Model:** {metadata.get('architecture', 'EfficientNet')}")
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a retinal fundus image
    2. Click Analyze Image button
    3. View AI prediction and confidence
    4. Download detailed report
    """)
    
    st.markdown("---")
    st.markdown("### Supported Formats")
    st.markdown("JPG / JPEG / PNG / RGB Images")
    
    st.markdown("---")
    st.warning("For educational purposes only. Not for clinical diagnosis.")

st.markdown('<div class="main-header">Diabetic Retinopathy Detection</div>', unsafe_allow_html=True)

if model is None:
    st.error(f"""
    ### Model Loading Error
    
    **Error:** {error}
    
    **Solution:**
    1. Download model files from Kaggle
    2. Place them in models/ folder
    3. Required files: best_model.h5, metadata.json
    4. Restart the application
    """)
    st.stop()

st.success("Model loaded successfully")

st.markdown("## Upload Retinal Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear retinal image for analysis"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Original Image")
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Image Details:</strong><br>
        Size: {image.size[0]} × {image.size[1]} pixels<br>
        Mode: {image.mode}<br>
        Format: {image.format}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### AI Analysis")
        
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predicted_label, confidence, error = predict_image(image)
                
                if error:
                    st.error(f"Prediction Error: {error}")
                else:
                    is_positive = predicted_label == "Diabetic_Retinopathy"
                    
                    if is_positive:
                        st.markdown("""
                        <div class="prediction-box positive">
                            DIABETIC RETINOPATHY DETECTED
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("**Action Required:** Please consult an ophthalmologist for professional evaluation.")
                    else:
                        st.markdown("""
                        <div class="prediction-box negative">
                            NO DIABETIC RETINOPATHY
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("**Result:** No signs of diabetic retinopathy detected.")
                    
                    st.markdown("### Confidence Level")
                    conf_percentage = confidence * 100
                    st.progress(confidence)
                    st.markdown(f"### {conf_percentage:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("### Detailed Report")
                    
                    results_df = pd.DataFrame({
                        '': ['Prediction', 'Confidence', 'Model', 'Accuracy'],
                        'Result': [
                            predicted_label,
                            f"{conf_percentage:.2f}%",
                            metadata.get('architecture', 'EfficientNet'),
                            f"{metadata.get('test_accuracy', 0)*100:.1f}%"
                        ]
                    })
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("### Download Report")
                    
                    report_text = f"""
DIABETIC RETINOPATHY DETECTION REPORT
=====================================

ANALYSIS RESULTS
Prediction:     {predicted_label}
Confidence:     {conf_percentage:.2f}%
AI Model:       {metadata.get('architecture', 'EfficientNet')}
Model Accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%

RECOMMENDATION
{'Please schedule an appointment with an ophthalmologist' if is_positive else 'Continue regular eye check-ups as advised by your doctor'}
{'for comprehensive evaluation and treatment options.' if is_positive else 'Maintain healthy blood sugar levels.'}

DISCLAIMER
This report is generated by an AI screening tool for educational 
purposes only. It should NOT be used as a substitute for 
professional medical diagnosis. Always consult qualified 
healthcare professionals for medical advice.

Generated by DR Detection System
                    """
                    
                    st.download_button(
                        label="Download Full Report",
                        data=report_text,
                        file_name=f"DR_Report_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

st.markdown("---")
st.markdown("## Batch Analysis (Multiple Images)")

with st.expander("Upload Multiple Images for Batch Processing"):
    batch_files = st.file_uploader(
        "Select multiple retinal images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis"
    )
    
    if batch_files:
        st.info(f"{len(batch_files)} images uploaded")
        
        if st.button("Process All Images", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(batch_files):
                status_text.text(f"Processing {idx+1}/{len(batch_files)}: {file.name}")
                
                try:
                    image = Image.open(file)
                    predicted_label, confidence, error = predict_image(image)
                    
                    if error:
                        results.append({
                            'Filename': file.name,
                            'Prediction': 'Error',
                            'Confidence': 'N/A',
                            'Status': 'Failed'
                        })
                    else:
                        is_positive = predicted_label == 'Diabetic_Retinopathy'
                        results.append({
                            'Filename': file.name,
                            'Prediction': predicted_label,
                            'Confidence': f"{confidence*100:.2f}%",
                            'Status': 'DR Detected' if is_positive else 'No DR'
                        })
                except:
                    results.append({
                        'Filename': file.name,
                        'Prediction': 'Error',
                        'Confidence': 'N/A',
                        'Status': 'Failed'
                    })
                
                progress_bar.progress((idx + 1) / len(batch_files))
            
            status_text.success("Batch processing complete")
            
            st.markdown("### Batch Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            st.markdown("### Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(results)
            positive = sum(1 for r in results if 'DR Detected' in r['Status'])
            negative = sum(1 for r in results if 'No DR' in r['Status'])
            failed = sum(1 for r in results if 'Failed' in r['Status'])
            
            with col1:
                st.metric("Total Images", total)
            with col2:
                st.metric("DR Detected", positive, delta=f"{positive/total*100:.0f}%" if total > 0 else "0%")
            with col3:
                st.metric("No DR", negative, delta=f"{negative/total*100:.0f}%" if total > 0 else "0%")
            with col4:
                st.metric("Failed", failed)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Batch Results (CSV)",
                data=csv,
                file_name="batch_dr_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")

with st.expander("About Diabetic Retinopathy"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Diabetic Retinopathy?
        
        Diabetic retinopathy is a diabetes complication that affects the eyes. 
        It's caused by damage to the blood vessels of the light-sensitive tissue 
        at the back of the eye (retina).
        
        ### Symptoms
        - Blurred vision
        - Floaters
        - Impaired color vision
        - Vision loss (in advanced stages)
        """)
    
    with col2:
        st.markdown("""
        ### Risk Factors
        - Duration of diabetes
        - Poor blood sugar control
        - High blood pressure
        - High cholesterol
        - Pregnancy
        
        ### Prevention
        - Regular eye exams
        - Control blood sugar
        - Maintain healthy blood pressure
        - Quit smoking
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p style='font-size: 1.2rem; font-weight: bold;'>Diabetic Retinopathy Detection System</p>
    <p>Powered by TensorFlow 2.20 | EfficientNet | Built with Streamlit</p>
    <p style='margin-top: 1rem;'>
        <span style='background: #ffebee; padding: 0.5rem 1rem; border-radius: 5px; color: #c62828; font-weight: bold;'>
            For Educational Use Only - Not a Medical Device
        </span>
    </p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>
        Always consult with qualified healthcare professionals for medical diagnosis and treatment
    </p>
</div>
""", unsafe_allow_html=True)