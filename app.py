"""
Leaf Health Check - Main Streamlit Application
AI-powered plant leaf disease detection and diagnosis system.

Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import logging
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocess import ImagePreprocessor
from utils.severity import SeverityGrader
from utils.recommendations import RecommendationEngine
from model.train import PlantDiseaseModel
from database.init_db import init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🍃 Leaf Health Check",
    page_icon="🍃",
    layout="wide"
)

# Session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'model' not in st.session_state:
    st.session_state.model = None


# Load Model
@st.cache_resource
def load_model():
    try:
        model = PlantDiseaseModel()
        return model
    except Exception as e:
        logger.error(e)
        return None


# Analyze Image
def analyze_leaf_image(image_path):

    image = ImagePreprocessor.load_image(image_path)
    original_image = image.copy()

    discoloration_data = ImagePreprocessor.detect_discoloration(image)

    processed_image = ImagePreprocessor.preprocess_for_model(image)

    st.session_state.model = load_model()

    disease_result = st.session_state.model.predict_disease(processed_image)
    plant_result = st.session_state.model.predict_plant(processed_image)

    severity_result = SeverityGrader.calculate_severity(
        discoloration_data,
        disease_result['disease'],
        disease_result['confidence']
    )

    recommendations = RecommendationEngine.get_recommendations(
        disease_result['disease'],
        severity_result['severity_level'],
        plant_result['plant']
    )

    analysis = {
        'plant': plant_result['plant'],
        'disease': disease_result['disease'],
        'severity': severity_result['severity_level'],
        'affected_percentage': severity_result['affected_percentage'],
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    }

    st.session_state.analysis_history.append(analysis)

    return analysis


# Main UI
def main():

    st.title("🍃 Leaf Health Check")
    st.subheader("AI Plant Disease Detection")

    menu = st.sidebar.selectbox(
        "Menu",
        ["Analyze Leaf", "Analysis History", "About"]
    )

    # Analyze Leaf
    if menu == "Analyze Leaf":

        uploaded_file = st.file_uploader(
            "Upload Leaf Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image("temp.jpg", caption="Uploaded Image")

            if st.button("Analyze"):

                with st.spinner("Analyzing..."):

                    result = analyze_leaf_image("temp.jpg")

                    st.success("Analysis Complete")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Plant", result['plant'])
                    col2.metric("Disease", result['disease'])
                    col3.metric("Severity", result['severity'])

                    st.write("### Affected Area")
                    st.write(result['affected_percentage'], "%")

                    st.write("### Recommendations")

                    for tip in result['recommendations']:
                        st.write("•", tip)

    # History
    elif menu == "Analysis History":

        st.header("Analysis History")

        if len(st.session_state.analysis_history) == 0:
            st.info("No analysis yet")

        else:
            for item in st.session_state.analysis_history:

                st.write("Plant:", item['plant'])
                st.write("Disease:", item['disease'])
                st.write("Severity:", item['severity'])
                st.write("Time:", item['timestamp'])
                st.write("---")

    # About
    else:

        st.header("About")

        st.write("""
        Leaf Health Check is an AI system that detects plant diseases
        using computer vision and deep learning.

        Features:
        - Upload leaf image
        - Detect plant species
        - Identify disease
        - Assess severity
        - Provide treatment tips
        """)


if __name__ == "__main__":
    main()