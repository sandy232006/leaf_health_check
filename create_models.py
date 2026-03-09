"""
Pre-trained Model Loader
Creates and saves pre-trained models compatible with the Streamlit app.
Run this script first to generate the model files.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import PlantDiseaseModel

def create_dummy_trained_models():
    """Create dummy trained models for demonstration."""
    
    model_dir = Path(__file__).parent
    
    # Create disease model
    print("Creating disease classification model...")
    disease_trainer = PlantDiseaseModel(architecture='efficientnet')
    disease_trainer.build_disease_model()
    disease_model_path = model_dir / 'leaf_disease_model.h5'
    disease_trainer.disease_model.save(str(disease_model_path))
    print(f"✓ Disease model saved: {disease_model_path}")
    
    # Create plant model  
    print("Creating plant species classification model...")
    plant_trainer = PlantDiseaseModel(architecture='efficientnet')
    plant_trainer.build_plant_model()
    plant_model_path = model_dir / 'plant_species_model.h5'
    plant_trainer.plant_model.save(str(plant_model_path))
    print(f"✓ Plant model saved: {plant_model_path}")
    
    print("\n✓ Models created successfully!")
    print("\nNote: These are untrained models for testing purposes.")
    print("For production, train on PlantVillage dataset: https://www.kaggle.com/datasets/emmarex/plantvillage-dataset")


if __name__ == "__main__":
    create_dummy_trained_models()
