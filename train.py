"""
Model Training Module
Trains CNN model on plant leaf disease dataset.
Supports multiple architectures: ResNet50, EfficientNetB0, Custom CNN
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)


# Disease classes for model output
DISEASE_CLASSES = [
    'healthy',
    'early_blight',
    'late_blight',
    'septoria_leaf_spot',
    'powdery_mildew',
    'rust',
    'gray_leaf_spot',
    'leaf_scab'
]

# Plant species classes
PLANT_CLASSES = [
    'tomato',
    'potato',
    'apple',
    'corn',
    'wheat'
]


class PlantDiseaseModel:
    """Plant disease detection CNN model."""
    
    DISEASE_CLASSES = DISEASE_CLASSES
    PLANT_CLASSES = PLANT_CLASSES
    
    def __init__(self, architecture='efficientnet'):
        """
        Initialize model.
        
        Args:
            architecture: str - 'resnet50', 'efficientnet', or 'custom'
        """
        self.architecture = architecture
        self.disease_model = None
        self.plant_model = None
        self.input_shape = (224, 224, 3)
    
    def build_disease_model(self):
        """Build disease classification model."""
        
        if self.architecture == 'resnet50':
            self.disease_model = self._build_resnet_disease()
        elif self.architecture == 'efficientnet':
            self.disease_model = self._build_efficientnet_disease()
        else:
            self.disease_model = self._build_custom_disease()
        
        self.disease_model.summary()
        return self.disease_model
    
    def build_plant_model(self):
        """Build plant species classification model."""
        
        if self.architecture == 'resnet50':
            self.plant_model = self._build_resnet_plant()
        elif self.architecture == 'efficientnet':
            self.plant_model = self._build_efficientnet_plant()
        else:
            self.plant_model = self._build_custom_plant()
        
        self.plant_model.summary()
        return self.plant_model
    
    def _build_resnet_disease(self):
        """Build ResNet50-based disease classification model."""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model weights
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(len(self.DISEASE_CLASSES), activation='softmax')
        ])
        
        return model
    
    def _build_efficientnet_disease(self):
        """Build EfficientNetB0-based disease classification model."""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model weights
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(len(self.DISEASE_CLASSES), activation='softmax')
        ])
        
        return model
    
    def _build_custom_disease(self):
        """Build custom CNN for disease classification."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.4),
            layers.Dense(len(self.DISEASE_CLASSES), activation='softmax')
        ])
        
        return model
    
    def _build_resnet_plant(self):
        """Build ResNet50-based plant species model."""
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.PLANT_CLASSES), activation='softmax')
        ])
        
        return model
    
    def _build_efficientnet_plant(self):
        """Build EfficientNetB0-based plant species model."""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.PLANT_CLASSES), activation='softmax')
        ])
        
        return model
    
    def _build_custom_plant(self):
        """Build custom CNN for plant species classification."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.PLANT_CLASSES), activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile model with optimizer and loss."""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline."""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
    
    def train(self, X_train, y_train, X_val, y_val, model_name='disease', epochs=50, batch_size=32):
        """
        Train model.
        
        Args:
            X_train: np.ndarray training images
            y_train: np.ndarray training labels (one-hot)
            X_val: np.ndarray validation images
            y_val: np.ndarray validation labels (one-hot)
            model_name: str 'disease' or 'plant'
            epochs: int number of epochs
            batch_size: int batch size
            
        Returns:
            History object
        """
        
        if model_name == 'disease':
            if self.disease_model is None:
                self.build_disease_model()
            model = self.disease_model
        else:
            if self.plant_model is None:
                self.build_plant_model()
            model = self.plant_model
        
        self.compile_model(model)
        
        # Data augmentation
        datagen = self.create_data_augmentation()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'model_{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_name='disease', path='model/'):
        """Save model."""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        if model_name == 'disease':
            model = self.disease_model
            filename = 'leaf_disease_model.h5'
        else:
            model = self.plant_model
            filename = 'plant_species_model.h5'
        
        filepath = path / filename
        model.save(str(filepath))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, model_name='disease', path='model/'):
        """Load pre-trained model."""
        path = Path(path)
        
        if model_name == 'disease':
            filename = 'leaf_disease_model.h5'
        else:
            filename = 'plant_species_model.h5'
        
        filepath = path / filename
        
        if filepath.exists():
            if model_name == 'disease':
                self.disease_model = keras.models.load_model(str(filepath))
            else:
                self.plant_model = keras.models.load_model(str(filepath))
            logger.info(f"Model loaded from {filepath}")
            return True
        else:
            logger.warning(f"Model file not found: {filepath}")
            return False
    
    def predict_disease(self, image):
        """
        Predict disease from image.
        
        Args:
            image: np.ndarray preprocessed image
            
        Returns:
            dict: Prediction results
        """
        if self.disease_model is None:
            raise ValueError("Disease model not loaded. Call load_model() or build_disease_model() first.")
        
        predictions = self.disease_model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        disease = self.DISEASE_CLASSES[class_idx]
        
        return {
            'disease': disease,
            'confidence': confidence,
            'predictions': {self.DISEASE_CLASSES[i]: float(predictions[0][i]) 
                           for i in range(len(self.DISEASE_CLASSES))}
        }
    
    def predict_plant(self, image):
        """
        Predict plant species from image.
        
        Args:
            image: np.ndarray preprocessed image
            
        Returns:
            dict: Prediction results
        """
        if self.plant_model is None:
            raise ValueError("Plant model not loaded. Call load_model() or build_plant_model() first.")
        
        predictions = self.plant_model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        plant = self.PLANT_CLASSES[class_idx]
        
        return {
            'plant': plant,
            'confidence': confidence,
            'predictions': {self.PLANT_CLASSES[i]: float(predictions[0][i]) 
                           for i in range(len(self.PLANT_CLASSES))}
        }


# Example training script
if __name__ == "__main__":
    
    # Create model
    model_trainer = PlantDiseaseModel(architecture='efficientnet')
    
    # Build models
    logger.info("Building disease classification model...")
    model_trainer.build_disease_model()
    
    logger.info("Building plant species classification model...")
    model_trainer.build_plant_model()
    
    logger.info("Models built successfully!")
    
    # For actual training, you would:
    # 1. Load dataset from PlantVillage or similar
    # 2. Preprocess images
    # 3. Create train/val/test splits
    # 4. Train models
    # 5. Evaluate and save
    
    print("""
    To train the models:
    
    1. Download PlantVillage dataset: https://www.kaggle.com/datasets/emmarex/plantvillage-dataset
    2. Organize images into directories by disease
    3. Preprocess and split data
    4. Run training script
    5. Save trained models to model/ directory
    
    Example:
    from model.train import PlantDiseaseModel
    trainer = PlantDiseaseModel('efficientnet')
    trainer.build_disease_model()
    trainer.train(X_train, y_train, X_val, y_val, epochs=50)
    trainer.save_model('disease', 'model/')
    """)
