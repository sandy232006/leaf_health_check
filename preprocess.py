"""
Image Preprocessing Module
Handles image loading, validation, resizing, and color detection.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing and validation."""
    
    VALID_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    MAX_IMAGE_SIZE = 25 * 1024 * 1024  # 25MB
    TARGET_SIZE = (224, 224)
    
    @staticmethod
    def validate_image(image_path):
        """
        Validate image format and size.
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            file_path = Path(image_path)
            
            # Check file extension
            if file_path.suffix.lower() not in ImagePreprocessor.VALID_FORMATS:
                return False, f"Invalid format. Supported: {', '.join(ImagePreprocessor.VALID_FORMATS)}"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > ImagePreprocessor.MAX_IMAGE_SIZE:
                return False, f"File too large. Max: 25MB, Got: {file_size / (1024*1024):.2f}MB"
            
            # Try to open and validate
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            return True, "Valid image"
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    @staticmethod
    def load_image(image_path):
        """
        Load image and convert to RGB.
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Image in RGB format
        """
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    @staticmethod
    def resize_image(image, target_size=TARGET_SIZE):
        """
        Resize image to target dimensions.
        
        Args:
            image: np.ndarray image
            target_size: tuple (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        if isinstance(image, np.ndarray):
            return cv2.resize(image, (target_size[1], target_size[0]))
        else:
            image = Image.fromarray(image)
            image = image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.array(image)
    
    @staticmethod
    def normalize_image(image):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: np.ndarray image
            
        Returns:
            np.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def detect_discoloration(image):
        """
        Detect yellow, brown, black, and white regions.
        
        Args:
            image: np.ndarray image in RGB
            
        Returns:
            dict: Discoloration analysis results
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        height, width = image.shape[:2]
        total_pixels = height * width
        
        results = {
            'yellow': 0,
            'brown': 0,
            'black': 0,
            'white': 0,
            'total_affected': 0,
            'affected_percentage': 0.0,
            'masks': {}
        }
        
        # Yellow detection (H: 20-40, S: 100-255, V: 100-255)
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
        results['yellow'] = np.count_nonzero(yellow_mask)
        results['masks']['yellow'] = yellow_mask
        
        # Brown detection (H: 10-20, S: 100-255, V: 50-200)
        brown_mask = cv2.inRange(hsv, (10, 100, 50), (20, 255, 200))
        results['brown'] = np.count_nonzero(brown_mask)
        results['masks']['brown'] = brown_mask
        
        # Black detection (V: 0-50, S: 0-255)
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        results['black'] = np.count_nonzero(black_mask)
        results['masks']['black'] = black_mask
        
        # White detection (S: 0-30, V: 200-255)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        results['white'] = np.count_nonzero(white_mask)
        results['masks']['white'] = white_mask
        
        # Combined affected area
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, brown_mask),
                                       cv2.bitwise_or(black_mask, white_mask))
        results['total_affected'] = np.count_nonzero(combined_mask)
        results['affected_percentage'] = (results['total_affected'] / total_pixels) * 100
        results['masks']['combined'] = combined_mask
        
        return results
    
    @staticmethod
    def preprocess_for_model(image, target_size=TARGET_SIZE):
        """
        Complete preprocessing pipeline for model input.
        
        Args:
            image: np.ndarray image
            target_size: tuple (height, width)
            
        Returns:
            np.ndarray: Preprocessed image ready for model
        """
        # Resize
        resized = ImagePreprocessor.resize_image(image, target_size)
        
        # Normalize
        normalized = ImagePreprocessor.normalize_image(resized)
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    @staticmethod
    def highlight_discolored_regions(image, mask, color=(255, 0, 0)):
        """
        Create visualization of discolored regions.
        
        Args:
            image: np.ndarray image
            mask: np.ndarray binary mask
            color: tuple RGB color for highlighting
            
        Returns:
            np.ndarray: Image with highlighted regions
        """
        image_copy = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_copy, contours, -1, color, 2)
        
        # Add semi-transparent overlay
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, color, -1)
        image_copy = cv2.addWeighted(image_copy, 0.7, overlay, 0.3, 0)
        
        return image_copy
