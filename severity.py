"""
Severity Assessment Module
Calculates disease severity based on discoloration and disease type.
"""

import logging

logger = logging.getLogger(__name__)


class SeverityGrader:
    """Grades disease severity based on multiple factors."""
    
    # Severity thresholds based on discoloration percentage
    SEVERITY_LEVELS = {
        'healthy': (0, 10),
        'mild': (11, 30),
        'moderate': (31, 60),
        'severe': (61, 80),
        'dying': (81, 100)
    }
    
    # Disease severity modifiers (adjusts threshold)
    DISEASE_MODIFIERS = {
        'Late Blight': 1.2,  # More aggressive, lower threshold
        'Powdery Mildew': 0.9,
        'Early Blight': 1.0,
        'Rust': 0.95,
        'Septoria Leaf Spot': 1.05,
        'Gray Leaf Spot': 1.1,
    }
    
    # Discoloration type weightage
    DISCOLORATION_WEIGHTS = {
        'black': 1.5,    # Most severe indicator
        'brown': 1.3,    # High severity
        'yellow': 0.8,   # Mild to moderate
        'white': 0.7,    # Usually mild
    }
    
    @staticmethod
    def calculate_severity(discoloration_data, disease_name=None, confidence=0.0):
        """
        Calculate overall severity grade.
        
        Args:
            discoloration_data: dict from detect_discoloration()
            disease_name: str disease name (optional)
            confidence: float model confidence (0-1)
            
        Returns:
            dict: Severity analysis results
        """
        affected_percent = discoloration_data['affected_percentage']
        
        # Apply disease modifier if available
        modifier = SeverityGrader.DISEASE_MODIFIERS.get(disease_name, 1.0)
        adjusted_percent = affected_percent * modifier
        
        # Calculate weighted discoloration score
        weighted_score = SeverityGrader._calculate_weighted_score(discoloration_data)
        
        # Determine severity level
        severity_level = SeverityGrader._get_severity_level(adjusted_percent)
        
        # Adjust based on weighted score if significantly different
        if weighted_score > adjusted_percent:
            severity_level = SeverityGrader._get_severity_level(weighted_score)
        
        # Calculate confidence in diagnosis
        diagnosis_confidence = SeverityGrader._calculate_diagnosis_confidence(
            affected_percent, confidence, disease_name
        )
        
        return {
            'severity_level': severity_level,
            'affected_percentage': affected_percent,
            'weighted_score': weighted_score,
            'adjusted_percentage': adjusted_percent,
            'diagnosis_confidence': diagnosis_confidence,
            'color_breakdown': {
                'black_pixels': discoloration_data['black'],
                'brown_pixels': discoloration_data['brown'],
                'yellow_pixels': discoloration_data['yellow'],
                'white_pixels': discoloration_data['white'],
            }
        }
    
    @staticmethod
    def _calculate_weighted_score(discoloration_data):
        """
        Calculate weighted discoloration score.
        
        Args:
            discoloration_data: dict from detect_discoloration()
            
        Returns:
            float: Weighted score
        """
        total_pixels = (discoloration_data['black'] + discoloration_data['brown'] +
                       discoloration_data['yellow'] + discoloration_data['white'])
        
        if total_pixels == 0:
            return 0.0
        
        weighted_sum = (
            discoloration_data['black'] * SeverityGrader.DISCOLORATION_WEIGHTS['black'] +
            discoloration_data['brown'] * SeverityGrader.DISCOLORATION_WEIGHTS['brown'] +
            discoloration_data['yellow'] * SeverityGrader.DISCOLORATION_WEIGHTS['yellow'] +
            discoloration_data['white'] * SeverityGrader.DISCOLORATION_WEIGHTS['white']
        )
        
        # Normalize to percentage
        max_weight = max(SeverityGrader.DISCOLORATION_WEIGHTS.values())
        weighted_percent = (weighted_sum / (total_pixels * max_weight)) * 100
        
        return weighted_percent
    
    @staticmethod
    def _get_severity_level(affected_percent):
        """
        Determine severity level from percentage.
        
        Args:
            affected_percent: float percentage of affected pixels
            
        Returns:
            str: Severity level
        """
        for level, (min_val, max_val) in SeverityGrader.SEVERITY_LEVELS.items():
            if min_val <= affected_percent <= max_val:
                return level
        return 'dying'
    
    @staticmethod
    def _calculate_diagnosis_confidence(affected_percent, model_confidence, disease_name=None):
        """
        Calculate overall confidence in the diagnosis.
        
        Args:
            affected_percent: float percentage affected
            model_confidence: float model prediction confidence
            disease_name: str disease name
            
        Returns:
            float: Overall confidence (0-1)
        """
        # Base confidence from model
        base_confidence = model_confidence
        
        # Increase confidence if clear visual symptoms
        if affected_percent > 5:  # Visual evidence present
            base_confidence += 0.1
        
        if affected_percent > 30:  # Clear symptoms
            base_confidence += 0.1
        
        # Decrease confidence if uncertain disease or very low symptoms
        if affected_percent < 2:  # Minimal visible symptoms
            base_confidence -= 0.15
        
        return min(1.0, max(0.0, base_confidence))
    
    @staticmethod
    def get_severity_badge(severity_level):
        """
        Get display badge for severity level.
        
        Args:
            severity_level: str severity level
            
        Returns:
            dict: Badge information
        """
        badges = {
            'healthy': {'emoji': '🟢', 'color': '#2ecc71', 'display': 'Healthy', 'description': 'Plant is in excellent health'},
            'mild': {'emoji': '🟡', 'color': '#f39c12', 'display': 'Mild', 'description': 'Minor symptoms detected'},
            'moderate': {'emoji': '🟠', 'color': '#e67e22', 'display': 'Moderate', 'description': 'Notable disease presence'},
            'severe': {'emoji': '🔴', 'color': '#e74c3c', 'display': 'Severe', 'description': 'Significant damage detected'},
            'dying': {'emoji': '⚫', 'color': '#2c3e50', 'display': 'Dying', 'description': 'Critical condition - immediate action required'},
        }
        return badges.get(severity_level, badges['healthy'])
