"""
Recommendation Engine Module
Generates personalized rescue tips based on disease and severity.
"""

import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates rescue recommendations for plant diseases."""
    
    # Fallback recommendations if database is unavailable
    FALLBACK_RECOMMENDATIONS = {
        'general_healthy': [
            "Continue regular watering and maintain current care routine to keep the plant healthy.",
            "Monitor the plant weekly for any new symptoms or changes in leaf color.",
            "Ensure adequate sunlight exposure and proper ventilation around the plant."
        ],
        'general_mild': [
            "Remove affected leaves carefully and dispose of them separately to prevent spread.",
            "Increase air circulation by pruning nearby foliage and spacing plants apart.",
            "Apply organic fungicide (neem oil or sulfur spray) following package instructions."
        ],
        'general_moderate': [
            "Prune 10-15% of affected foliage to improve air flow and plant health.",
            "Spray with copper-based or sulfur fungicide every 7-10 days for 2-3 weeks.",
            "Adjust watering schedule - water only at base and avoid wetting leaves."
        ],
        'general_severe': [
            "Remove and destroy severely affected leaves immediately to limit disease spread.",
            "Apply systemic fungicide weekly and monitor closely for improvement.",
            "Isolate the plant from others and ensure humidity levels are controlled (40-60%)."
        ],
        'general_dying': [
            "Consider propagation from healthy parts if possible before plant dies.",
            "Apply intensive fungicide treatment daily with proper protective equipment.",
            "If no improvement in 1 week, consider removing plant to prevent pest attraction."
        ]
    }
    
    # Disease-specific tips
    DISEASE_SPECIFIC_TIPS = {
        'Late Blight': {
            'mild': [
                "Remove affected leaflets and maintain 1-2 meter spacing between plants.",
                "Apply Bordeaux mixture (1% copper sulfate) every 7 days.",
                "Mulch soil surface to prevent spore splash from soil to leaves."
            ],
            'moderate': [
                "Prune lower leaves to improve air circulation significantly.",
                "Use systemic fungicide like metalaxyl or mancozeb twice weekly.",
                "Increase plant elevation and ensure water drains away from foliage."
            ],
            'severe': [
                "Apply fungicide every 3-5 days and monitor for stem blight.",
                "Remove all infected foliage immediately upon detection.",
                "Consider staking plant vertically to maximize air exposure."
            ]
        },
        'Early Blight': {
            'mild': [
                "Prune lower leaves that show circular brown spots with concentric rings.",
                "Apply copper fungicide preventatively every 10-14 days.",
                "Remove fallen leaves from soil surface to break disease cycle."
            ],
            'moderate': [
                "Remove lower third of plant foliage systematically over 1 week.",
                "Alternate fungicides: chlorothalonil, mancozeb, and copper compounds.",
                "Support plant stakes to improve air penetration through canopy."
            ],
            'severe': [
                "Intensive pruning may be needed - remove up to 50% of foliage if necessary.",
                "Apply fungicide every 5 days with alternate active ingredients.",
                "Ensure temperatures stay below 25°C if possible to slow disease."
            ]
        },
        'Powdery Mildew': {
            'mild': [
                "Spray leaves with 1% potassium bicarbonate solution weekly.",
                "Ensure minimum 30% humidity - avoid over-watering but maintain moisture.",
                "Prune to improve air circulation and reduce disease pressure."
            ],
            'moderate': [
                "Apply sulfur-based fungicide every 7-10 days (avoid if temp > 30°C).",
                "Mix baking soda solution (1 tbsp per gallon) and spray thoroughly.",
                "Remove heavily affected leaves and dispose in sealed bag."
            ],
            'severe': [
                "Use systemic fungicide like myclobutanil or triadimefon weekly.",
                "Ensure room temperature stays 18-24°C for best fungicide efficacy.",
                "Complete defoliation may be needed for aggressive cases."
            ]
        },
        'Septoria Leaf Spot': {
            'mild': [
                "Remove infected leaves showing brown spots with gray centers.",
                "Avoid overhead watering - water at soil level only.",
                "Apply copper or mancozeb fungicide every 14 days."
            ],
            'moderate': [
                "Prune 15-20% of affected foliage and improve spacing.",
                "Apply fungicide every 10 days alternating active ingredients.",
                "Mulch soil surface and avoid direct water contact with leaves."
            ],
            'severe': [
                "Intensive pruning required - remove up to 40% of foliage.",
                "Apply systemic fungicide weekly for 4-6 weeks.",
                "Sterilize pruning tools between cuts with 10% bleach solution."
            ]
        },
        'Rust': {
            'mild': [
                "Remove leaves with early rust pustules (yellow-orange spores).",
                "Reduce humidity by improving air flow and reducing watering frequency.",
                "Apply sulfur dust or spray weekly starting at first symptoms."
            ],
            'moderate': [
                "Prune affected foliage and increase spacing between plants.",
                "Apply oil-based fungicide every 7-10 days for 3-4 weeks.",
                "Ensure humidity remains below 60% using dehumidifier if needed."
            ],
            'severe': [
                "Remove heavily infected leaves aggressively (up to 50%).",
                "Use combination of sulfur and oil spray, alternating applications.",
                "Maintain very low humidity (30-40%) and cool temperatures (15-20°C)."
            ]
        },
        'Gray Leaf Spot': {
            'mild': [
                "Remove spots with tweezers and destroy immediately in sealed bag.",
                "Apply preventative fungicide every 14 days.",
                "Maintain proper spacing for air circulation."
            ],
            'moderate': [
                "Prune infected leaves and increase spacing further.",
                "Spray with mancozeb or chlorothalonil every 10 days.",
                "Reduce nitrogen fertilizer which promotes soft, susceptible growth."
            ],
            'severe': [
                "Aggressive pruning - remove up to 50% of foliage if infected.",
                "Apply fungicide every 5-7 days for 6-8 weeks.",
                "Feed with balanced fertilizer (NPK ratio 1:1:1) to build resistance."
            ]
        }
    }
    
    @staticmethod
    def get_recommendations(disease_name, severity_level, plant_name=None, db_path=None):
        """
        Get personalized recommendations for disease and severity.
        
        Args:
            disease_name: str disease name
            severity_level: str severity level
            plant_name: str plant name (optional)
            db_path: str path to database (optional)
            
        Returns:
            list: Three recommended tips
        """
        try:
            # First try disease-specific tips
            if disease_name in RecommendationEngine.DISEASE_SPECIFIC_TIPS:
                disease_tips = RecommendationEngine.DISEASE_SPECIFIC_TIPS[disease_name]
                if severity_level in disease_tips:
                    return disease_tips[severity_level][:3]
            
            # Fall back to general recommendations
            general_key = f'general_{severity_level}'
            if general_key in RecommendationEngine.FALLBACK_RECOMMENDATIONS:
                return RecommendationEngine.FALLBACK_RECOMMENDATIONS[general_key]
            
            # Final fallback
            return RecommendationEngine.FALLBACK_RECOMMENDATIONS['general_moderate']
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return RecommendationEngine.FALLBACK_RECOMMENDATIONS['general_moderate']
    
    @staticmethod
    def get_recommendations_from_db(disease_name, severity_level, db_path):
        """
        Get recommendations from database.
        
        Args:
            disease_name: str disease name
            severity_level: str severity level
            db_path: str path to database
            
        Returns:
            list: Recommendations from database
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get disease ID
            cursor.execute("SELECT id FROM diseases WHERE name = ?", (disease_name,))
            disease_row = cursor.fetchone()
            
            if not disease_row:
                return None
            
            disease_id = disease_row[0]
            
            # Get tips for this disease and severity
            cursor.execute(
                """SELECT tip FROM tips 
                   WHERE disease_id = ? AND severity = ? 
                   ORDER BY order_index LIMIT 3""",
                (disease_id, severity_level)
            )
            
            tips = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return tips if tips else None
        
        except Exception as e:
            logger.error(f"Error getting recommendations from DB: {str(e)}")
            return None
    
    @staticmethod
    def get_plant_info(plant_name, db_path):
        """
        Get plant information from database.
        
        Args:
            plant_name: str plant name
            db_path: str path to database
            
        Returns:
            dict: Plant information
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, scientific_name, common_diseases, optimal_conditions FROM plants WHERE name = ?",
                (plant_name,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'name': plant_name,
                    'scientific_name': row[1],
                    'common_diseases': row[2],
                    'optimal_conditions': row[3]
                }
            return None
        
        except Exception as e:
            logger.error(f"Error getting plant info: {str(e)}")
            return None
    
    @staticmethod
    def save_analysis_history(analysis_data, db_path):
        """
        Save analysis result to history table.
        
        Args:
            analysis_data: dict containing analysis results
            db_path: str path to database
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """INSERT INTO analysis_history 
                   (plant_name, disease_name, severity, confidence, discoloration_percent, image_filename)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    analysis_data.get('plant_name'),
                    analysis_data.get('disease_name'),
                    analysis_data.get('severity'),
                    analysis_data.get('confidence'),
                    analysis_data.get('discoloration_percent'),
                    analysis_data.get('image_filename')
                )
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Error saving analysis history: {str(e)}")
            return False
