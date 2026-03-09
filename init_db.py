"""
Database Initialization Script
Creates SQLite database with plants, diseases, and tips tables.
"""

import sqlite3
import os
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "plants.db"


def init_database():
    """Initialize the SQLite database with schema."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Plants table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            scientific_name TEXT,
            common_diseases TEXT,
            optimal_conditions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Diseases table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            plant_id INTEGER,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (plant_id) REFERENCES plants(id)
        )
    """)

    # Rescue Tips table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_id INTEGER NOT NULL,
            severity TEXT NOT NULL,
            tip TEXT NOT NULL,
            order_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (disease_id) REFERENCES diseases(id)
        )
    """)

    # Analysis history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_name TEXT,
            disease_name TEXT,
            severity TEXT,
            confidence REAL,
            discoloration_percent REAL,
            image_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    print("Database tables created successfully!")

    # Insert sample data
    insert_sample_data(cursor)
    conn.commit()
    conn.close()


def insert_sample_data(cursor):
    """Insert sample plant and disease data."""
    
    # Sample plants
    plants_data = [
        ("Tomato", "Solanum lycopersicum", "Early Blight, Late Blight, Septoria Leaf Spot", "Warm, 21-29°C, Well-drained soil"),
        ("Potato", "Solanum tuberosum", "Early Blight, Late Blight, Verticillium Wilt", "Cool, 16-18°C, High humidity tolerance"),
        ("Apple", "Malus domestica", "Powdery Mildew, Scab, Rust", "Cool, 15-20°C, Well-drained soil"),
        ("Corn", "Zea mays", "Northern Corn Leaf Blight, Gray Leaf Spot, Rust", "Warm, 25-30°C, Moderate moisture"),
        ("Wheat", "Triticum aestivum", "Septoria Leaf Blotch, Powdery Mildew, Stripe Rust", "Cool, 15-20°C, Moderate water"),
    ]
    
    disease_data = [
        ("Early Blight", "Fungal disease affecting tomato and potato leaves"),
        ("Late Blight", "Severe fungal disease causing rapid plant death"),
        ("Septoria Leaf Spot", "Fungal disease with characteristic spots"),
        ("Powdery Mildew", "White fungal coating on leaves"),
        ("Northern Corn Leaf Blight", "Fungal disease specific to corn"),
        ("Gray Leaf Spot", "Gray lesions on corn leaves"),
        ("Rust", "Fungal disease with rusty appearance"),
        ("Leaf Scab", "Fungal disease with scab-like spots"),
    ]
    
    try:
        # Check if data exists
        cursor.execute("SELECT COUNT(*) FROM plants")
        if cursor.fetchone()[0] == 0:
            for plant in plants_data:
                cursor.execute(
                    "INSERT INTO plants (name, scientific_name, common_diseases, optimal_conditions) VALUES (?, ?, ?, ?)",
                    plant
                )
            
            # Get plant IDs and insert diseases
            for i, disease in enumerate(disease_data):
                plant_id = (i % len(plants_data)) + 1
                cursor.execute(
                    "INSERT INTO diseases (name, plant_id, description) VALUES (?, ?, ?)",
                    (disease[0], plant_id, disease[1])
                )
                
                # Insert tips for each disease
                disease_id = cursor.lastrowid
                tips = [
                    f"Prune affected leaves and improve air circulation around the {plants_data[plant_id-1][0]}.",
                    f"Apply fungicide spray with proper dilution and coverage to all leaf surfaces.",
                    f"Water from the base only, avoid wetting foliage to prevent disease spread."
                ]
                
                for idx, tip in enumerate(tips):
                    for severity in ["Mild", "Moderate", "Severe"]:
                        cursor.execute(
                            "INSERT INTO tips (disease_id, severity, tip, order_index) VALUES (?, ?, ?, ?)",
                            (disease_id, severity, tip, idx + 1)
                        )
            
            print("Sample data inserted successfully!")
        else:
            print("Database already contains data. Skipping sample data insertion.")
    except sqlite3.IntegrityError:
        print("Data already exists in database.")


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DATABASE_PATH)


if __name__ == "__main__":
    init_database()
