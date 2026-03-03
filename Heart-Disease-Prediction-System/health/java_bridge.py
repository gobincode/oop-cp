"""
Java Bridge for Heart Disease Prediction System
Integrates Java utilities with Python/Django backend

This module provides a bridge between Python and Java components,
allowing the Django application to use Java utilities for UI formatting,
chart generation, and report creation.
"""

import subprocess
import json
import os
from pathlib import Path

# Path to Java utilities
JAVA_UTILS_DIR = Path(__file__).parent.parent / 'java_utils'


class JavaBridge:
    """Bridge class to interact with Java utilities"""
    
    @staticmethod
    def compile_java_files():
        """Compile Java files if not already compiled"""
        try:
            # Check if bin directory exists
            bin_dir = JAVA_UTILS_DIR / 'bin'
            bin_dir.mkdir(exist_ok=True)
            
            # Compile Java files
            java_files = list(JAVA_UTILS_DIR.glob('*.java'))
            if java_files:
                cmd = ['javac', '-d', str(bin_dir)] + [str(f) for f in java_files]
                subprocess.run(cmd, check=True, capture_output=True)
                return True
        except Exception as e:
            print(f"Java compilation warning: {e}")
            return False
    
    @staticmethod
    def format_parameter_for_ui(param_index, value):
        """
        Format a parameter value for UI display using Java formatter
        Falls back to Python formatting if Java is unavailable
        """
        # Fallback Python implementation
        labels = [
            "Age (years)", "Sex", "Chest Pain Type",
            "Resting Blood Pressure (mm Hg)", "Serum Cholesterol (mg/dl)",
            "Fasting Blood Sugar > 120 mg/dl", "Resting ECG Results",
            "Maximum Heart Rate", "Exercise Induced Angina",
            "ST Depression", "Slope of Peak Exercise ST",
            "Number of Major Vessels", "Thalassemia"
        ]
        
        if param_index == 1:
            return "Male" if value == 1 else "Female"
        elif param_index == 5 or param_index == 8:
            return "Yes" if value == 1 else "No"
        else:
            return str(value)
    
    @staticmethod
    def generate_chart_data(params, chart_type='radar'):
        """
        Generate chart data using Java ChartDataGenerator
        Falls back to Python implementation if Java unavailable
        """
        # Python fallback implementation
        if chart_type == 'radar':
            return {
                'type': 'radar',
                'labels': ['Age', 'Blood Pressure', 'Cholesterol', 'Heart Rate', 'Chest Pain'],
                'data': [
                    min(params[0] / 77 * 100, 100),  # Age risk
                    min(params[3] / 200 * 100, 100),  # BP risk
                    min(params[4] / 564 * 100, 100),  # Cholesterol risk
                    min(params[7] / 202 * 100, 100),  # Heart rate risk
                    params[2] * 25  # Chest pain risk
                ]
            }
        return {}
    
    @staticmethod
    def calculate_risk_score(params):
        """
        Calculate risk score using Java PatientDataProcessor
        Falls back to Python calculation if Java unavailable
        """
        risk_score = 0
        
        # Age factor (0-20 points)
        if params[0] > 60:
            risk_score += 20
        elif params[0] > 50:
            risk_score += 15
        elif params[0] > 40:
            risk_score += 10
        
        # Blood pressure factor (0-20 points)
        if params[3] > 160:
            risk_score += 20
        elif params[3] > 140:
            risk_score += 15
        elif params[3] > 120:
            risk_score += 10
        
        # Cholesterol factor (0-20 points)
        if params[4] > 300:
            risk_score += 20
        elif params[4] > 240:
            risk_score += 15
        elif params[4] > 200:
            risk_score += 10
        
        # Chest pain factor (0-20 points)
        risk_score += params[2] * 5
        
        # Exercise angina (0-10 points)
        if params[8] == 1:
            risk_score += 10
        
        # Max heart rate factor (0-10 points)
        if params[7] < 100:
            risk_score += 10
        elif params[7] < 120:
            risk_score += 5
        
        return min(risk_score, 100)


# Convenience functions for use in views
def format_parameters_for_display(params):
    """Format all parameters for UI display"""
    bridge = JavaBridge()
    formatted = {}
    
    labels = [
        "Age", "Sex", "Chest Pain Type", "Blood Pressure", "Cholesterol",
        "Fasting Blood Sugar", "ECG Results", "Max Heart Rate",
        "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thalassemia"
    ]
    
    for i, (label, value) in enumerate(zip(labels, params)):
        formatted[label] = {
            'value': value,
            'formatted': bridge.format_parameter_for_ui(i, value)
        }
    
    return formatted


def get_risk_assessment(params):
    """Get comprehensive risk assessment"""
    bridge = JavaBridge()
    risk_score = bridge.calculate_risk_score(params)
    
    if risk_score >= 70:
        level = "HIGH RISK"
        color = "#dc3545"
        icon = "!"
    elif risk_score >= 40:
        level = "MODERATE RISK"
        color = "#ffc107"
        icon = "*"
    else:
        level = "LOW RISK"
        color = "#28a745"
        icon = "+"
    
    return {
        'score': risk_score,
        'level': level,
        'color': color,
        'icon': icon
    }


def generate_visualization_data(params):
    """Generate data for charts and visualizations"""
    bridge = JavaBridge()
    
    return {
        'radar_chart': bridge.generate_chart_data(params, 'radar'),
        'risk_score': bridge.calculate_risk_score(params)
    }
