from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import traceback

# Import logic from AI and Visualisation folders
from ai.preprocessing.preprocessing_engine import PreprocessingEngine
from ai.model.model_predictor import ModelPredictor
from Visualisation.format_for_charts import (
    format_for_charts, 
    format_24h_summary,
    calculate_statistics
)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DATA_PATH = os.path.join(BASE_DIR, 'Client-data', 'nilm_ready_dataset.parquet')
MODEL_PATH = os.path.join(BASE_DIR, 'ai', 'model', 'transformer_heatpump_best.pth')
CACHE_DIR = os.path.join(BASE_DIR, '24h-cache')

# Ensure cache folder exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables for caching (lazy loading)
_preprocessor = None
_model_predictor = None


def get_preprocessor():
    """Lazy loading of preprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = PreprocessingEngine()
    return _preprocessor


def get_model_predictor():
    """Lazy loading of model predictor."""
    global _model_predictor
    if _model_predictor is None:
        _model_predictor = ModelPredictor(MODEL_PATH)
    return _model_predictor


@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "NILM Energy Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/run-pipeline",
            "/predict",
            "/cached-predictions",
            "/health"
        ]
    })


@app.route('/health')
def health():
    """Endpoint to check service status."""
    checks = {
        "api": True,
        "model_loaded": _model_predictor is not None,
        "data_available": os.path.exists(CLIENT_DATA_PATH)
    }
    
    status = "healthy" if all(checks.values()) else "degraded"
    
    return jsonify({
        "status": status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/run-pipeline')
def run_pipeline():
    """
    Execute the complete NILM prediction pipeline.
    
    Pipeline:
        1. Read raw data from Client-data
        2. Preprocess the data
        3. Run model inference
        4. Save to cache
        5. Return formatted results
    """
    try:
        # 1. Read raw data
        if not os.path.exists(CLIENT_DATA_PATH):
            return jsonify({
                "error": "Dataset not found",
                "path": CLIENT_DATA_PATH
            }), 404
        
        raw_df = pd.read_parquet(CLIENT_DATA_PATH)
        
        # 2. Preprocessing
        preprocessor = get_preprocessor()
        cleaned = preprocessor.clean_data(raw_df)
        sequences = preprocessor.prepare_sequences(cleaned)
        
        # 3. Model inference
        model_predictor = get_model_predictor()
        predictions = model_predictor.predict_batch(sequences)
        
        # 4. Save to cache
        cache_path = os.path.join(CACHE_DIR, 'predictions.json')
        predictions_dict = predictions.to_dict(orient='records')
        with open(cache_path, 'w') as f:
            json.dump({
                "predictions": predictions_dict,
                "generated_at": datetime.now().isoformat(),
                "model": "transformer_heatpump"
            }, f, indent=2)
        
        # 5. Format for frontend
        output = format_for_charts(predictions)
        
        return jsonify(output)
    
    except Exception as e:
        app.logger.error(f"Pipeline error: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predictions on custom data sent by the client.
    
    JSON Body:
        {
            "data": [[...], [...], ...],  // Array of sequences
            "format": "array" | "timeseries"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"error": "'data' field required"}), 400
        
        input_data = np.array(data['data'], dtype=np.float32)
        
        model_predictor = get_model_predictor()
        predictions = model_predictor.get_prediction(input_data)
        
        output = format_for_charts(predictions)
        return jsonify(output)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/cached-predictions')
def get_cached_predictions():
    """
    Return predictions saved in cache.
    Useful for frontend to quickly retrieve latest results.
    """
    cache_path = os.path.join(CACHE_DIR, 'predictions.json')
    
    if not os.path.exists(cache_path):
        return jsonify({
            "error": "No predictions in cache. Run /run-pipeline first."
        }), 404
    
    with open(cache_path, 'r') as f:
        cached_data = json.load(f)
    
    return jsonify(cached_data)


@app.route('/statistics')
def get_statistics():
    """
    Return aggregated statistics from latest predictions.
    """
    cache_path = os.path.join(CACHE_DIR, 'predictions.json')
    
    if not os.path.exists(cache_path):
        return jsonify({
            "error": "No predictions in cache"
        }), 404
    
    with open(cache_path, 'r') as f:
        cached_data = json.load(f)
    
    predictions_df = pd.DataFrame(cached_data['predictions'])
    stats = calculate_statistics(predictions_df)
    
    return jsonify({
        "statistics": stats,
        "generated_at": cached_data.get('generated_at'),
        "model": cached_data.get('model')
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("NILM Energy Prediction API")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Data path: {CLIENT_DATA_PATH}")
    print(f"Cache dir: {CACHE_DIR}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)