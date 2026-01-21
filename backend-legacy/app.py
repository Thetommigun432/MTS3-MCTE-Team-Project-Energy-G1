from flask import Flask, jsonify, request
from flask.cli import load_dotenv
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import traceback
import redis

# Import logic from AI and Visualisation folders
from ai.preprocessing.preprocessing_engine import PreprocessingEngine
from ai.model.model_predictor import ModelPredictor
from cache_24h.cache_manager import CacheManager, get_cache_manager
from Visualisation.format_for_charts import (
    format_for_charts, 
    format_24h_summary,
    calculate_statistics
)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DATA_PATH = os.path.join(BASE_DIR, 'Client-data', 'nilm_ready_dataset.parquet')
MODEL_PATH = os.path.join(BASE_DIR, 'ai', 'model', 'transformer_heatpump_best.pth')

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables for caching (lazy loading)
_preprocessor = None
_model_predictor = None
_cache_manager = None

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
            "/stream",
            "/stream/latest",
            "/cache-status",
            "/health"
        ]
    })

@app.route('/health')
def health():
    """Endpoint to check service status."""
    cache_manager = get_cache_manager()
    
    checks = {
        "api": True,
        "model_loaded": _model_predictor is not None,
        "data_available": os.path.exists(CLIENT_DATA_PATH),
        "redis_connected": cache_manager.is_connected(),
        "cache_worker_active": cache_manager.is_worker_active()
    }
    
    status = "healthy" if all(checks.values()) else "degraded"
    
    return jsonify({
        "status": status,
        "checks": checks,
        "cache_stream_length": cache_manager.get_stream_length(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/run-pipeline')
def run_pipeline():
    """
    Execute the complete NILM prediction pipeline.
    
    NOTA: In produzione, il Cache Worker processa i dati automaticamente.
    Questo endpoint è per debug/testing manuale.
    
    Pipeline:
        1. Read raw data from Client-data
        2. Preprocess the data
        3. Run model inference
        4. Save to Redis cache
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
        sequence = preprocessor.prepare_sequences(cleaned)[0]

        # 3. Salva i dati puliti in Redis
        cache_manager = get_cache_manager()
        if isinstance(cleaned, pd.DataFrame):
            cache_manager.set_cache('energy:cleaned', cleaned.to_json(orient='records'))
        else:
            cache_manager.set_cache('energy:cleaned', json.dumps(cleaned))

        # 4. Model inference
        model_predictor = get_model_predictor()
        predictions = model_predictor.predict_batch(sequence)

        # 5. Salva predizioni in Redis
        predictions_dict = predictions.to_dict(orient='records')
        cache_manager.set_cache('energy:predictions', json.dumps({
            "predictions": predictions_dict,
            "generated_at": datetime.now().isoformat(),
            "model": "transformer_heatpump"
        }))
        
        # 6. Format for frontend
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
    Return predictions saved in Redis cache.
    The Cache Worker keeps these updated automatically.
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get_predictions()
    
    if not cached_data:
        return jsonify({
            "error": "No predictions in cache. Cache Worker may not be running.",
            "worker_active": cache_manager.is_worker_active()
        }), 404
    
    return jsonify(cached_data)


@app.route('/stream')
def get_stream():
    """
    Return the real-time data stream from Redis.
    Contains the last 24 hours of processed data.
    
    Query params:
        limit: Number of elements to return (default: all)
    """
    cache_manager = get_cache_manager()
    limit = request.args.get('limit', type=int)
    
    stream_data = cache_manager.get_stream(limit=limit)
    
    return jsonify({
        "stream": stream_data,
        "count": len(stream_data),
        "total_in_cache": cache_manager.get_stream_length(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/stream/latest')
def get_stream_latest():
    """
    Return the latest data points from the stream.
    
    Query params:
        count: Number of elements to return (default: 10)
    """
    cache_manager = get_cache_manager()
    count = request.args.get('count', default=10, type=int)
    
    latest_data = cache_manager.get_latest(count=count)
    
    return jsonify({
        "data": latest_data,
        "count": len(latest_data),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/cache-status')
def get_cache_status():
    """
    Return the complete status of the cache system.
    """
    cache_manager = get_cache_manager()
    return jsonify(cache_manager.get_status())


@app.route('/statistics')
def get_statistics():
    """
    Return aggregated statistics from latest predictions.
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get_predictions()
    
    if not cached_data:
        return jsonify({
            "error": "No predictions in cache"
        }), 404
    
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
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)