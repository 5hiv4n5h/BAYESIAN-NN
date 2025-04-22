import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import datetime
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from werkzeug.utils import secure_filename

# Import functions from the BNN model script
from bnn import (
    StockDataManager, 
    calculate_technical_indicators,
    calculate_enhanced_technical_indicators,
    fetch_news_sentiment_for_multiple_stocks,
    preprocess_multiple_stocks,
    create_features_target_multi_stock,
    download_or_load_stock_data,
    predict_today_for_multiple_stocks,
    predict_multiple_timeframes,
    calculate_feature_importance,
    calculate_market_sentiment,
    add_macroeconomic_features,
    implement_mock_trading_simulation,
    negative_log_likelihood
)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configuration
UPLOAD_FOLDER = 'uploaded_models'
ALLOWED_EXTENSIONS = {'keras', 'h5', 'pkl'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Global storage for loaded models
loaded_models = {}
loaded_scalers = {}
loaded_sequences = {}
feature_importance_data = {}
market_sentiment_data = None

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """API root endpoint with documentation"""
    return jsonify({
        "message": "Stock Price Prediction API with Bayesian Neural Networks",
        "version": "1.0",
        "endpoints": {
            "GET /": "API information",
            "GET /symbols": "Get available stock symbols",
            "GET /models": "Get list of loaded models",
            "POST /models/upload": "Upload a trained model file",
            "POST /models/load": "Load models for specified symbols",
            "GET /predict/{symbol}": "Get prediction for specific symbol",
            "POST /predict/batch": "Get predictions for multiple symbols",
            "GET /predict/timeframes/{symbol}": "Get multi-timeframe predictions for a symbol",
            "POST /predict/timeframes/batch": "Get multi-timeframe predictions for multiple symbols",
            "GET /history/{symbol}": "Get historical data and analysis",
            "GET /sentiment": "Get market sentiment data",
            "GET /feature_importance/{symbol}": "Get feature importance data for a symbol",
            "POST /simulation": "Run trading simulation based on predictions",
            "POST /train": "Train model for specified symbols"
        }
    })

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Return available stock symbols"""
    # Default symbols that the API supports
    default_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    
    # Add any symbols that have models loaded
    all_symbols = list(set(default_symbols + list(loaded_models.keys())))
    
    return jsonify({
        "symbols": all_symbols,
        "loaded_models": list(loaded_models.keys())
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Return information about loaded models"""
    model_info = {}
    
    for symbol, model in loaded_models.items():
        if model is not None:
            try:
                model_info[symbol] = {
                    "loaded": True,
                    "type": type(model).__name__,
                    "layers": len(model.layers),
                    "input_shape": str(model.input_shape),
                    "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                model_info[symbol] = {
                    "loaded": True,
                    "error": str(e)
                }
    
    return jsonify({
        "loaded_models": model_info,
        "count": len(loaded_models)
    })

@app.route('/models/upload', methods=['POST'])
def upload_model():
    """Upload a trained model file"""
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    file = request.files['model']
    symbol = request.form.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{symbol}_model.keras")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Load the model
            model = tf.keras.models.load_model(
                file_path, 
                custom_objects={'negative_log_likelihood': negative_log_likelihood}
            )
            loaded_models[symbol] = model
            
            return jsonify({
                "message": f"Model for {symbol} uploaded and loaded successfully",
                "model_info": {
                    "symbol": symbol,
                    "path": file_path,
                    "layers": len(model.layers),
                    "input_shape": str(model.input_shape)
                }
            })
        except Exception as e:
            return jsonify({
                "error": f"Error loading model: {str(e)}"
            }), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/models/load', methods=['POST'])
def load_models():
    """Load pre-trained models for specified symbols"""
    data = request.get_json()
    
    if not data or 'symbols' not in data:
        return jsonify({"error": "No symbols provided"}), 400
    
    symbols = data['symbols']
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    results = {}
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            # Look for model files in UPLOAD_FOLDER
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_final_model.keras")
            model_path_alt = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_model.keras")
            scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_scaler.pkl")
            
            print(f"Looking for model at: {model_path}")
            print(f"Alternative path: {model_path_alt}")
            print(f"Looking for scaler at: {scaler_path}")
            
            # Check if standard model exists
            if os.path.exists(model_path):
                print(f"Found model at {model_path}")
                model_file_to_use = model_path
            # Check for alternative naming (from upload)
            elif os.path.exists(model_path_alt):
                print(f"Found model at alternative path {model_path_alt}")
                model_file_to_use = model_path_alt
            else:
                # Check current directory as fallback
                current_dir_model = f"{symbol}_final_model.keras"
                current_dir_model_alt = f"{symbol}_model.keras"
                
                print(f"Checking current directory: {current_dir_model}")
                print(f"Checking current directory alternative: {current_dir_model_alt}")
                
                if os.path.exists(current_dir_model):
                    print(f"Found model in current directory: {current_dir_model}")
                    model_file_to_use = current_dir_model
                elif os.path.exists(current_dir_model_alt):
                    print(f"Found model in current directory: {current_dir_model_alt}")
                    model_file_to_use = current_dir_model_alt
                else:
                    print(f"No model file found for {symbol}")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No pre-trained model found for {symbol}"
                    }
                    continue
            
            # Load the model
            print(f"Attempting to load model from {model_file_to_use}")
            try:
                model = tf.keras.models.load_model(
                    model_file_to_use, 
                    custom_objects={'negative_log_likelihood': negative_log_likelihood}
                )
                loaded_models[symbol] = model
                print(f"Successfully loaded model for {symbol}")
                
                # Try to load scaler if available
                if os.path.exists(scaler_path):
                    print(f"Loading scaler from {scaler_path}")
                    try:
                        with open(scaler_path, 'rb') as f:
                            scaler_data = pickle.load(f)
                            loaded_scalers[symbol] = scaler_data
                            print(f"Successfully loaded scaler for {symbol}")
                    except Exception as scaler_error:
                        print(f"Error loading scaler: {scaler_error}")
                else:
                    current_dir_scaler = f"{symbol}_scaler.pkl"
                    if os.path.exists(current_dir_scaler):
                        print(f"Loading scaler from current directory: {current_dir_scaler}")
                        try:
                            with open(current_dir_scaler, 'rb') as f:
                                scaler_data = pickle.load(f)
                                loaded_scalers[symbol] = scaler_data
                                print(f"Successfully loaded scaler for {symbol}")
                        except Exception as scaler_error:
                            print(f"Error loading scaler: {scaler_error}")
                    else:
                        print(f"No scaler found for {symbol}, continuing without it")
                
                # Try to load feature importance data if available
                feature_importance_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_feature_importance.pkl")
                if os.path.exists(feature_importance_path):
                    try:
                        with open(feature_importance_path, 'rb') as f:
                            feature_importance = pickle.load(f)
                            feature_importance_data[symbol] = feature_importance
                            print(f"Successfully loaded feature importance data for {symbol}")
                    except Exception as fi_error:
                        print(f"Error loading feature importance data: {fi_error}")
                
                results[symbol] = {
                    "status": "success",
                    "message": f"Model for {symbol} loaded from {model_file_to_use}"
                }
            except Exception as model_error:
                print(f"Error loading model: {str(model_error)}")
                results[symbol] = {
                    "status": "error",
                    "message": f"Error loading model: {str(model_error)}"
                }
        except Exception as e:
            print(f"Unexpected error for {symbol}: {str(e)}")
            results[symbol] = {
                "status": "error",
                "message": str(e)
            }
    
    # Try to load market sentiment data if available
    try:
        market_sentiment_path = os.path.join(app.config['UPLOAD_FOLDER'], 'market_sentiment_data.csv')
        if os.path.exists(market_sentiment_path):
            global market_sentiment_data
            market_sentiment_data = pd.read_csv(market_sentiment_path)
            print("Successfully loaded market sentiment data")
    except Exception as sentiment_error:
        print(f"Error loading market sentiment data: {sentiment_error}")
    
    return jsonify({
        "results": results,
        "loaded_models": list(loaded_models.keys())
    })

@app.route('/predict/<symbol>', methods=['GET'])
def predict_symbol(symbol):
    """Get prediction for a specific symbol"""
    symbol = symbol.upper()
    
    if symbol not in loaded_models:
        return jsonify({
            "error": f"No model loaded for {symbol}",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    try:
        # Download latest stock data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=120)  # Get the last 120 days for prediction
        
        # Get stock data
        stock_data = download_or_load_stock_data([symbol], start_date, end_date)
        
        # Get sentiment data
        sentiment_data = fetch_news_sentiment_for_multiple_stocks([symbol], start_date, end_date)
        
        # Add macroeconomic features
        enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
        
        # Calculate enhanced technical indicators
        enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
        
        # Preprocess data
        processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
        
        # Create sequences if not already available
        if symbol not in loaded_sequences:
            sequences, scalers = create_features_target_multi_stock(processed_data, n_steps=60)
            loaded_sequences[symbol] = sequences.get(symbol, {})
            loaded_scalers[symbol] = scalers.get(symbol, {})
        
        # Generate prediction
        predictions = predict_today_for_multiple_stocks(
            {symbol: loaded_models[symbol]}, 
            processed_data, 
            loaded_scalers, 
            loaded_sequences
        )
        
        if symbol not in predictions:
            return jsonify({
                "error": f"Failed to generate prediction for {symbol}"
            }), 500
        
        # Format the prediction results
        prediction = predictions[symbol]
        formatted_prediction = {
            "symbol": symbol,
            "latest_date": prediction["latest_date"].strftime("%Y-%m-%d"),
            "prediction_date": prediction["prediction_date"].strftime("%Y-%m-%d"),
            "latest_price": float(prediction["latest_price"]),
            "predicted_price": float(prediction["predicted_price"]),
            "lower_bound": float(prediction["lower_bound"]),
            "upper_bound": float(prediction["upper_bound"]),
            "uncertainty": float(prediction["uncertainty"]),
            "percent_change": float(prediction["pct_change"])
        }
        
        return jsonify({
            "prediction": formatted_prediction,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error generating prediction: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Get predictions for multiple symbols"""
    data = request.get_json()
    
    if not data or 'symbols' not in data:
        return jsonify({"error": "No symbols provided"}), 400
    
    symbols = data['symbols']
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    # Filter to only symbols with loaded models
    valid_symbols = [symbol.upper() for symbol in symbols if symbol.upper() in loaded_models]
    
    if not valid_symbols:
        return jsonify({
            "error": "None of the provided symbols have loaded models",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    try:
        # Download latest stock data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=120)
        
        # Get stock data
        stock_data = download_or_load_stock_data(valid_symbols, start_date, end_date)
        
        # Get sentiment data
        sentiment_data = fetch_news_sentiment_for_multiple_stocks(valid_symbols, start_date, end_date)
        
        # Add macroeconomic features
        enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
        
        # Calculate enhanced technical indicators for each stock
        for symbol in valid_symbols:
            enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
        
        # Preprocess data
        processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
        
        # Create or update sequences
        for symbol in valid_symbols:
            if symbol not in loaded_sequences:
                symbol_data = {k: v for k, v in processed_data.items() if k == symbol}
                sequences, scalers = create_features_target_multi_stock(symbol_data, n_steps=60)
                loaded_sequences[symbol] = sequences.get(symbol, {})
                loaded_scalers[symbol] = scalers.get(symbol, {})
        
        # Filter models to only valid symbols
        models_subset = {symbol: loaded_models[symbol] for symbol in valid_symbols}
        
        # Generate predictions
        predictions = predict_today_for_multiple_stocks(
            models_subset, 
            processed_data, 
            loaded_scalers, 
            loaded_sequences
        )
        
        # Format the predictions
        formatted_predictions = {}
        for symbol, prediction in predictions.items():
            formatted_predictions[symbol] = {
                "symbol": symbol,
                "latest_date": prediction["latest_date"].strftime("%Y-%m-%d"),
                "prediction_date": prediction["prediction_date"].strftime("%Y-%m-%d"),
                "latest_price": float(prediction["latest_price"]),
                "predicted_price": float(prediction["predicted_price"]),
                "lower_bound": float(prediction["lower_bound"]),
                "upper_bound": float(prediction["upper_bound"]),
                "uncertainty": float(prediction["uncertainty"]),
                "percent_change": float(prediction["pct_change"])
            }
        
        return jsonify({
            "predictions": formatted_predictions,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error generating batch predictions: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/predict/timeframes/<symbol>', methods=['GET'])
def predict_timeframes(symbol):
    """Get multi-timeframe predictions for a symbol"""
    symbol = symbol.upper()
    
    if symbol not in loaded_models:
        return jsonify({
            "error": f"No model loaded for {symbol}",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    # Get timeframes from query parameters or use defaults
    timeframes_param = request.args.get('timeframes', '1,3,7,30')
    try:
        timeframes = [int(t) for t in timeframes_param.split(',')]
    except ValueError:
        timeframes = [1, 3, 7, 30]  # Default timeframes
    
    try:
        # Download latest stock data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=120)
        
        # Get stock data
        stock_data = download_or_load_stock_data([symbol], start_date, end_date)
        
        # Get sentiment data
        sentiment_data = fetch_news_sentiment_for_multiple_stocks([symbol], start_date, end_date)
        
        # Add macroeconomic features
        enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
        
        # Calculate enhanced technical indicators
        enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
        
        # Preprocess data
        processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
        
        # Create sequences if not already available
        if symbol not in loaded_sequences:
            sequences, scalers = create_features_target_multi_stock(processed_data, n_steps=60)
            loaded_sequences[symbol] = sequences.get(symbol, {})
            loaded_scalers[symbol] = scalers.get(symbol, {})
        
        # Generate multi-timeframe predictions
        models_subset = {symbol: loaded_models[symbol]}
        multi_timeframe_predictions = predict_multiple_timeframes(
            models_subset,
            processed_data,
            loaded_scalers,
            loaded_sequences,
            n_steps=60,
            timeframes=timeframes
        )
        
        if symbol not in multi_timeframe_predictions:
            return jsonify({
                "error": f"Failed to generate multi-timeframe predictions for {symbol}"
            }), 500
        
        # Format the prediction results
        prediction_data = multi_timeframe_predictions[symbol]
        latest_price = float(prediction_data['latest_price'])
        
        formatted_predictions = {
            "symbol": symbol,
            "latest_price": latest_price,
            "latest_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "timeframes": {}
        }
        
        for timeframe, data in prediction_data['timeframes'].items():
            formatted_predictions["timeframes"][timeframe] = {
                "predicted_price": float(data['predicted_price']),
                "pct_change": float(data['pct_change']),
                "lower_bound": float(data.get('lower_bound', data['predicted_price'] * 0.95)),
                "upper_bound": float(data.get('upper_bound', data['predicted_price'] * 1.05)),
                "prediction_date": (datetime.datetime.now() + datetime.timedelta(days=int(timeframe))).strftime("%Y-%m-%d")
            }
        
        return jsonify({
            "multi_timeframe_prediction": formatted_predictions,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error generating multi-timeframe predictions: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/predict/timeframes/batch', methods=['POST'])
def predict_timeframes_batch():
    """Get multi-timeframe predictions for multiple symbols"""
    data = request.get_json()
    
    if not data or 'symbols' not in data:
        return jsonify({"error": "No symbols provided"}), 400
    
    symbols = data['symbols']
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    # Get timeframes from the request or use defaults
    timeframes = data.get('timeframes', [1, 3, 7, 30])
    
    # Filter to only symbols with loaded models
    valid_symbols = [symbol.upper() for symbol in symbols if symbol.upper() in loaded_models]
    
    if not valid_symbols:
        return jsonify({
            "error": "None of the provided symbols have loaded models",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    try:
        # Download latest stock data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=120)
        
        # Get stock data
        stock_data = download_or_load_stock_data(valid_symbols, start_date, end_date)
        
        # Get sentiment data
        sentiment_data = fetch_news_sentiment_for_multiple_stocks(valid_symbols, start_date, end_date)
        
        # Add macroeconomic features
        enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
        
        # Calculate enhanced technical indicators for each stock
        for symbol in valid_symbols:
            enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
        
        # Preprocess data
        processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
        
        # Create or update sequences
        for symbol in valid_symbols:
            if symbol not in loaded_sequences:
                symbol_data = {k: v for k, v in processed_data.items() if k == symbol}
                sequences, scalers = create_features_target_multi_stock(symbol_data, n_steps=60)
                loaded_sequences[symbol] = sequences.get(symbol, {})
                loaded_scalers[symbol] = scalers.get(symbol, {})
        
        # Filter models to only valid symbols
        models_subset = {symbol: loaded_models[symbol] for symbol in valid_symbols}
        
        # Generate multi-timeframe predictions
        multi_timeframe_predictions = predict_multiple_timeframes(
            models_subset,
            processed_data,
            loaded_scalers,
            loaded_sequences,
            n_steps=60,
            timeframes=timeframes
        )
        
        # Format the predictions
        formatted_predictions = {}
        for symbol, prediction_data in multi_timeframe_predictions.items():
            latest_price = float(prediction_data['latest_price'])
            
            formatted_predictions[symbol] = {
                "symbol": symbol,
                "latest_price": latest_price,
                "latest_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "timeframes": {}
            }
            
            for timeframe, data in prediction_data['timeframes'].items():
                formatted_predictions[symbol]["timeframes"][timeframe] = {
                    "predicted_price": float(data['predicted_price']),
                    "pct_change": float(data['pct_change']),
                    "lower_bound": float(data.get('lower_bound', data['predicted_price'] * 0.95)),
                    "upper_bound": float(data.get('upper_bound', data['predicted_price'] * 1.05)),
                    "prediction_date": (datetime.datetime.now() + datetime.timedelta(days=int(timeframe))).strftime("%Y-%m-%d")
                }
        
        return jsonify({
            "multi_timeframe_predictions": formatted_predictions,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error generating batch multi-timeframe predictions: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/history/<symbol>', methods=['GET'])
def get_history(symbol):
    """Get historical data and analysis for a symbol"""
    symbol = symbol.upper()
    
    # Get query parameters
    days = request.args.get('days', default=30, type=int)
    include_technicals = request.args.get('technicals', default='true').lower() == 'true'
    include_enhanced = request.args.get('enhanced', default='false').lower() == 'true'
    
    try:
        # Download stock data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get stock data
        stock_data = download_or_load_stock_data([symbol], start_date, end_date)
        
        if symbol not in stock_data:
            return jsonify({
                "error": f"No historical data found for {symbol}"
            }), 404
        
        # Process data for API response
        df = stock_data[symbol]
        
        # Calculate technical indicators
        if include_technicals:
            if include_enhanced:
                df = calculate_enhanced_technical_indicators(df)
            else:
                df = calculate_technical_indicators(df)
        
        # Convert to JSON serializable format
        history_data = df.tail(days).to_dict(orient='records')
        
        # Format dates to string for JSON serialization
        for record in history_data:
            record['Date'] = record['Date'].strftime("%Y-%m-%d")
            # Convert numpy values to native Python types for JSON serialization
            for key, value in record.items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32, np.float16)):
                    record[key] = float(value)
        
        # Calculate basic statistics
        latest_price = float(df['Close'].iloc[-1])
        price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-days])
        percent_change = (price_change / df['Close'].iloc[-days]) * 100 if days > 0 else 0
        
        stats = {
            "symbol": symbol,
            "days": days,
            "latest_price": latest_price,
            "price_change": price_change,
            "percent_change": percent_change,
            "high": float(df['High'].max()),
            "low": float(df['Low'].min()),
            "average_volume": float(df['Volume'].mean()),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        return jsonify({
            "symbol": symbol,
            "statistics": stats,
            "history": history_data
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error retrieving history for {symbol}: {str(e)}"
        }), 500

@app.route('/sentiment', methods=['GET'])
def get_market_sentiment():
    """Get market sentiment data"""
    try:
        if market_sentiment_data is None:
            # If not already loaded, try to generate the market sentiment data
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)  # Get the last 30 days
            
            # Get symbols with loaded models
            symbols = list(loaded_models.keys())
            if not symbols:
                symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']  # Default symbols
            
            # Get stock data and sentiment data
            stock_data = download_or_load_stock_data(symbols, start_date, end_date)
            sentiment_data = fetch_news_sentiment_for_multiple_stocks(symbols, start_date, end_date)
            
            # Calculate market sentiment
            sentiment = calculate_market_sentiment(stock_data, sentiment_data)
            
            # Format the data
            sentiment_records = sentiment.to_dict(orient='records')
            for record in sentiment_records:
                if 'Date' in record:
                    record['Date'] = record['Date'].strftime("%Y-%m-%d") if isinstance(record['Date'], datetime.datetime) else str(record['Date'])
            
            return jsonify({
                "market_sentiment": sentiment_records,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # Use the already loaded market sentiment data
            sentiment_records = market_sentiment_data.to_dict(orient='records')
            for record in sentiment_records:
                if 'Date' in record:
                    record['Date'] = str(record['Date'])
                    
            return jsonify({
                "market_sentiment": sentiment_records,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    except Exception as e:
        return jsonify({
            "error": f"Error retrieving market sentiment data: {str(e)}"
        }), 500

@app.route('/feature_importance/<symbol>', methods=['GET'])
def get_feature_importance(symbol):
    """Get feature importance data for a symbol"""
    symbol = symbol.upper()
    
    if symbol not in loaded_models:
        return jsonify({
            "error": f"No model loaded for {symbol}",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    try:
        # Check if feature importance data is already available
        if symbol in feature_importance_data:
            importance = feature_importance_data[symbol]
        else:
            # Download stock data for feature importance calculation
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=120)
            
            # Get stock data
            stock_data = download_or_load_stock_data([symbol], start_date, end_date)
            
            # Get sentiment data
            sentiment_data = fetch_news_sentiment_for_multiple_stocks([symbol], start_date, end_date)
            
            # Add macroeconomic features
            enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
            
            # Calculate enhanced technical indicators
            enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
            
            # Preprocess data
            processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
            
            # Calculate feature importance
            importance = calculate_feature_importance(
                loaded_models[symbol],
                processed_data[symbol],
                n_steps=60
            )
            
            # Save feature importance data
            feature_importance_data[symbol] = importance
            
            # Save to disk for future use
            feature_importance_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_feature_importance.pkl")
            with open(feature_importance_path, 'wb') as f:
                pickle.dump(importance, f)
        
        # Format for response
        formatted_importance = []
        for feature, score in importance.items():
            formatted_importance.append({
                "feature": feature,
                "importance": float(score)  # Convert numpy float to Python float
            })
        
        # Sort by importance score
        formatted_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return jsonify({
            "symbol": symbol,
            "feature_importance": formatted_importance,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error calculating feature importance: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/simulation', methods=['POST'])
def run_simulation():
    """Run trading simulation based on predictions"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    symbols = data.get('symbols', [])
    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400
    
    if not isinstance(symbols, list):
        symbols = [symbols]
        
    # Filter to only symbols with loaded models
    valid_symbols = [symbol.upper() for symbol in symbols if symbol.upper() in loaded_models]
    
    if not valid_symbols:
        return jsonify({
            "error": "None of the provided symbols have loaded models",
            "loaded_models": list(loaded_models.keys())
        }), 404
    
    # Get simulation parameters
    initial_capital = data.get('initial_capital', 100000)
    days = data.get('days', 30)
    strategy = data.get('strategy', 'bayesian')  # bayesian, mean_reversion, momentum
    
    try:
        # Download historical data for simulation
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days*2)  # Get extra data for technical indicators
        
        # Get stock data
        stock_data = download_or_load_stock_data(valid_symbols, start_date, end_date)
        
        # Get sentiment data
        sentiment_data = fetch_news_sentiment_for_multiple_stocks(valid_symbols, start_date, end_date)
        
        # Add macroeconomic features
        enhanced_stock_data = add_macroeconomic_features(stock_data, start_date, end_date)
        
        # Calculate enhanced technical indicators for each stock
        for symbol in valid_symbols:
            enhanced_stock_data[symbol] = calculate_enhanced_technical_indicators(enhanced_stock_data[symbol])
        
        # Preprocess data
        processed_data = preprocess_multiple_stocks(enhanced_stock_data, sentiment_data)
        
        # Create or update sequences
        for symbol in valid_symbols:
            if symbol not in loaded_sequences:
                symbol_data = {k: v for k, v in processed_data.items() if k == symbol}
                sequences, scalers = create_features_target_multi_stock(symbol_data, n_steps=60)
                loaded_sequences[symbol] = sequences.get(symbol, {})
                loaded_scalers[symbol] = scalers.get(symbol, {})
        
        # Filter models to only valid symbols
        models_subset = {symbol: loaded_models[symbol] for symbol in valid_symbols}
        
        # Run simulation
        simulation_results = implement_mock_trading_simulation(
            models_subset,
            processed_data,
            loaded_scalers,
            loaded_sequences,
            initial_capital=initial_capital,
            days_to_simulate=days,
            strategy=strategy
        )
        
        # Format the simulation results
        formatted_results = {
            "initial_capital": float(simulation_results["initial_capital"]),
            "final_capital": float(simulation_results["final_capital"]),
            "profit_loss": float(simulation_results["profit_loss"]),
            "profit_loss_percent": float(simulation_results["profit_loss_percent"]),
            "trade_count": int(simulation_results["trade_count"]),
            "profitable_trades": int(simulation_results["profitable_trades"]),
            "losing_trades": int(simulation_results["losing_trades"]),
            "win_rate": float(simulation_results["win_rate"]),
            "strategy": strategy,
            "days_simulated": days,
            "symbols": valid_symbols
        }
        
        # Format the daily portfolio values
        portfolio_values = []
        for date, value in simulation_results["portfolio_values"].items():
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": float(value)
            })
        
        # Format the trade history
        trade_history = []
        for trade in simulation_results["trade_history"]:
            trade_history.append({
                "symbol": trade["symbol"],
                "type": trade["type"],
                "date": trade["date"].strftime("%Y-%m-%d"),
                "price": float(trade["price"]),
                "shares": float(trade["shares"]),
                "total": float(trade["total"]),
                "profit_loss": float(trade.get("profit_loss", 0))
            })
        
        formatted_results["portfolio_values"] = portfolio_values
        formatted_results["trade_history"] = trade_history
        
        return jsonify({
            "simulation_results": formatted_results,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Error running simulation: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train model for specified symbols"""
    data = request.get_json()
    
    if not data or 'symbols' not in data:
        return jsonify({"error": "No symbols provided"}), 400
    
    symbols = data['symbols']
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    symbols = [symbol.upper() for symbol in symbols]
    
    # Get training parameters
    days = data.get('days', 1000)
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 32)
    validation_split = data.get('validation_split', 0.2)
    
    results = {}
    
    for symbol in symbols:
        try:
            # Download historical data for training
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Create a StockDataManager instance for this symbol
            data_manager = StockDataManager(symbol, start_date, end_date, use_cache=True)
            
            # Load and preprocess data
            data_manager.load_data()
            data_manager.add_technical_indicators()
            data_manager.add_macroeconomic_data()
            data_manager.add_sentiment_analysis()
            data_manager.preprocess_data()
            
            # Create and train model
            data_manager.create_sequences()
            data_manager.setup_bayesian_model()
            
            # Train the model
            history = data_manager.train_model(
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            # Save the model and scaler
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_final_model.keras")
            scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_scaler.pkl")
            
            data_manager.save_model(model_path)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(data_manager.scaler, f)
            
            # Load the model
            loaded_models[symbol] = data_manager.model
            loaded_scalers[symbol] = data_manager.scaler
            loaded_sequences[symbol] = data_manager.sequences
            
            # Calculate feature importance
            feature_importance = data_manager.calculate_feature_importance()
            feature_importance_data[symbol] = feature_importance
            
            # Save feature importance
            feature_importance_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_feature_importance.pkl")
            with open(feature_importance_path, 'wb') as f:
                pickle.dump(feature_importance, f)
            
            # Calculate training metrics
            train_metrics = {
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "model_saved_at": model_path
            }
            
            results[symbol] = {
                "status": "success",
                "message": f"Model for {symbol} trained successfully",
                "metrics": train_metrics
            }
            
        except Exception as e:
            results[symbol] = {
                "status": "error",
                "message": f"Error training model for {symbol}: {str(e)}"
            }
    
    return jsonify({
        "training_results": results,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    # Load market sentiment data if available
    try:
        market_sentiment_path = os.path.join(app.config['UPLOAD_FOLDER'], 'market_sentiment_data.csv')
        if os.path.exists(market_sentiment_path):
            market_sentiment_data = pd.read_csv(market_sentiment_path)
            print("Successfully loaded market sentiment data")
    except Exception as sentiment_error:
        print(f"Error loading market sentiment data: {sentiment_error}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)