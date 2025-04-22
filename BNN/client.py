import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import sys
import pickle
from typing import List, Dict, Union, Optional

class StockPredictionClient:
    """Client for connecting to Stock Prediction API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        """
        Initialize the API client
        
        Args:
            base_url: URL of the Stock Prediction API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_connection(self) -> bool:
        """
        Check if connection to API is working
        
        Returns:
            bool: True if connection successful
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available stock symbols
        
        Returns:
            List of available symbols
        """
        response = self.session.get(f"{self.base_url}/symbols")
        data = response.json()
        return data.get("symbols", [])
    
    def get_loaded_models(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
            Dict with model information
        """
        response = self.session.get(f"{self.base_url}/models")
        return response.json()
    
    def load_models(self, symbols: List[str]) -> Dict:
        """
        Load models for specified symbols
        
        Args:
            symbols: List of stock symbols to load models for
            
        Returns:
            Dict with load results
        """
        payload = {"symbols": symbols}
        response = self.session.post(
            f"{self.base_url}/models/load",
            json=payload
        )
        return response.json()
    
    def upload_model(self, symbol: str, model_file_path: str) -> Dict:
        """
        Upload a model file for a specific symbol
        
        Args:
            symbol: Stock symbol
            model_file_path: Path to model file
            
        Returns:
            Dict with upload result
        """
        with open(model_file_path, 'rb') as file:
            files = {'model': file}
            data = {'symbol': symbol}
            response = self.session.post(
                f"{self.base_url}/models/upload",
                files=files,
                data=data
            )
        return response.json()
    
    def get_prediction(self, symbol: str) -> Dict:
        """
        Get prediction for a specific symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with prediction results
        """
        response = self.session.get(f"{self.base_url}/predict/{symbol}")
        return response.json()
    
    def get_batch_predictions(self, symbols: List[str]) -> Dict:
        """
        Get predictions for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict with batch prediction results
        """
        payload = {"symbols": symbols}
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        return response.json()
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Dict:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
            
        Returns:
            Dict with historical data
        """
        response = self.session.get(
            f"{self.base_url}/history/{symbol}",
            params={"days": days}
        )
        return response.json()
    
    def request_training(self, symbols: List[str], epochs: int = 150, days: int = 2190) -> Dict:
        """
        Request model training for specific symbols
        
        Args:
            symbols: List of stock symbols
            epochs: Number of training epochs
            days: Days of historical data to use (default ~6 years)
            
        Returns:
            Dict with training request status
        """
        payload = {
            "symbols": symbols,
            "epochs": epochs,
            "days": days
        }
        response = self.session.post(
            f"{self.base_url}/train",
            json=payload
        )
        return response.json()
    
    def get_multi_timeframe_predictions(self, symbol: str, timeframes: List[int] = [1, 3, 7, 30]) -> Dict:
        """
        Get predictions for multiple timeframes
        
        Args:
            symbol: Stock symbol
            timeframes: List of day counts for predictions
            
        Returns:
            Dict with multi-timeframe predictions
        """
        payload = {
            "symbol": symbol,
            "timeframes": timeframes
        }
        response = self.session.post(
            f"{self.base_url}/predict/timeframes",
            json=payload
        )
        return response.json()
    
    def get_market_sentiment(self) -> Dict:
        """
        Get current market sentiment data
        
        Returns:
            Dict with market sentiment information
        """
        response = self.session.get(f"{self.base_url}/sentiment")
        return response.json()
    
    def get_feature_importance(self, symbol: str) -> Dict:
        """
        Get feature importance for a specific symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with feature importance data
        """
        response = self.session.get(f"{self.base_url}/importance/{symbol}")
        return response.json()
    
    def run_trading_simulation(self, symbols: List[str], days: int = 30, 
                           strategy: str = "default") -> Dict:
        """
        Run a mock trading simulation
        
        Args:
            symbols: List of stock symbols
            days: Number of days to simulate
            strategy: Trading strategy to use
            
        Returns:
            Dict with simulation results
        """
        payload = {
            "symbols": symbols,
            "days": days,
            "strategy": strategy
        }
        response = self.session.post(
            f"{self.base_url}/simulation",
            json=payload
        )
        return response.json()
    
    def visualize_prediction(self, symbol: str, save: bool = False) -> None:
        """
        Visualize prediction for a symbol
        
        Args:
            symbol: Stock symbol
            save: Whether to save visualization to file
        """
        # Get prediction
        prediction_data = self.get_prediction(symbol)
        if "error" in prediction_data:
            print(f"Error getting prediction: {prediction_data['error']}")
            return
        
        # Get historical data (30 days)
        history_data = self.get_historical_data(symbol, days=30)
        if "error" in history_data:
            print(f"Error getting history: {history_data['error']}")
            return
        
        # Create DataFrame from historical data
        history_df = pd.DataFrame(history_data["history"])
        history_df["Date"] = pd.to_datetime(history_df["Date"])
        
        # Extract prediction
        prediction = prediction_data["prediction"]
        pred_date = datetime.datetime.strptime(prediction["prediction_date"], "%Y-%m-%d")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["Date"], history_df["Close"], label="Historical Price", color="blue")
        
        # Add prediction point and uncertainty
        plt.scatter([pred_date], [prediction["predicted_price"]], color="red", s=50, zorder=5, label="Prediction")
        plt.vlines(pred_date, prediction["lower_bound"], prediction["upper_bound"], color="red", linestyle="-", lw=2, alpha=0.5)
        
        # Add last known price
        latest_date = datetime.datetime.strptime(prediction["latest_date"], "%Y-%m-%d")
        plt.scatter([latest_date], [prediction["latest_price"]], color="green", s=50, zorder=5, label="Latest Price")
        
        # Format the plot
        plt.title(f"{symbol} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add prediction info
        info_text = (
            f"Prediction: ${prediction['predicted_price']:.2f}\n"
            f"Change: {prediction['percent_change']:.2f}%\n"
            f"Range: ${prediction['lower_bound']:.2f} to ${prediction['upper_bound']:.2f}"
        )
        plt.annotate(info_text, xy=(0.02, 0.95), xycoords="axes fraction", 
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        if save:
            plt.savefig(f"{symbol}_prediction.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def visualize_multi_timeframe_predictions(self, symbol: str, save: bool = False) -> None:
        """
        Visualize multi-timeframe predictions for a symbol
        
        Args:
            symbol: Stock symbol
            save: Whether to save visualization to file
        """
        # Get multi-timeframe predictions
        timeframes = [1, 3, 7, 30]  # Default timeframes
        prediction_data = self.get_multi_timeframe_predictions(symbol, timeframes)
        
        if "error" in prediction_data:
            print(f"Error getting multi-timeframe predictions: {prediction_data['error']}")
            return
        
        # Extract predictions
        predictions = prediction_data.get("predictions", {})
        
        if not predictions:
            print("No prediction data available")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        # Get historical data for context
        history_data = self.get_historical_data(symbol, days=60)
        if "error" not in history_data:
            history_df = pd.DataFrame(history_data["history"])
            history_df["Date"] = pd.to_datetime(history_df["Date"])
            plt.plot(history_df["Date"], history_df["Close"], label="Historical Price", color="blue")
        
        # Get latest price and date
        latest_price = predictions.get("latest_price", 0)
        latest_date = datetime.datetime.strptime(predictions.get("latest_date", datetime.datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
        
        # Plot predictions for each timeframe
        x_dates = []
        y_prices = []
        labels = []
        colors = ['red', 'orange', 'green', 'purple']
        
        for i, timeframe in enumerate(timeframes):
            if str(timeframe) in predictions.get("timeframes", {}):
                tf_pred = predictions["timeframes"][str(timeframe)]
                pred_date = latest_date + datetime.timedelta(days=timeframe)
                x_dates.append(pred_date)
                y_prices.append(tf_pred["predicted_price"])
                labels.append(f"{timeframe}-Day")
                
                # Plot uncertainty bounds
                if "lower_bound" in tf_pred and "upper_bound" in tf_pred:
                    plt.vlines(pred_date, tf_pred["lower_bound"], tf_pred["upper_bound"], 
                               color=colors[i % len(colors)], linestyle="-", lw=2, alpha=0.5)
        
        # Plot the predictions
        for i, (x, y, label) in enumerate(zip(x_dates, y_prices, labels)):
            plt.scatter([x], [y], color=colors[i % len(colors)], s=50, zorder=5, label=label)
        
        # Add latest price point
        plt.scatter([latest_date], [latest_price], color="black", s=60, zorder=5, label="Latest Price")
        
        # Format the plot
        plt.title(f"{symbol} Multi-Timeframe Stock Price Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save:
            plt.savefig(f"{symbol}_multi_timeframe_prediction.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def visualize_feature_importance(self, symbol: str, save: bool = False) -> None:
        """
        Visualize feature importance for a symbol
        
        Args:
            symbol: Stock symbol
            save: Whether to save visualization to file
        """
        # Get feature importance data
        feature_data = self.get_feature_importance(symbol)
        
        if "error" in feature_data:
            print(f"Error getting feature importance: {feature_data['error']}")
            return
        
        # Extract data
        features = feature_data.get("features", [])
        importances = feature_data.get("importances", [])
        
        if not features or not importances:
            print("No feature importance data available")
            return
        
        # Sort by importance
        sorted_indices = sorted(range(len(importances)), key=lambda i: importances[i])
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.barh(sorted_features[-15:], sorted_importances[-15:])  # Show top 15 features
        plt.xlabel('Importance')
        plt.title(f'Top Feature Importance for {symbol}')
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{symbol}_feature_importance.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def visualize_market_sentiment(self, save: bool = False) -> None:
        """
        Visualize market sentiment over time
        
        Args:
            save: Whether to save visualization to file
        """
        # Get market sentiment data
        sentiment_data = self.get_market_sentiment()
        
        if "error" in sentiment_data:
            print(f"Error getting market sentiment: {sentiment_data['error']}")
            return
        
        # Extract data
        history = sentiment_data.get("history", [])
        
        if not history:
            print("No market sentiment data available")
            return
        
        # Create DataFrame
        df = pd.DataFrame(history)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot sentiment score
        plt.subplot(2, 1, 1)
        plt.plot(df["Date"], df["Composite_Score"], color="blue", marker="o", markersize=4)
        plt.title("Market Sentiment Over Time")
        plt.ylabel("Sentiment Score (0-100)")
        plt.grid(True, alpha=0.3)
        
        # Plot sentiment categories as heatmap
        if "Sentiment_Category" in df.columns:
            plt.subplot(2, 1, 2)
            categories = df["Sentiment_Category"].unique()
            category_map = {cat: i for i, cat in enumerate(sorted(categories))}
            df["Category_Num"] = df["Sentiment_Category"].map(category_map)
            
            plt.scatter(df["Date"], df["Category_Num"], c=df["Category_Num"], cmap="RdYlGn", s=100, alpha=0.7)
            plt.yticks(range(len(category_map)), sorted(categories))
            plt.title("Sentiment Categories")
            plt.ylabel("Category")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig("market_sentiment.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def visualize_trading_simulation(self, simulation_results: Dict, save: bool = False) -> None:
        """
        Visualize trading simulation results
        
        Args:
            simulation_results: Dictionary with simulation results
            save: Whether to save visualization to file
        """
        if not simulation_results or "trades" not in simulation_results:
            print("No simulation results available")
            return
        
        # Extract data
        trades = simulation_results["trades"]
        portfolio_value = simulation_results.get("portfolio_value", [])
        
        # Create DataFrame for trades
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty and "date" in trades_df.columns:
            trades_df["date"] = pd.to_datetime(trades_df["date"])
        
        # Create DataFrame for portfolio value
        portfolio_df = None
        if portfolio_value:
            portfolio_df = pd.DataFrame(portfolio_value)
            if "date" in portfolio_df.columns:
                portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        if portfolio_df is not None and not portfolio_df.empty:
            plt.plot(portfolio_df["date"], portfolio_df["value"], color="blue", lw=2)
            plt.title("Portfolio Value Over Time")
            plt.ylabel("Value ($)")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No portfolio value data available", 
                     horizontalalignment='center', verticalalignment='center')
        
        # Plot trades
        plt.subplot(2, 1, 2)
        if not trades_df.empty and "date" in trades_df.columns and "type" in trades_df.columns:
            buy_trades = trades_df[trades_df["type"] == "buy"]
            sell_trades = trades_df[trades_df["type"] == "sell"]
            
            if not buy_trades.empty:
                plt.scatter(buy_trades["date"], buy_trades["price"], color="green", label="Buy", marker="^", s=100)
            
            if not sell_trades.empty:
                plt.scatter(sell_trades["date"], sell_trades["price"], color="red", label="Sell", marker="v", s=100)
            
            plt.title("Trading Activity")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No trade data available", 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig("trading_simulation.png", dpi=300, bbox_inches="tight")
        
        plt.show()


def main():
    """Main function demonstrating use of the client"""
    # Create client
    client = StockPredictionClient()
    
    # Check connection
    if not client.check_connection():
        print("Failed to connect to API. Make sure the server is running.")
        sys.exit(1)
    
    print("===== Stock Price Prediction with Bayesian Neural Networks =====")
    
    # Get available symbols
    symbols = client.get_available_symbols()
    if not symbols:
        print("No stock symbols available. Using default symbols.")
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    
    print(f"Analyzing stocks: {', '.join(symbols)}")
    
    # Load models for all symbols
    print("Loading prediction models...")
    load_result = client.load_models(symbols)
    loaded_symbols = load_result.get('loaded_models', [])
    print(f"Loaded models: {', '.join(loaded_symbols) if loaded_symbols else 'None'}")
    
    if not loaded_symbols:
        print("No models loaded. Please make sure models are available on the server.")
        sys.exit(1)
    
    # Get batch predictions for loaded models
    print("\nGenerating predictions...")
    predictions = client.get_batch_predictions(loaded_symbols)
    
    if "predictions" in predictions:
        # Print prediction results
        print("\n===== SUMMARY OF PREDICTIONS =====")
        for symbol, pred in predictions["predictions"].items():
            print(f"\n{symbol} Predictions:")
            print(f"Current Price: ${pred['latest_price']:.2f}")
            print(f"Next-Day Forecast: ${pred['predicted_price']:.2f} " +
                  ("↑" if pred['percent_change'] > 0 else "↓") +
                  f" ({pred['percent_change']:.2f}%)")
            print(f"Prediction Range: ${pred['lower_bound']:.2f} to ${pred['upper_bound']:.2f}")
    
    # Get market sentiment
    print("\nFetching market sentiment...")
    try:
        sentiment = client.get_market_sentiment()
        if "error" not in sentiment:
            latest = sentiment.get("latest", {})
            print(f"Current Market Sentiment: {latest.get('Sentiment_Category', 'N/A')} " +
                  f"({latest.get('Composite_Score', 0):.1f}/100)")
        else:
            print(f"Error fetching market sentiment: {sentiment['error']}")
    except Exception as e:
        print(f"Error fetching market sentiment: {e}")
    
    # Display visualization for a single stock
    if loaded_symbols:
        print(f"\nGenerating visualization for {loaded_symbols[0]}...")
        try:
            # Regular prediction visualization
            client.visualize_prediction(loaded_symbols[0], save=True)
            
            # Try multi-timeframe visualization
            try:
                client.visualize_multi_timeframe_predictions(loaded_symbols[0], save=True)
            except Exception as e:
                print(f"Multi-timeframe visualization not available: {e}")
            
            # Try feature importance visualization
            try:
                client.visualize_feature_importance(loaded_symbols[0], save=True)
            except Exception as e:
                print(f"Feature importance visualization not available: {e}")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    # Run a trading simulation
    print("\nRunning trading simulation...")
    try:
        simulation = client.run_trading_simulation(loaded_symbols[:3])
        if "error" not in simulation:
            # Display simulation results
            portfolio_value = simulation.get("final_value", 0)
            initial_value = simulation.get("initial_value", 0)
            profit = portfolio_value - initial_value
            profit_percent = (profit / initial_value) * 100 if initial_value > 0 else 0
            
            print(f"Trading Simulation Results:")
            print(f"Initial Portfolio: ${initial_value:.2f}")
            print(f"Final Portfolio: ${portfolio_value:.2f}")
            print(f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)")
            
            # Visualize simulation results
            client.visualize_trading_simulation(simulation, save=True)
        else:
            print(f"Error running simulation: {simulation['error']}")
    except Exception as e:
        print(f"Trading simulation not available: {e}")
    
    print("\nAnalysis complete! Use the predictions and visualizations to inform your trading decisions.")


if __name__ == "__main__":
    main()