
    class StockPredictionApp {
      constructor() {
        this.apiUrl = 'http://localhost:5000';
        this.connected = false;
        this.selectedSymbol = null;
        this.availableSymbols = [];
        this.loadedModels = [];
        this.predictionChart = null;
        this.historyChart = null;
        this.indicatorsChart = null;
        
        // Initialize the application
        this.init();
      }
      
      init() {
        // Set up event listeners
        document.getElementById('connect-btn').addEventListener('click', () => this.connectToApi());
        document.getElementById('refresh-symbols-btn').addEventListener('click', () => this.fetchSymbols());
        document.getElementById('load-models-btn').addEventListener('click', () => this.loadModels());
        document.getElementById('get-prediction-btn').addEventListener('click', () => this.getPrediction());
        document.getElementById('get-history-btn').addEventListener('click', () => this.getHistory());
        document.getElementById('update-history-btn').addEventListener('click', () => this.getHistory());
        
        // Set up tab navigation
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
          button.addEventListener('click', (event) => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button and corresponding pane
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
          });
        });
        
        // Get saved API URL from localStorage if available
        const savedApiUrl = localStorage.getItem('stockApiUrl');
        if (savedApiUrl) {
          document.getElementById('api-url').value = savedApiUrl;
        }
        
        // Try to connect automatically
        this.connectToApi();
      }
      
      showNotification(message, type = 'info') {
        const connectionStatus = document.getElementById('connection-status');
        connectionStatus.innerHTML = `<div class="notification ${type}">${message}</div>`;
      }
      
      async connectToApi() {
        const apiUrlInput = document.getElementById('api-url');
        this.apiUrl = apiUrlInput.value.trim();
        
        if (!this.apiUrl) {
          this.showNotification('Please enter a valid API URL', 'error');
          return;
        }
        
        this.showNotification('Connecting to API...', 'info');
        
        try {
          const response = await fetch(`${this.apiUrl}/`);
          
          if (response.ok) {
            const data = await response.json();
            this.connected = true;
            this.showNotification(`Connected to API: ${data.message}`, 'success');
            
            // Save the API URL to localStorage
            localStorage.setItem('stockApiUrl', this.apiUrl);
            
            // Enable buttons
            document.getElementById('refresh-symbols-btn').disabled = false;
            
            // Fetch symbols
            this.fetchSymbols();
          } else {
            this.connected = false;
            this.showNotification('Failed to connect to API', 'error');
          }
        } catch (error) {
          this.connected = false;
          this.showNotification(`Connection error: ${error.message}`, 'error');
        }
      }
      
      async fetchSymbols() {
        if (!this.connected) {
          this.showNotification('Not connected to API', 'error');
          return;
        }
        
        const symbolsList = document.getElementById('symbols-list');
        symbolsList.innerHTML = '<div class="loading"></div>';
        
        try {
          const response = await fetch(`${this.apiUrl}/symbols`);
          
          if (response.ok) {
            const data = await response.json();
            this.availableSymbols = data.symbols;
            this.loadedModels = data.loaded_models;
            
            // Populate symbols list
            symbolsList.innerHTML = '';
            this.availableSymbols.forEach(symbol => {
              const item = document.createElement('div');
              item.className = 'symbol-item';
              if (this.loadedModels.includes(symbol)) {
                item.innerHTML = `${symbol} <small>(Model loaded)</small>`;
              } else {
                item.textContent = symbol;
              }
              
              item.addEventListener('click', () => this.selectSymbol(symbol));
              symbolsList.appendChild(item);
            });
            
            // Enable load models button
            document.getElementById('load-models-btn').disabled = false;
          } else {
            symbolsList.innerHTML = '<p>Failed to fetch symbols</p>';
          }
        } catch (error) {
          symbolsList.innerHTML = `<p>Error: ${error.message}</p>`;
        }
      }
      
      selectSymbol(symbol) {
        this.selectedSymbol = symbol;
        
        // Update selected symbol info
        const selectedStockInfo = document.getElementById('selected-stock-info');
        selectedStockInfo.innerHTML = `
          <h4>${symbol}</h4>
          <p>${this.loadedModels.includes(symbol) ? 'Model loaded' : 'Model not loaded'}</p>
        `;
        
        // Update active class on symbol items
        const symbolItems = document.querySelectorAll('.symbol-item');
        symbolItems.forEach(item => {
          if (item.textContent.startsWith(symbol)) {
            item.classList.add('active');
          } else {
            item.classList.remove('active');
          }
        });
        
        // Enable buttons
        document.getElementById('get-prediction-btn').disabled = !this.loadedModels.includes(symbol);
        document.getElementById('get-history-btn').disabled = false;
        document.getElementById('update-history-btn').disabled = false;
      }
      
      async loadModels() {
        if (!this.connected || this.availableSymbols.length === 0) {
          this.showNotification('Cannot load models - not connected or no symbols available', 'error');
          return;
        }
        
        const symbolsList = document.getElementById('symbols-list');
        symbolsList.innerHTML = '<div class="loading"></div>';
        
        try {
          const response = await fetch(`${this.apiUrl}/models/load`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              symbols: this.availableSymbols
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            this.loadedModels = data.loaded_models;
            this.showNotification(`Loaded ${this.loadedModels.length} models`, 'success');
            
            // Refresh symbols to update the loaded status
            this.fetchSymbols();
            
            // Enable prediction button if the selected symbol has a model
            if (this.selectedSymbol && this.loadedModels.includes(this.selectedSymbol)) {
              document.getElementById('get-prediction-btn').disabled = false;
            }
          } else {
            this.showNotification('Failed to load models', 'error');
          }
        } catch (error) {
          this.showNotification(`Error loading models: ${error.message}`, 'error');
        }
      }
      
      async getPrediction() {
        if (!this.connected || !this.selectedSymbol) {
          this.showNotification('Cannot get prediction - not connected or no symbol selected', 'error');
          return;
        }
        
        // Check if model is loaded for the selected symbol
        if (!this.loadedModels.includes(this.selectedSymbol)) {
          this.showNotification(`No model loaded for ${this.selectedSymbol}`, 'error');
          return;
        }
        
        // Show loading state
        const predictionContent = document.getElementById('prediction-content');
        predictionContent.innerHTML = '<div class="loading"></div>';
        document.getElementById('prediction-details').innerHTML = '';
        
        try {
          const response = await fetch(`${this.apiUrl}/predict/${this.selectedSymbol}`);
          
          if (response.ok) {
            const data = await response.json();
            const prediction = data.prediction;
            
            // Update prediction content
            predictionContent.innerHTML = `
              <h3>${this.selectedSymbol} Prediction</h3>
              <p>Latest data: ${prediction.latest_date} | Prediction for: ${prediction.prediction_date}</p>
            `;
            
            // Update prediction details
            const details = document.getElementById('prediction-details');
            details.innerHTML = `
              <div class="prediction-item">
                <div>Latest Price</div>
                <div class="prediction-value">$${prediction.latest_price.toFixed(2)}</div>
              </div>
              <div class="prediction-item">
                <div>Predicted Price</div>
                <div class="prediction-value">$${prediction.predicted_price.toFixed(2)}</div>
              </div>
              <div class="prediction-item">
                <div>Change</div>
                <div class="prediction-value ${prediction.percent_change >= 0 ? 'positive-change' : 'negative-change'}">
                  ${prediction.percent_change >= 0 ? '+' : ''}${prediction.percent_change.toFixed(2)}%
                </div>
              </div>
              <div class="prediction-item">
                <div>Uncertainty Range</div>
                <div class="prediction-value">$${prediction.lower_bound.toFixed(2)} - $${prediction.upper_bound.toFixed(2)}</div>
              </div>
            `;
            
            // Create or update chart
            this.createPredictionChart(prediction);
            
            // Also fetch history data to enhance the chart
            this.getHistory(true);
          } else {
            const errorData = await response.json();
            predictionContent.innerHTML = `<p>Failed to get prediction: ${errorData.error || 'Unknown error'}</p>`;
          }
        } catch (error) {
          predictionContent.innerHTML = `<p>Error: ${error.message}</p>`;
        }
      }
      
      createPredictionChart(prediction) {
        const ctx = document.getElementById('prediction-chart').getContext('2d');
        
        // If chart already exists, destroy it
        if (this.predictionChart) {
          this.predictionChart.destroy();
        }
        
        // Parse dates
        const latestDate = new Date(prediction.latest_date);
        const predictionDate = new Date(prediction.prediction_date);
        
        this.predictionChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: [prediction.latest_date, prediction.prediction_date],
            datasets: [
              {
                label: 'Actual Price',
                data: [
                  { x: prediction.latest_date, y: prediction.latest_price }
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                pointRadius: 6
              },
              {
                label: 'Predicted Price',
                data: [
                  { x: prediction.prediction_date, y: prediction.predicted_price }
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                pointRadius: 6
              },
              {
                label: 'Uncertainty Range',
                data: [
                  { x: prediction.prediction_date, y: prediction.lower_bound },
                  { x: prediction.prediction_date, y: prediction.upper_bound }
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 0.5)',
                borderWidth: 0,
                pointRadius: 0,
                fill: false
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                type: 'time',
                time: {
                  unit: 'day',
                  displayFormats: {
                    day: 'MMM dd'
                  }
                },
                title: {
                  display: true,
                  text: 'Date'
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Price ($)'
                }
              }
            },
            plugins: {
              title: {
                display: true,
                text: `${this.selectedSymbol} Stock Price Prediction`
              },
              tooltip: {
                mode: 'index',
                intersect: false
              },
              legend: {
                display: true,
                position: 'top'
              }
            }
          }
        });
      }
      
      async getHistory(skipChartUpdate = false) {
        if (!this.connected || !this.selectedSymbol) {
          this.showNotification('Cannot get history - not connected or no symbol selected', 'error');
          return;
        }
        
        // Get number of days from select element
        const days = document.getElementById('history-days').value;
        
        // Show loading state
        const historyData = document.getElementById('history-data');
        historyData.innerHTML = '<div class="loading"></div>';
        
        try {
          const response = await fetch(`${this.apiUrl}/history/${this.selectedSymbol}?days=${days}`);
          
          if (response.ok) {
            const data = await response.json();
            
            // Update history data with a table
            historyData.innerHTML = `
              <h3>${this.selectedSymbol} Historical Data</h3>
              <p>Period: ${data.statistics.start_date} to ${data.statistics.end_date}</p>
              <div class="prediction-details">
                <div class="prediction-item">
                  <div>Latest Price</div>
                  <div class="prediction-value">$${data.statistics.latest_price.toFixed(2)}</div>
                </div>
                <div class="prediction-item">
                  <div>Change (${days} days)</div>
                  <div class="prediction-value ${data.statistics.percent_change >= 0 ? 'positive-change' : 'negative-change'}">
                    ${data.statistics.percent_change >= 0 ? '+' : ''}${data.statistics.percent_change.toFixed(2)}%
                  </div>
                </div>
                <div class="prediction-item">
                  <div>High</div>
                  <div class="prediction-value">$${data.statistics.high.toFixed(2)}</div>
                </div>
                <div class="prediction-item">
                  <div>Low</div>
                  <div class="prediction-value">$${data.statistics.low.toFixed(2)}</div>
                </div>
              </div>
              
              <table class="stock-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Open</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>Close</th>
                    <th>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  ${data.history.slice(0, 10).map(day => `
                    <tr>
                      <td>${day.Date}</td>
                      <td>$${day.Open.toFixed(2)}</td>
                      <td>$${day.High.toFixed(2)}</td>
                      <td>$${day.Low.toFixed(2)}</td>
                      <td>$${day.Close.toFixed(2)}</td>
                      <td>${this.formatNumber(day.Volume)}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
              <p style="text-align:center; margin-top:10px; font-size:12px;">Showing 10 most recent days of ${data.history.length} total days</p>
            `;
            
            // Create history chart
            if (!skipChartUpdate) {
              this.createHistoryChart(data.history);
            } else {
              // If this was called from prediction, update the prediction chart with historical data
              this.updatePredictionChartWithHistory(data.history);
            }
            
            // Create indicators chart
            this.createIndicatorsChart(data.history);
          } else {
            const errorData = await response.json();
            historyData.innerHTML = `<p>Failed to get history: ${errorData.error || 'Unknown error'}</p>`;
          }
        } catch (error) {
          historyData.innerHTML = `<p>Error: ${error.message}</p>`;
        }
      }
      
      createHistoryChart(historyData) {
        const ctx = document.getElementById('history-chart').getContext('2d');
        
        // If chart already exists, destroy it
        if (this.historyChart) {
          this.historyChart.destroy();
        }
        
        // Prepare data
        const dates = historyData.map(day => day.Date);
        const closePrices = historyData.map(day => day.Close);
        const highPrices = historyData.map(day => day.High);
        const lowPrices = historyData.map(day => day.Low);
        
        this.historyChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: dates,
            datasets: [
              {
                label: 'Close Price',
                data: closePrices,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                fill: true
              },
              {
                label: 'High',
                data: highPrices,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 1,
                pointRadius: 0,
                fill: false
              },
              {
                label: 'Low',
                data: lowPrices,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderWidth: 1,
                pointRadius: 0,
                fill: false
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Date'
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Price ($)'
                }
              }
            },
            plugins: {
              title: {
                display: true,
                text: `${this.selectedSymbol} Historical Price Data`
              },
              tooltip: {
                mode: 'index',
                intersect: false
              }
            }
          }
        });
      }
      
      updatePredictionChartWithHistory(historyData) {
        if (!this.predictionChart) return;
        
        // Add historical data to the prediction chart
        const historicalData = historyData.map(day => {
          return { x: day.Date, y: day.Close };
        });
        
        // Update the 'Actual Price' dataset
        this.predictionChart.data.datasets[0].data = historicalData;
        this.predictionChart.update();
      }
      
      createIndicatorsChart(historyData) {
        const ctx = document.getElementById('indicators-chart').getContext('2d');
        
        // If chart already exists, destroy it
        if (this.indicatorsChart) {
          this.indicatorsChart.destroy();
        }
        
        // Calculate moving averages
        const ma20 = this.calculateMovingAverage(historyData, 20);
        const ma50 = this.calculateMovingAverage(historyData, 50);
        
        // Prepare data
        const dates = historyData.map(day => day.Date);
        const closePrices = historyData.map(day => day.Close);
        
        // Update indicators content
        const indicatorsContent = document.getElementById('indicators-content');
        const latestPrice = historyData[0].Close;
        const ma20Latest = ma20.length > 0 ? ma20[ma20.length - 1] : null;
        const ma50Latest = ma50.length > 0 ? ma50[ma50.length - 1] : null;
        
        indicatorsContent.innerHTML = `
          <h3>${this.selectedSymbol} Technical Indicators</h3>
          <div class="prediction-details">
            <div class="prediction-item">
              <div>20-Day MA</div>
              <div class="prediction-value">$${ma20Latest ? ma20Latest.toFixed(2) : 'N/A'}</div>
              ${ma20Latest ? `<div class="${latestPrice > ma20Latest ? 'positive-change' : 'negative-change'}">
                ${latestPrice > ma20Latest ? 'Price above MA' : 'Price below MA'}
              </div>` : ''}
            </div>
            <div class="prediction-item">
              <div>50-Day MA</div>
              <div class="prediction-value">$${ma50Latest ? ma50Latest.toFixed(2) : 'N/A'}</div>
              ${ma50Latest ? `<div class="${latestPrice > ma50Latest ? 'positive-change' : 'negative-change'}">
                ${latestPrice > ma50Latest ? 'Price above MA' : 'Price below MA'}
              </div>` : ''}
            </div>
          </div>
        `;
        
        this.indicatorsChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: dates,
            datasets: [
              {
                label: 'Close Price',
                data: closePrices,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                fill: false
              },
              {
                label: '20-Day MA',
                data: ma20,
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
              },
              {
                label: '50-Day MA',
                data: ma50,
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Date'
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Price ($)'
                }
              }
            },
            plugins: {
              title: {
                display: true,
                text: `${this.selectedSymbol} Technical Indicators`
              },
              tooltip: {
                mode: 'index',
                intersect: false
              }
            }
          }
        });
      }
      
      calculateMovingAverage(data, period) {
        const result = [];
        const prices = data.map(day => day.Close);
        
        for (let i = period - 1; i < prices.length; i++) {
          const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          result.push(sum / period);
        }
        
        // Pad with nulls for proper alignment on chart
        const padding = Array(period - 1).fill(null);
        return [...padding, ...result];
      }
      
      formatNumber(num) {
        if (num >= 1000000) {
          return (num / 1000000).toFixed(2) + 'M';
        } else if (num >= 1000) {
          return (num / 1000).toFixed(2) + 'K';
        }
        return num.toString();
      }
    }
    
    // Initialize the app when document is loaded
    document.addEventListener('DOMContentLoaded', () => {
      const app = new StockPredictionApp();
    });
  