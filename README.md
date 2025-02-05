# Cloud Utilization Prediction using LSTMs

## Overview
This project predicts **CPU, Memory, Network, and Disk utilization** in a cloud environment using **Long Short-Term Memory (LSTM) neural networks**. By leveraging AI-powered time-series forecasting, this system provides insights into cloud resource usage trends, helping cloud providers optimize performance, reduce costs, and prevent downtime.

## Features
- **Multi-Resource Prediction**: Forecasts CPU, Memory, Network, and Disk usage.
- **Anomaly Detection**: Identifies unexpected spikes in resource utilization.
- **Hyperparameter Tuning**: Optimized LSTM model for improved accuracy.
- **Real-Time API Deployment**: Uses FastAPI to serve predictions.
- **Enhanced Visualizations**: Provides clear insights with time-series graphs.
- **Model Saving & Deployment**: Enables real-world integration and inference.

## Dataset
The dataset consists of simulated cloud workload metrics, including:
- **CPU Usage** (Percentage)
- **Memory Usage** (Percentage)
- **Network Utilization** (MB/s)
- **Disk Utilization** (MB/s)

In real-world scenarios, data can be sourced from **AWS CloudWatch, Google Cloud Metrics, or Azure Monitor**.

## Installation & Setup
### 1. Clone the Repository
```sh
git clone https://github.com/your-username/Cloud-Resource-Predictor.git
cd Cloud-Resource-Predictor
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```sh
jupyter notebook
```
Open `Cloud_Resource_Utilization_Prediction_With_Graphs.ipynb` and execute the cells.

## Model Architecture
The LSTM model consists of:
- **LSTM Layers**: Captures time-series dependencies.
- **Dropout Layers**: Prevents overfitting.
- **Dense Layers**: Outputs multi-resource predictions.

## Usage
### Training the Model
Run the training script in the notebook:
```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

### Evaluating the Model
```python
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
```

### Saving and Loading the Model
```python
model.save('cloud_utilization_prediction.h5')
```
To load the model for inference:
```python
from tensorflow.keras.models import load_model
model = load_model('cloud_utilization_prediction.h5')
```

## Visualization
To compare actual vs. predicted CPU usage:
```python
plt.plot(y_test[:, 0], label='Actual CPU Usage', color='blue')
plt.plot(predictions[:, 0], label='Predicted CPU Usage', linestyle='dashed', color='orange')
plt.xlabel('Time')
plt.ylabel('CPU Usage')
plt.title('Actual vs. Predicted CPU Utilization')
plt.legend()
plt.show()
```

## Future Enhancements
- **Integrate with Real Cloud Data**: AWS CloudWatch, Google Cloud Metrics.
- **Deploy Model as an API**: Provide real-time cloud usage forecasts.
- **Extend to Additional Metrics**: Include latency, IOPS, and other factors.
- **Improve Anomaly Detection**: Implement advanced outlier detection methods.

## License
This project is open-source under the **MIT License**.

## Contributors
- **Your Name** – [GitHub Profile](https://github.com/your-username)

## Acknowledgments
- TensorFlow/Keras for deep learning support.
- Open-source datasets for simulation.
