# Stock Trend Predictor

## Overview
The **Stock Trend Predictor** project aims to forecast stock prices using historical data and machine learning techniques. This project primarily focuses on predicting the stock prices of Apple Inc. (AAPL) by leveraging Long Short-Term Memory (LSTM) neural networks. The goal is to provide accurate predictions that can assist investors in making informed decisions.

## Features
- `Data Collection:` Collected 20 years of historical stock data for Apple Inc. (AAPL) from Yahoo Finance.
- `Data Preprocessing:` Cleaned and prepared the data by calculating moving averages (100-day, 200-day, 250-day) and normalizing the dataset.
- `Model Development:` Developed an LSTM neural network to predict future stock prices based on the preprocessed data.
- `Model Evaluation:` Evaluated the model's performance using Root Mean Squared Error (RMSE) and compared predicted prices with actual prices.
- `Interactive Visualization:` Created a Streamlit web application for visualizing historical stock prices, moving averages, and model predictions.

## Project Structure
- `stock_price.ipynb:` Jupyter notebook containing the complete process of data collection, preprocessing, model development, and evaluation for stock price prediction.
- `stock.py:` Python script that includes the implementation of the LSTM model and related functions for stock price prediction.
- `Latest_stock_price_model.keras:` The saved Keras model file for the latest trained LSTM model used for predicting stock prices.
- `requirements.txt:` A list of Python packages and dependencies required to run the project.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Stock-Trend-Predictor.git
cd Stock-Trend-Predictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit Application:
```bash
streamlit run app.py
```
- Input your desired stock symbol, date range, and other parameters.
- The app will display historical data, moving averages, and predicted stock prices.

2. View the Tableau Notebook:

- [Link to Tableau Notebook](https://public.tableau.com/views/AAPLstockanalysis_17222744580350/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- The Tableau workbook provides additional visual insights into the stock trends and prediction analysis.

## Results
- The LSTM model successfully predicted stock prices with a reasonable level of accuracy, as indicated by the RMSE value.
- The Streamlit app and Tableau notebook provided clear, interactive visualizations that help in understanding stock trends and making data-driven decisions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please contact monalisaburma@gmail.com 

