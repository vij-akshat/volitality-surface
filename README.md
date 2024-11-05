# Dynamic Implied Volatility Visualization

This project provides a visual, interactive tool for analyzing implied volatility surfaces using real-time stock options data. Built with Streamlit and Plotly, this app allows users to configure options parameters and view a 3D volatility surface for better understanding the volatility landscape of a selected stock.

![Volatility Surface Preview]

---

## Features

- **Black-Scholes Call Option Pricing**: Implements the Black-Scholes model to calculate call option prices.
- **Implied Volatility Calculation**: Estimates implied volatility using market prices and Brent's optimization method.
- **Customizable Inputs**: Users can set the risk-free rate, dividend yield, stock ticker, and strike price range.
- **Dynamic 3D Visualization**: Renders the implied volatility surface as a function of time to expiration and strike price, powered by Plotly.

## Installation

To run the app, ensure you have the following dependencies:

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- pandas
- numpy
- scipy
- plotly

Install them with:
```bash
pip install streamlit yfinance pandas numpy scipy plotly
