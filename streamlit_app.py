import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go

# Heading of 
st.title("Dynamic Implied Volatility Visualization")

# Black-Scholes formula for call option pricing
def black_scholes_call(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_value

# Function to estimate implied volatility using Brent's method
def calculate_iv(market_price, S, K, T, r, q=0):
    if T <= 0 or market_price <= 0:
        return np.nan

    def obj_func(sigma):
        return black_scholes_call(S, K, T, r, sigma, q) - market_price

    try:
        iv = brentq(obj_func, 1e-6, 5)
    except (ValueError, RuntimeError):
        iv = np.nan

    return iv

# Sidebar input for model parameters
st.sidebar.subheader('Configure Model Parameters')

# Adding hover tooltips for parameters
risk_rate = st.sidebar.number_input(
    'Risk-Free Rate (%)',
    value=1.5,
    step=0.1,
    help='The rate of return on a risk-free investment. Important for determining the present value of future cash flows.'
)
div_yield = st.sidebar.number_input(
    'Dividend Yield (%)',
    value=1.0,
    step=0.1,
    help='The dividend yield of the stock. Affects the option pricing by reducing the future cash flows of the underlying asset.'
)

# User inputs for stock symbol and strikes
st.sidebar.subheader('Input Options')
stock_symbol = st.sidebar.text_input('Stock Ticker', 'AAPL').upper()

min_strike_perc = st.sidebar.slider(
    'Minimum Strike (% of Spot)', 
    50, 
    200, 
    80,
    help='Defines the lower bound for the strike prices considered. Affects the range of options analyzed.'
)
max_strike_perc = st.sidebar.slider(
    'Maximum Strike (% of Spot)', 
    50, 
    200, 
    120,
    help='Defines the upper bound for the strike prices considered. Important for understanding the volatility surface.'
)

if min_strike_perc >= max_strike_perc:
    st.sidebar.error('Minimum strike percentage must be less than the maximum.')
    st.stop()

# Fetch stock data
stock = yf.Ticker(stock_symbol)
today = pd.Timestamp('today').normalize()

try:
    expiration_dates = stock.options
except Exception as e:
    st.error(f'Error fetching options data: {e}')
    st.stop()

valid_dates = [pd.Timestamp(date) for date in expiration_dates if pd.Timestamp(date) > today + timedelta(days=7)]

if not valid_dates:
    st.error(f'No valid option expiration dates for {stock_symbol}.')
else:
    option_chains = []
    
    for exp_date in valid_dates:
        try:
            option_chain = stock.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = option_chain.calls
        except Exception as e:
            st.warning(f'Error retrieving data for expiration {exp_date.date()}: {e}')
            continue
        
        valid_calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        
        for idx, row in valid_calls.iterrows():
            mid_price = (row['bid'] + row['ask']) / 2
            option_chains.append({
                'expiration': exp_date,
                'strike': row['strike'],
                'mid': mid_price
            })
    
    if not option_chains:
        st.error('No valid options data after filtering.')
    else:
        df = pd.DataFrame(option_chains)

        # Spot price retrieval
        try:
            spot_data = stock.history(period='5d')
            if spot_data.empty:
                st.error(f"Couldn't retrieve spot price for {stock_symbol}.")
                st.stop()
            else:
                spot_price = spot_data['Close'].iloc[-1]
        except Exception as e:
            st.error(f'Error fetching spot price: {e}')
            st.stop()

        df['days_to_exp'] = (df['expiration'] - today).dt.days
        df['time_to_exp'] = df['days_to_exp'] / 365

        # Strike price filtering
        df = df[
            (df['strike'] >= spot_price * (min_strike_perc / 100)) &
            (df['strike'] <= spot_price * (max_strike_perc / 100))
        ].reset_index(drop=True)

        # Calculate implied volatility
        df['implied_vol'] = df.apply(
            lambda row: calculate_iv(
                market_price=row['mid'], 
                S=spot_price, 
                K=row['strike'], 
                T=row['time_to_exp'], 
                r=risk_rate, 
                q=div_yield
            ), axis=1
        )

        df.dropna(subset=['implied_vol'], inplace=True)
        df['implied_vol'] *= 100  # Convert to percentage

        # Visualization
        X, Y, Z = df['time_to_exp'], df['strike'], df['implied_vol']
        grid_X, grid_Y = np.linspace(X.min(), X.max(), 50), np.linspace(Y.min(), Y.max(), 50)
        grid_X, grid_Y = np.meshgrid(grid_X, grid_Y)
        grid_Z = griddata((X, Y), Z, (grid_X, grid_Y), method='linear')

        fig = go.Figure(data=[go.Surface(z=grid_Z, x=grid_X, y=grid_Y, colorscale='Plasma')])
        fig.update_layout(
            title=f'Implied Volatility Surface for {stock_symbol}',
            scene=dict(
                xaxis_title='Time to Expiration (Years)',
                yaxis_title='Strike Price ($)',
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=800,
            height=800
        )

        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("By Akshat Vij | [GitHub](https://github.com/AkshatVij)")
