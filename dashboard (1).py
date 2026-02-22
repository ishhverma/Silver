
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.interpolate import interp1d
from scipy import stats

# Define HFT_COLORS (Copied from notebook's global state)
HFT_COLORS = {'bg': '#0E1117', 'card': '#1E1E1E', 'text': '#FFFFFF', 'accent1': '#00FF9D', 'accent2': '#FF6B6B', 'accent3': '#4ECDC4', 'accent4': '#FFD93D', 'grid': '#2E2E2E', 'positive': '#00FF9D', 'negative': '#FF6B6B', 'neutral': '#4ECDC4', 'volatility': '#FFD93D'}

# --- Helper Functions (to make dashboard self-contained) ---
@st.cache_data
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_svix(returns, window=30):
    rv = returns.rolling(window=window).std() * np.sqrt(252)
    # Simplified IV proxy for dashboard, usually based on options data
    # For this dashboard, we'll use a smoothed version of RV as IV proxy
    iv_proxy = rv.ewm(span=20).mean() # Use EWM for smoothing

    # Normalize similar to VIX logic, but based on historical RV average
    if not iv_proxy.empty and iv_proxy.first_valid_index() is not None:
        first_valid_iv_proxy = iv_proxy.loc[iv_proxy.first_valid_index()]
        if first_valid_iv_proxy != 0:
            svix = iv_proxy / first_valid_iv_proxy * 100
        else:
            svix = pd.Series(100.0, index=iv_proxy.index) # Default if first_valid_iv_proxy is zero
    else:
        svix = pd.Series(dtype=float) # Empty series if no valid iv_proxy

    return svix.replace([np.inf, -np.inf], np.nan) # Handle potential inf values

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    try:
        df = pd.read_csv('data/silver_updated.csv', index_col=0, parse_dates=True)
        # Ensure 'Close' column exists. If not, create from 'Close_SI=F'
        if 'Close' not in df.columns and 'Close_SI=F' in df.columns:
            df['Close'] = df['Close_SI=F']
        elif 'Close' not in df.columns:
            st.error("Error: 'Close' column not found in data. Please ensure data contains a 'Close' price.")
            return pd.DataFrame() # Return empty DataFrame on error

        # Recalculate essential columns for consistency and robustness
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_7'] = df['returns'].rolling(window=7).std() * np.sqrt(252)
        df['rv_14'] = df['returns'].rolling(window=14).std() * np.sqrt(252)
        df['rv_30'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
        df['rv_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['rv_90'] = df['returns'].rolling(window=90).std() * np.sqrt(252)
        df['rv_180'] = df['returns'].rolling(window=180).std() * np.sqrt(252)
        df['vol_of_vol'] = df['rv_30'].rolling(window=30).std()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        df['rsi_14'] = calculate_rsi(df['Close'], 14)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Generate SVIX
        df['svix'] = calculate_svix(df['returns'])

        # Ensure garch_vol_annualized exists for dashboard
        if 'garch_vol_annualized' not in df.columns:
            # Placeholder/fallback if GARCH not pre-calculated
            df['garch_vol_annualized'] = df['rv_30'].ewm(span=50).mean() * 1.05 # Add a slight premium
        
        # Volatility signal for demonstration (simplified from notebook)
        # Assuming higher IV proxy than RV means overvalued (sell), lower means undervalued (buy)
        df['vol_signal'] = 0
        if 'implied_vol_proxy' in df.columns:
             vol_spread = (df['implied_vol_proxy'] - df['rv_30']) / df['rv_30']
             df.loc[vol_spread < -0.1, 'vol_signal'] = 1   # Buy vol
             df.loc[vol_spread > 0.1, 'vol_signal'] = -1    # Sell vol
        elif 'garch_vol_annualized' in df.columns:
            vol_spread = (df['garch_vol_annualized'] - df['rv_30']) / df['rv_30']
            df.loc[vol_spread < -0.1, 'vol_signal'] = 1   # Buy vol
            df.loc[vol_spread > 0.1, 'vol_signal'] = -1    # Sell vol
        
        # Fill NA for vol_signal and others after calculation
        df['vol_signal'] = df['vol_signal'].ffill().fillna(0)


        # Add a placeholder for signal_strength, if not already present
        if 'vol_signal_strength' not in df.columns:
            if 'implied_vol_proxy' in df.columns:
                df['vol_signal_strength'] = abs((df['implied_vol_proxy'] - df['rv_30']) / df['rv_30'])
            elif 'garch_vol_annualized' in df.columns:
                df['vol_signal_strength'] = abs((df['garch_vol_annualized'] - df['rv_30']) / df['rv_30'])
            else:
                df['vol_signal_strength'] = 0.0 # Default if no vol data for strength

        return df.dropna(subset=['Close', 'returns', 'rv_30', 'svix', 'garch_vol_annualized', 'vol_signal'])
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure 'data/silver_updated.csv' exists and is correctly formatted.")
        return pd.DataFrame()

# --- Volatility Surface Logic (Adapted from notebook) ---
@st.cache_data
def generate_volatility_surface(df_current, current_sigma):
    maturities_days = np.array([7, 14, 30, 60, 90, 180, 270, 365])
    maturities = maturities_days / 365 # in years
    moneyness = np.linspace(0.7, 1.3, 20) # Strike/Spot ratio

    # Get current term structure using available realized vols
    term_structure_vols = []
    term_structure_mats = []

    if 'rv_7' in df_current.columns and df_current['rv_7'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_7'].iloc[-1] * 100)
        term_structure_mats.append(7/365)
    if 'rv_14' in df_current.columns and df_current['rv_14'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_14'].iloc[-1] * 100)
        term_structure_mats.append(14/365)
    if 'rv_30' in df_current.columns and df_current['rv_30'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_30'].iloc[-1] * 100)
        term_structure_mats.append(30/365)
    if 'rv_60' in df_current.columns and df_current['rv_60'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_60'].iloc[-1] * 100)
        term_structure_mats.append(60/365)
    if 'rv_90' in df_current.columns and df_current['rv_90'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_90'].iloc[-1] * 100)
        term_structure_mats.append(90/365)
    if 'rv_180' in df_current.columns and df_current['rv_180'].iloc[-1] is not None:
        term_structure_vols.append(df_current['rv_180'].iloc[-1] * 100)
        term_structure_mats.append(180/365)

    # Ensure enough points for interpolation, fallback to current_sigma if not
    if len(term_structure_mats) >= 2:
        # Sort by maturity
        sorted_indices = np.argsort(term_structure_mats)
        term_structure_mats = np.array(term_structure_mats)[sorted_indices]
        term_structure_vols = np.array(term_structure_vols)[sorted_indices]

        term_interp = interp1d(term_structure_mats, term_structure_vols,
                               kind='linear', fill_value='extrapolate', bounds_error=False)
    else:
        # Fallback to a flat term structure if not enough data points
        st.warning("Not enough data points for accurate volatility term structure. Using current volatility as base.")
        term_interp = lambda x: current_sigma * 100 # Use current sigma (annualized)

    vol_surface = np.zeros((len(maturities), len(moneyness)))

    for i, T in enumerate(maturities):
        # Use a reasonable cap for extrapolation to prevent extreme values
        base_vol = term_interp(min(T, term_structure_mats[-1] if term_structure_mats.size > 0 else 1.0)) / 100 # Convert back to decimal

        # Simple smile effect (can be more sophisticated)
        for j, m in enumerate(moneyness):
            # Smile effect: higher vol for OTM options, lower for ATM
            # A common approach uses a quadratic function centered at 1 (ATM)
            smile_factor = 1 + 0.5 * (m - 1)**2 # Adjust coefficient (0.5) for intensity
            vol_surface[i, j] = base_vol * smile_factor

    # Ensure volatility is non-negative
    vol_surface[vol_surface < 0] = 0.001
    
    return vol_surface * 100 # Return in percentage

# --- Dashboard Layout ---
st.title("🥈 Silver Volatility Trading System")
st.markdown("### Institutional-Grade Volatility Analytics")

# Refresh button to clear cache and reload data
if st.sidebar.button("Refresh Data", help="Click to reload data and re-run calculations"):
    st.cache_data.clear()
    st.rerun()

df = load_data()

if df.empty:
    st.stop()

# Get latest data point
latest_data = df.iloc[-1]
current_price = latest_data['Close']
current_vol_rv30 = latest_data['rv_30'] * 100
current_svix = latest_data['svix']
current_signal = latest_data['vol_signal']

# Calculate changes for metrics
prev_data = df.iloc[-2]
price_change = (current_price / prev_data['Close'] - 1) * 100
vol_change = (current_vol_rv30 / (prev_data['rv_30'] * 100) - 1) * 100

# Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Silver Price",
        f"${current_price:.2f}",
        f"{price_change:.2f}%",
        delta_color="normal",
        help="Latest closing price of Silver futures."
    )

with col2:
    st.metric(
        "30-Day Realized Volatility",
        f"{current_vol_rv30:.2f}%",
        f"{vol_change:.2f}%",
        delta_color="inverse", # Volatility increase is usually negative
        help="Annualized 30-day historical volatility."
    )

with col3:
    st.metric(
        "SVIX Index",
        f"{current_svix:.2f}",
        help="Silver Volatility Index (a measure of market's expectation of future volatility)."
    )

with col4:
    signal_text = "BUY VOLATILITY" if current_signal == 1 else "SELL VOLATILITY" if current_signal == -1 else "NEUTRAL"
    signal_color = HFT_COLORS['positive'] if current_signal == 1 else HFT_COLORS['negative'] if current_signal == -1 else HFT_COLORS['neutral']
    st.markdown(f"<div style='background-color:{HFT_COLORS['card']}; padding: 15px; border-radius: 5px;'>"
                f"<h4 style='color:{HFT_COLORS['text']}; margin-bottom: 5px;'>Volatility Signal</h4>"
                f"<p style='color:{signal_color}; font-size: 24px; font-weight: bold;'>{signal_text}</p>"
                f"</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Price and Volatility Chart ---
st.subheader("Price and Volatility Overview", divider='rainbow')
fig_price_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.6, 0.4])

# Price chart
fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                 name='Silver Price', line=dict(color=HFT_COLORS['accent1'], width=2),
                                 hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>Price:</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64<extra></extra>"),
                      row=1, col=1)
fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['ma_20'],
                                 name='20-day MA', line=dict(color=HFT_COLORS['accent3'], dash='dot'),
                                 hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>MA(20):</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64<extra></extra>"),
                      row=1, col=1)
fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['ma_50'],
                                 name='50-day MA', line=dict(color=HFT_COLORS['accent4'], dash='dash'),
                                 hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>MA(50):</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64<extra></extra>"),
                      row=1, col=1)

# Volatility chart
fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['rv_30']*100,
                                 name='Realized Vol (30d)', line=dict(color=HFT_COLORS['accent2'], width=2),
                                 hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>RV(30):</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64%<extra></extra>"),
                      row=2, col=1)
if 'garch_vol_annualized' in df.columns:
    fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['garch_vol_annualized']*100,
                                     name='GARCH Vol', line=dict(color=HFT_COLORS['volatility'], dash='dot'),
                                     hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>GARCH Vol:</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64%<extra></extra>"),
                          row=2, col=1)
fig_price_vol.add_trace(go.Scatter(x=df.index, y=df['svix'],
                                 name='SVIX Index', line=dict(color=HFT_COLORS['neutral'], dash='dash'),
                                 hovertemplate="<b>Date:</b> %[-0.37610309 -0.37097831 -0.36585353 -0.36072876 -0.35560398 -0.3504792
 -0.34535442 -0.34022965 -0.33510487 -0.32998009 -0.32485531 -0.31973054
 -0.31460576 -0.30948098 -0.30435621 -0.29923143 -0.29410665 -0.28898187
 -0.2838571  -0.27873232 -0.27360754 -0.26848277 -0.26335799 -0.25823321
 -0.25310843 -0.24798366 -0.24285888 -0.2377341  -0.23260932 -0.22748455
 -0.22235977 -0.21723499 -0.21211022 -0.20698544 -0.20186066 -0.19673588
 -0.19161111 -0.18648633 -0.18136155 -0.17623678 -0.171112   -0.16598722
 -0.16086244 -0.15573767 -0.15061289 -0.14548811 -0.14036333 -0.13523856
 -0.13011378 -0.124989   -0.11986423 -0.11473945 -0.10961467 -0.10448989
 -0.09936512 -0.09424034 -0.08911556 -0.08399079 -0.07886601 -0.07374123
 -0.06861645 -0.06349168 -0.0583669  -0.05324212 -0.04811735 -0.04299257
 -0.03786779 -0.03274301 -0.02761824 -0.02249346 -0.01736868 -0.0122439
 -0.00711913 -0.00199435  0.00313043  0.0082552   0.01337998  0.01850476
  0.02362954  0.02875431  0.03387909  0.03900387  0.04412864  0.04925342
  0.0543782   0.05950298  0.06462775  0.06975253  0.07487731  0.08000209
  0.08512686  0.09025164  0.09537642  0.10050119  0.10562597  0.11075075
  0.11587553  0.1210003   0.12612508  0.13124986]<br><b>SVIX:</b> %Date
2000-11-22    0.132671
2000-11-27    0.124357
2000-11-28    0.125475
2000-11-29    0.122452
2000-11-30    0.121691
                ...   
2025-12-31    1.441226
2026-01-02    1.452231
2026-01-05    1.443504
2026-01-06    1.434012
2026-01-07    1.438920
Name: target_vol, Length: 6280, dtype: float64<extra></extra>"),
                      row=2, col=1)

fig_price_vol.update_layout(
    height=600,
    title_text='Silver Price and Volatility Trends',
    template='plotly_dark',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
fig_price_vol.update_yaxes(title_text="Price ($")", row=1, col=1) # Corrected extra double quote
fig_price_vol.update_yaxes(title_text="Volatility (%)", row=2, col=1)

st.plotly_chart(fig_price_vol, use_container_width=True)

st.markdown("---")

# --- Volatility Surface ---
st.subheader("Silver Volatility Surface", divider='rainbow')

col_surf_1, col_surf_2 = st.columns([0.6, 0.4])

with col_surf_1:
    # Generate the actual volatility surface
    vol_surface_data = generate_volatility_surface(df, current_vol_rv30/100) # Pass decimal volatility

    maturities_days_surf = np.array([7, 14, 30, 60, 90, 180, 270, 365])
    moneyness_surf = np.linspace(0.7, 1.3, 20)

    fig_surface = go.Figure(data=[go.Surface(z=vol_surface_data, x=moneyness_surf, y=maturities_days_surf,
                                             colorscale='Viridis',
                                             colorbar=dict(title='Implied Volatility (%)'))])
    fig_surface.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Moneyness (Strike/Spot)',
            yaxis_title='Maturity (Days)',
            zaxis_title='Implied Volatility (%)',
            bgcolor=HFT_COLORS['bg'],
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig_surface, use_container_width=True)

with col_surf_2:
    st.markdown("This 3D plot visualizes the **Volatility Surface** for Silver options. It shows how implied volatility varies with both **Moneyness** (Strike Price relative to Spot Price) and **Time to Maturity**.")
    st.markdown("- **Moneyness (X-axis):** Options further out-of-the-money (lower moneyness for calls, higher for puts) typically have higher implied volatility (the 'smile' or 'skew').")
    st.markdown("- **Maturity (Y-axis):** Volatility changes across different maturities, reflecting the market's view on short-term vs. long-term risks (the 'term structure').")
    st.markdown("A rising surface indicates increasing implied volatility for those option characteristics.")

st.markdown("---")

# --- Volatility Smile for Selected Maturities ---
st.subheader("Volatility Smile", divider='rainbow')

smile_maturity_options = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}
selected_maturity_key = st.selectbox("Select Maturity to View Smile:", list(smile_maturity_options.keys()))
selected_maturity_days = smile_maturity_options[selected_maturity_key]

# Find the closest maturity in the generated surface
maturities_days_surf = np.array([7, 14, 30, 60, 90, 180, 270, 365])
closest_maturity_idx = np.argmin(np.abs(maturities_days_surf - selected_maturity_days))

fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(x=moneyness_surf, y=vol_surface_data[closest_maturity_idx, :],
                             mode='lines+markers', name=f'{maturities_days_surf[closest_maturity_idx]} Days Maturity',
                             line=dict(color=HFT_COLORS['accent1'], width=3),
                             marker=dict(size=8, color=HFT_COLORS['accent1'])))

fig_smile.add_vline(x=1.0, line_width=2, line_dash="dash", line_color=HFT_COLORS['text'], annotation_text="ATM", annotation_position="top right")

fig_smile.update_layout(
    title=f'Volatility Smile for {maturities_days_surf[closest_maturity_idx]} Days Maturity',
    xaxis_title='Moneyness (Strike/Spot)',
    yaxis_title='Implied Volatility (%)',
    template='plotly_dark',
    hovermode="x unified",
    height=450
)
st.plotly_chart(fig_smile, use_container_width=True)

st.markdown("---")

# --- Recent Data Table ---
st.subheader("Recent Data & Signals", divider='rainbow')
st.dataframe(df[['Close', 'returns', 'rv_30', 'svix', 'garch_vol_annualized', 'vol_signal', 'vol_signal_strength']].tail(10).style.highlight_max(axis=0, subset=['rv_30', 'svix', 'garch_vol_annualized', 'vol_signal_strength'], color=HFT_COLORS['accent1']))

st.markdown("---")

# --- Key Risk Metrics ---
st.subheader("Key Risk Metrics (Daily)", divider='rainbow')
col_risk1, col_risk2, col_risk3 = st.columns(3)

returns_series = df['returns'].dropna()
position_size = 100 # Assuming a unit position for percentage VaR

# VaR (Historical)
var_95_hist_pct = np.percentile(returns_series, 5) * 100
var_99_hist_pct = np.percentile(returns_series, 1) * 100

with col_risk1:
    st.metric("95% Historical VaR", f"{abs(var_95_hist_pct):.2f}%", help="Maximum expected loss (in percent) over one day with 95% confidence based on historical data.")
with col_risk2:
    st.metric("99% Historical VaR", f"{abs(var_99_hist_pct):.2f}%", help="Maximum expected loss (in percent) over one day with 99% confidence based on historical data.")
with col_risk3:
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
    st.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.3f}", help="Risk-adjusted return for the Silver futures. Assumes risk-free rate is zero for simplicity.")


st.markdown("---")
st.markdown(f"<div style='text-align: center; color: {HFT_COLORS['text']}; opacity: 0.7;'>© 2024 Silver Volatility Quant System</div>", unsafe_allow_html=True)
