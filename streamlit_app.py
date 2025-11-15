import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="URO Executive Dashboard - RUL Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🤖"
)

# --- Simulate Data (The Hybrid LSTM Output) ---
@st.cache_data
def load_simulated_data():
    data = { # <--- INDENTED
        'Asset ID': ['T-459', 'T-211', 'P-803', 'T-601', 'B-112'],
        'RUL (Days)': [47.3, 75.1, 120.5, 310.8, 550.2],
        'Confidence': [0.93, 0.88, 0.75, 0.99, 0.95],
        'Cost Avoided ($)': [3350000, 1800000, 500000, 0, 0]
    }
    df = pd.DataFrame(data) # <--- INDENTED
    df['Risk Status'] = np.where(df['RUL (Days)'] < 60, 'CRITICAL', # <--- INDENTED
                              np.where(df['RUL (Days)'] < 120, 'HIGH', 'LOW')) # <--- INDENTED
    return df.sort_values(by='RUL (Days)', ascending=True) # <--- INDENTED

df_rul = load_simulated_data()

# --- 2. Title and Summary Metrics ---
st.title("⚡ Universal Resource Optimizer (URO) Dashboard")
st.markdown("### Real-Time Asset Performance Monitoring")

total_cost_avoided = df_rul['Cost Avoided ($)'].sum()
critical_assets = df_rul[df_rul['Risk Status'] == 'CRITICAL'].shape[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Cost Avoided YTD (Pilot)", # <--- INDENTED
              value=f"${total_cost_avoided:,.0f}", # <--- INDENTED
              delta="Target: $5,000,000") # <--- INDENTED

with col2:
    st.metric(label="Assets in CRITICAL Status (<60 Days RUL)", # <--- INDENTED
              value=critical_assets, # <--- INDENTED
              delta="1 Asset Moved to Low-Risk Last Week", # <--- INDENTED
              delta_color="normal") # <--- INDENTED

with col3:
    st.metric(label="Average Portfolio RUL", # <--- INDENTED
              value=f"{df_rul['RUL (Days)'].mean():.1f} Days") # <--- INDENTED

st.markdown("---")

# --- 3. Asset Detail Table ---
st.subheader("Asset RUL & Risk Overview")
st.dataframe(df_rul, use_container_width=True, hide_index=True)

# --- 4. RUL Trend Chart (Simulation of the LSTM Output) ---
st.subheader("Critical Asset Trend: T-459 (Transformer)")

# Simulate the RUL trend data for the chart
rul_data = pd.DataFrame({
    'Day': np.arange(1, 101),
    'RUL (Days)': np.exp(-0.02 * np.arange(1, 101)) * 365,
})
# Simulate the sharp drop due to the anomaly detected by the LSTM
rul_data.loc[90:, 'RUL (Days)'] = rul_data.loc[90:, 'RUL (Days)'] * 0.5 + 47.3

st.line_chart(rul_data.set_index('Day'))
st.markdown("""
<div style='background-color:#ffebeb; padding:10px; border-radius:5px;'>
⚠️ **Model Insight (T-459):** The sharp RUL drop after Day 90 reflects the detection of an accelerated degradation anomaly
by the Hybrid LSTM, validating the 47-day alert.
</div>
""", unsafe_allow_html=True)
