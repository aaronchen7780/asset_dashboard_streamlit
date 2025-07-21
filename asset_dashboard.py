import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Portfolio Allocation Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_factor_data():
    """Load Fama-French factor data from CSV files"""
    try:
        # Load five factors data
        factor_data = pd.read_csv('exposure_datasets/factors_cleaned.csv', index_col=0)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        return factor_data
    except FileNotFoundError:
        # Fallback to mock data if files not found
        st.warning("Factor data files not found. Using mock data for demonstration.")
        dates = pd.date_range(start='2022-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        
        factor_data = pd.DataFrame({
            'date': dates,
            'Mkt-RF': np.random.normal(0.05, 1.2, len(dates)),
            'SMB': np.random.normal(0, 0.8, len(dates)),
            'HML': np.random.normal(0, 0.9, len(dates)),
            'RMW': np.random.normal(0, 0.7, len(dates)),
            'CMA': np.random.normal(0, 0.6, len(dates)),
            'RF': np.random.normal(0.02, 0.1, len(dates)),
            'MOM': np.random.normal(0, 1.1, len(dates))
        })
        
        return factor_data

@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date=None):
    """Fetch stock data for given tickers"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=730)
    
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=datetime.now(), progress=False)
            if not data.empty:
                stock_data[ticker] = data
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    return stock_data

@st.cache_data(ttl=1800)
def get_exchange_rate():
    """Get USD/EUR exchange rate"""
    return 0.86

def calculate_portfolio_metrics(stock_data, quantities):
    """Calculate portfolio metrics and positions"""
    positions_values = {}
    
    for ticker, data in stock_data.items():
        if ticker in quantities and not data.empty:
            # Ensure the last price is always a single float, not a Series
            last_price = float(data['Close'].iloc[-1])
            quantity = quantities.get(ticker, 0)
            positions_values[ticker] = last_price * quantity
    
    return positions_values

def get_asset_exposures(tickers, positions_values, factor_data, decay_rate=0.97):
    """Calculate factor exposures for assets"""
    stock_data = get_stock_data(tickers)
    exposures = []
    
    factor_df = factor_data.copy()
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df = factor_df.set_index('date')
    
    for ticker in tickers:
        if ticker not in stock_data or stock_data[ticker].empty:
            print(f"No data available for {ticker}")
            continue
            
        stock_df = stock_data[ticker].copy()
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            stock_df.index = pd.to_datetime(stock_df.index)
        
        daily_returns = stock_df['Close'].pct_change().dropna() * 100
        
        weekly_returns = daily_returns.resample('W-FRI').last()
        factor_weekly = factor_df.resample('W-FRI').last()
        
        if isinstance(weekly_returns, pd.Series):
            returns_df = weekly_returns.to_frame('return')
        else:
            # If it's already a DataFrame, just rename the column
            returns_df = weekly_returns.copy()
            returns_df.columns = ['return']
        
        merged = returns_df.join(factor_weekly, how='inner')
        merged = merged.dropna()
        
        merged['excess_return'] = merged['return'] - merged['RF']
        
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
            
        X = merged[factor_cols]
        y = merged['excess_return']
        
        days_from_end = (merged.index.max() - merged.index).days
        weights = decay_rate ** (days_from_end / 30)
        
        model = LinearRegression()
        
        model.fit(X, y, sample_weight=weights)
        r2_weighted = model.score(X, y, sample_weight=weights)
    
        exposures.append({
            'asset': ticker,
            'alpha': model.intercept_,
            'mkt': model.coef_[0],
            'size': model.coef_[1], 
            'value': model.coef_[2],
            'profit': model.coef_[3],
            'invest': model.coef_[4],
            'mom': model.coef_[5],
            'r2': r2_weighted,
            'dollar_exposure': positions_values.get(ticker, 0)
        })
    
    return pd.DataFrame(exposures)

def calculate_portfolio_projections(current_value, growth_rate, end_date, standard_invest, fmv, vest_per_day):
    """Calculate portfolio value projections including contributions"""
    days_to_end = (end_date - datetime.now().date()).days
    daily_growth_rate = (1 + growth_rate) ** (1/365)
    
    intervals = list(range(0, days_to_end + 1, max(1, days_to_end // 100)))
    if intervals[-1] != days_to_end:
        intervals.append(days_to_end)
    
    dates = [datetime.now().date() + timedelta(days=d) for d in intervals]
    values = []
    
    for dte in intervals:
        # Calculate contributions (simplified)
        num_contributions = int(dte / 15.22)  # Bi-weekly contributions
        future_contributions = sum([
            standard_invest * (daily_growth_rate ** max(0, dte - n * 15))
            for n in range(num_contributions)
        ])
        
        # Calculate vesting value
        vesting = dte * vest_per_day * (fmv - 3.21) * (daily_growth_rate ** dte)
        
        # Total estimated value
        estimated_value = (current_value * (daily_growth_rate ** dte) + future_contributions + vesting)//100 * 100
        values.append(estimated_value)
    
    return dates, values, future_contributions

def main():
    st.title("📈 Portfolio Allocation Dashboard")
    
    # Initialize session state
    if 'default_quantities' not in st.session_state:
        st.session_state.default_quantities = {
            'VOO': 5,
            'VXUS': 20,
            'VB': 5
        }
    
    # Sidebar inputs
    st.sidebar.header("Portfolio Settings")
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date() + timedelta(days=180)
    )
    
    cash = st.sidebar.number_input("Cash", value=1000.0, step=100.0)
    fmv = st.sidebar.number_input("Current FMV", value=10.0, step=0.01)
    growth_rate = st.sidebar.number_input("CAGR (%)", value=7.0, step=0.1) / 100
    standard_invest = st.sidebar.number_input("Standard Contribution", value=1000.0, step=100.0)
    
    st.sidebar.subheader("Stock Holdings")
    
    # Stock quantity inputs
    quantities = {}
    for ticker, default_qty in st.session_state.default_quantities.items():
        quantities[ticker] = st.sidebar.number_input(
            f"{ticker}",
            value=float(default_qty),
            step=0.1,
            key=f"qty_{ticker}"
        )
    
    tickers = list(quantities.keys())
    stock_data = get_stock_data(tickers)
    exchange_rate = get_exchange_rate()
    factor_data = load_factor_data()
    
    # Calculate portfolio metrics
    positions_values = calculate_portfolio_metrics(stock_data, quantities)
    
    # Group assets
    asset_groups = {
        'Cash': int(cash),
        'ISOs': 1000.0,  # Simplified calculation
        'VOO': int(positions_values.get('VOO', 0)),
        'VB': int(positions_values.get('VB', 0)),
        'VXUS': int(positions_values.get('VXUS', 0))
    }
    
    # Create portfolio dataframe
    portfolio_df = pd.DataFrame([
        {'Asset': asset, 'Value': value}
        for asset, value in asset_groups.items()
        if value > 0
    ]).sort_values('Value', ascending=False)
    
    portfolio_df['Percentage'] = portfolio_df['Value'] / portfolio_df['Value'].sum()
    total_value = portfolio_df['Value'].sum()


    st.text("This app represents a flavor of what I personally use to track various aspects of my portfolio and how I project growth in the future." \
    "\n\nSince Dataiku's Data Scientist job description mentioned building web apps with Python frameworks, I've replicated my actual personal tracker originally built in RShiny in Streamlit with hypothetical assets and parameters to demonstrate how I see the skills being very transferrable." \
    "\n\nThe app automatically calls the yfinance API, fetches the latest asset prices, and calculates various metrics based on user inputs on the left.")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Exposures", "Projections", "ISO Visualizer"])
    
    with tab1:
        st.header("Portfolio Overview")
        # Calculate end estimate with contributions
        _, _, expected_contributions = calculate_portfolio_projections(
            total_value, growth_rate, end_date, standard_invest, fmv, 1250/365
        )
        
        # Top metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Portfolio Value", f"${total_value:,.0f}")
        
        with metric_col2:
            st.metric("EUR Value", f"€{exchange_rate * total_value:,.0f}")
        
        with metric_col3:
            days_to_end = (end_date - datetime.now().date()).days
            end_estimate = total_value * ((1 + growth_rate) ** (days_to_end / 365)) + expected_contributions
            st.metric("End Date Estimate", f"${end_estimate:,.0f}")
        
        chart_col, empty_col = st.columns([0.8, 0.2])

        with chart_col:
            fig = px.pie(
                portfolio_df,
                values='Value',
                names='Asset',
                title="Portfolio Allocation"
            )
            fig.update_traces(
                textinfo='label+percent',
                textposition='outside'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with empty_col:
            st.empty()
                
        # Portfolio table
        table_col, empty_col2 = st.columns([0.85, 0.15])

        with table_col:
            st.subheader("Portfolio Details")
            display_df = portfolio_df.copy()
            display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.0f}")
            display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with empty_col2:
            st.empty() 
    
    with tab3:
        st.header("Portfolio Value Projection")
        
        # Calculate projections
        dates, values, _ = calculate_portfolio_projections(
            total_value, growth_rate, end_date, standard_invest, fmv, 1250/365
        )
        
        # Create projection chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Projected Value',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Portfolio Growth Projection",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Portfolio Exposures")
        
        if len(positions_values) > 0:
            exposures_df = get_asset_exposures(tickers, positions_values, factor_data)
            
            if not exposures_df.empty:
                st.subheader("Asset Factor Exposures")
                
                st.text("Here we run a multiple linear regressions for each asset where we regress their performance against the five fama french and momentum factors. The goal is to understand how each asset in related to the factors that influence returns.")
                # Display exposures table
                display_exposures = exposures_df.copy()
                numeric_cols = ['alpha', 'mkt', 'size', 'value', 'profit', 'invest', 'mom', 'r2']
                for col in numeric_cols:
                    if col in display_exposures.columns:
                        display_exposures[col] = display_exposures[col].apply(lambda x: f"{x:.3f}")
                
                # Ensure values are numeric and handle NaN
                display_exposures['dollar_exposure'] = (
                    pd.to_numeric(display_exposures['dollar_exposure'], errors='coerce')
                    .fillna(0)
                    .astype(float)
                    .apply(lambda x: f"${x:,.0f}")
                )
                
                st.dataframe(display_exposures, use_container_width=True, hide_index=True)
                
                # Portfolio-level exposures
                st.subheader("Portfolio Factor Exposures")
                st.text("Here we run a multiple linear regression of the portfolio's performance against the five fama french and momentum factors. The resulting loadings will give guidance for the amount to short if one wishes to hedge against or amplify certain factors.")
                weights = exposures_df['dollar_exposure'] / exposures_df['dollar_exposure'].sum()
                portfolio_exposures = {}
                
                for col in numeric_cols[:-1]:
                    if col in exposures_df.columns:
                        portfolio_exposures[col] = (exposures_df[col] * weights).sum()
                
                portfolio_exp_df = pd.DataFrame([
                    {
                        'Factor': factor,
                        'Loading': f"{exposure:.3f}",
                        'Dollar Exposure': f"{exposure * exposures_df['dollar_exposure'].sum():,.0f}"
                    }
                    for factor, exposure in portfolio_exposures.items()
                ])

                
                st.dataframe(portfolio_exp_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Unable to calculate exposures - insufficient data")
        else:
            st.warning("No position data available for exposure analysis")
    
    with tab4:
        st.header("ISO Visualizer")

        fmv_range = list(range(10, 81, 10))
        
        iso_data = []
        for fmv_scenario in fmv_range:
            vested_value = 1000 * fmv_scenario 
            unvested_value = 2000 * fmv_scenario 
            # Removed Expected Appreciation as requested
            
            iso_data.append({
                'FMV': fmv_scenario,
                'Vested Value': vested_value,
                'Unvested Value': unvested_value
            })
        
        iso_df = pd.DataFrame(iso_data)
        iso_df_melted = iso_df.melt(
            id_vars=['FMV'], 
            var_name='Component', 
            value_name='Value'
        )
        
        # Create stacked bar chart
        fig = px.bar(
            iso_df_melted,
            x='FMV',
            y='Value',
            color='Component',
            title="ISO Value Analysis by FMV Scenario"
        )
        
        fig.update_layout(
            xaxis_title="Fair Market Value ($)",
            yaxis_title="Value at End Date ($)",
            legend_title="Component",
            xaxis=dict(
                tickmode='linear',
                tick0=10,
                dtick=10
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ISO summary table
        st.subheader("ISO Scenario Analysis")
        iso_display = iso_df.copy()
        for col in iso_display.columns:
            if col != 'FMV':
                iso_display[col] = iso_display[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(iso_display, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()