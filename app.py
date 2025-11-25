"""
Occupancy & Price Optimizer - Streamlit App
Predicts occupancy rates and finds optimal pricing for Airbnb listings
Uses trained XGBoost models from RDS files via rpy2
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import subprocess
import shutil
warnings.filterwarnings('ignore')

# ============================================================================
# CHECK R INSTALLATION
# ============================================================================

def check_r_installed():
    """Check if R is installed on the system"""
    r_executable = shutil.which('R')
    return r_executable is not None

r_is_installed = check_r_installed()

if not r_is_installed:
    st.error("""
    ❌ **R is not installed on this system**

    This app requires R and the following R packages:
    - tidymodels
    - stats
    - tune

    ### To install R:
    - **macOS**: `brew install r`
    - **Ubuntu/Debian**: `sudo apt-get install r-base r-base-dev`
    - **Windows**: Download from https://cran.r-project.org/bin/windows/base/

    ### After installing R, install R packages:
    ```r
    install.packages("tidymodels")
    install.packages("tune")
    ```

    After installation, please restart this app.
    """)

    # Diagnostic info
    with st.expander("📋 Diagnostic Information"):
        st.write("**System Information:**")
        try:
            import platform
            st.write(f"- Platform: {platform.system()}")
            st.write(f"- Architecture: {platform.machine()}")
            st.write(f"- Python: {platform.python_version()}")
        except:
            pass

        st.write("**R Status:**")
        st.write(f"- R executable found: {r_is_installed}")
        st.write(f"- PATH location: {shutil.which('R')}")

        st.write("**If using Streamlit Cloud:**")
        st.write("Ensure `packages.txt` in your repo root contains:")
        st.code("""r-base
r-base-dev""", language="text")
        st.write("(The file should be at the repo root, NOT in the app directory)")

    st.stop()

# Set page config
st.set_page_config(
    page_title="Occupancy & Price Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL LOADING (Using rpy2 to load RDS files directly)
# ============================================================================

@st.cache_resource
def load_models_from_rds():
    """Load pre-trained models directly from RDS files using rpy2"""
    try:
        from rpy2.robjects.packages import importr

        # Get models path - models are in ./models/ (same directory as app.py)
        script_dir = Path(__file__).parent.absolute()
        models_path = script_dir / "models"

        # Check path exists
        if not models_path.exists():
            st.error(f"❌ Models directory not found at {models_path}")
            return None, None, False

        # Load models using rpy2
        base = importr('base')

        # Load occupancy model
        model_file = str(models_path / "occupancy_model_final.rds")
        occupancy_model = base.readRDS(model_file)

        # Default values for neighborhoods, property types, and test metrics
        neighborhoods = ["Duomo", "Brera", "Navigli", "Centro Storico", "Corso Como"]
        property_types = ["Entire home/apt", "Private room"]
        test_metrics = {'rmse': 0.163, 'mae': 0.131, 'rsq': 0.504}

        model_info_dict = {
            'neighborhoods': neighborhoods,
            'property_types': property_types,
            'test_metrics': test_metrics
        }

        return occupancy_model, model_info_dict, True

    except Exception as e:
        st.warning(f"⚠️ Error loading RDS models with rpy2: {str(e)}")
        return None, None, False


@st.cache_resource
def load_models_from_csv():
    """Fallback: Load metadata from CSV (requires manual data export)"""
    try:
        models_path = Path(__file__).parent / "models"

        # Load neighborhoods and property types from CSV
        neighborhoods = pd.read_csv(models_path / "neighborhoods.csv")['neighbourhood'].tolist() if (models_path / "neighborhoods.csv").exists() else []
        property_types = pd.read_csv(models_path / "property_types.csv")['property_type'].tolist() if (models_path / "property_types.csv").exists() else []

        # Load metrics from JSON
        if (models_path / "test_metrics.json").exists():
            with open(models_path / "test_metrics.json") as f:
                test_metrics = json.load(f)
        else:
            test_metrics = {'rmse': 0.163, 'mae': 0.131, 'rsq': 0.504}

        model_info_dict = {
            'neighborhoods': sorted(neighborhoods) if neighborhoods else ["Duomo", "Brera", "Navigli"],
            'property_types': sorted(property_types) if property_types else ["Entire home/apt", "Private room"],
            'test_metrics': test_metrics
        }

        return None, model_info_dict, False

    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        return None, None, False


# Try loading models
occupancy_model, model_info, models_loaded = load_models_from_rds()

if not models_loaded or model_info is None:
    st.error("❌ Could not load models. Please ensure occupancy_model_final.rds exists in ./models/")
    model_info = {
        'neighborhoods': ["Duomo", "Brera", "Navigli", "Centro Storico"],
        'property_types': ["Entire home/apt", "Private room"],
        'test_metrics': {'rmse': 0.163, 'mae': 0.131, 'rsq': 0.504}
    }
else:
    st.success("✅ Models loaded successfully from RDS")

# ============================================================================
# INITIALIZE SESSION STATE FOR PRICE OPTIMIZATION
# ============================================================================
# Initialize all predictor session state keys with defaults to ensure they persist across page navigation

default_session_state = {
    'pred_neighbourhood': 'Duomo',
    'pred_property_type': 'Entire home/apt',
    'pred_price': 100,
    'pred_bedrooms': 1,
    'pred_bathrooms': 1,
    'pred_accommodates': 2,
    'pred_beds': 1,
    'pred_minimum_nights': 2,
    'pred_maximum_nights': 30,
    'pred_number_of_reviews': 100,
    'pred_host_is_superhost': True,
    'pred_host_has_profile_pic': True,
    'pred_host_identity_verified': True,
    'pred_host_phone_verification': True,
    'pred_host_work_email_verification': False,
    'pred_host_email_verification': True,
    'pred_calculated_host_listings_count': 1,
    'pred_review_scores_value': 4.8,
    'pred_review_scores_cleanliness': 4.7,
    'pred_review_scores_location': 4.9,
    'pred_review_scores_communication': 4.8,
    'pred_review_scores_rating': 4.85,
    'pred_amenities_count': 18,
    'pred_quality_rating': 'high',
    'pred_instant_bookable': 1,
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================================
# SIDEBAR - Navigation
# ============================================================================

st.sidebar.title("🏠 Occupancy Optimizer")
page = st.sidebar.radio(
    "Select Page",
    ["🎯 Predictor", "💰 Price Optimization", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: OCCUPANCY PREDICTOR
# ============================================================================

if page == "🎯 Predictor":
    st.header("Occupancy Predictor")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property Details")

        # Get neighborhoods from model info
        neighborhoods = sorted(model_info['neighborhoods']) if models_loaded else ["Duomo", "Brera", "Navigli"]
        property_types = sorted(model_info['property_types']) if models_loaded else ["Entire home/apt", "Private room"]

        neighbourhood = st.selectbox("Neighbourhood:", neighborhoods, key="pred_neighbourhood")
        property_type = st.selectbox("Property Type:", property_types, key="pred_property_type")

        col_price, col_beds = st.columns(2)
        with col_price:
            price = st.number_input("Price (€/night):", value=100, min_value=20, max_value=1000, step=5, key="pred_price")
        with col_beds:
            bedrooms = st.number_input("Bedrooms:", value=1, min_value=1, max_value=10, step=1, key="pred_bedrooms")

        col_bath, col_acc = st.columns(2)
        with col_bath:
            bathrooms = st.number_input("Bathrooms:", value=1, min_value=1, max_value=5, step=1, key="pred_bathrooms")
        with col_acc:
            accommodates = st.number_input("Accommodates:", value=2, min_value=1, max_value=12, step=1, key="pred_accommodates")

        beds = st.number_input("Beds:", value=1, min_value=1, max_value=10, step=1, key="pred_beds")

        st.subheader("Booking Policies")
        col_min, col_max = st.columns(2)
        with col_min:
            minimum_nights = st.number_input("Min Stay (nights):", value=2, min_value=1, max_value=90, key="pred_minimum_nights")
        with col_max:
            maximum_nights = st.number_input("Max Stay (nights):", value=30, min_value=30, max_value=365, key="pred_maximum_nights")

        number_of_reviews = st.number_input("Number of Reviews:", value=100, min_value=0, max_value=1000, key="pred_number_of_reviews")

    with col2:
        st.subheader("Host Features")

        col_super, col_pic = st.columns(2)
        with col_super:
            host_is_superhost = st.checkbox("Superhost", value=True, key="pred_host_is_superhost")
        with col_pic:
            host_has_profile_pic = st.checkbox("Has Profile Picture", value=True, key="pred_host_has_profile_pic")

        col_id, col_phone = st.columns(2)
        with col_id:
            host_identity_verified = st.checkbox("Identity Verified", value=True, key="pred_host_identity_verified")
        with col_phone:
            host_phone_verification = st.checkbox("Phone Verified", value=True, key="pred_host_phone_verification")

        col_work, col_email = st.columns(2)
        with col_work:
            host_work_email_verification = st.checkbox("Work Email Verified", value=False, key="pred_host_work_email_verification")
        with col_email:
            host_email_verification = st.checkbox("Email Verified", value=True, key="pred_host_email_verification")

        calculated_host_listings_count = st.number_input("Host Listings Count:", value=1, min_value=1, max_value=100, key="pred_calculated_host_listings_count")

        st.subheader("Review Scores")
        col_val, col_clean = st.columns(2)
        with col_val:
            review_scores_value = st.number_input("Value Score:", value=4.8, min_value=1.0, max_value=5.0, step=0.1, key="pred_review_scores_value")
        with col_clean:
            review_scores_cleanliness = st.number_input("Cleanliness Score:", value=4.7, min_value=1.0, max_value=5.0, step=0.1, key="pred_review_scores_cleanliness")

        col_loc, col_comm = st.columns(2)
        with col_loc:
            review_scores_location = st.number_input("Location Score:", value=4.9, min_value=1.0, max_value=5.0, step=0.1, key="pred_review_scores_location")
        with col_comm:
            review_scores_communication = st.number_input("Communication Score:", value=4.8, min_value=1.0, max_value=5.0, step=0.1, key="pred_review_scores_communication")

        col_rating, col_amenities = st.columns(2)
        with col_rating:
            review_scores_rating = st.number_input("Overall Rating:", value=4.85, min_value=1.0, max_value=5.0, step=0.1, key="pred_review_scores_rating")
        with col_amenities:
            amenities_count = st.number_input("Amenities Count:", value=18, min_value=0, max_value=50, key="pred_amenities_count")

        quality_rating = st.selectbox("Quality Rating:", ["low", "medium", "high"], index=2, key="pred_quality_rating")

    # Prediction button
    if st.button("🔮 Predict Occupancy", use_container_width=True, type="primary"):
        if models_loaded:
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'price': [price],
                    'instant_bookable': [1],
                    'maximum_nights': [maximum_nights],
                    'minimum_nights': [minimum_nights],
                    'number_of_reviews': [number_of_reviews],
                    'neighbourhood_cleansed': [neighbourhood],
                    'property_type': [property_type],
                    'accommodates': [accommodates],
                    'bathrooms': [bathrooms],
                    'bedrooms': [bedrooms],
                    'beds': [beds],
                    'amenities_count': [amenities_count],
                    'calculated_host_listings_count': [calculated_host_listings_count],
                    'host_has_profile_pic': [int(host_has_profile_pic)],
                    'host_is_superhost': [int(host_is_superhost)],
                    'host_phone_verification': [int(host_phone_verification)],
                    'host_work_email_verification': [int(host_work_email_verification)],
                    'host_email_verification': [int(host_email_verification)],
                    'host_identity_verified': [int(host_identity_verified)],
                    'review_scores_value': [review_scores_value],
                    'review_scores_cleanliness': [review_scores_cleanliness],
                    'review_scores_location': [review_scores_location],
                    'review_scores_communication': [review_scores_communication],
                    'review_scores_rating': [review_scores_rating],
                    'quality_rating': [quality_rating]
                })

                # Make prediction using R model via rpy2
                from rpy2.robjects import conversion, pandas2ri
                from rpy2 import robjects as ro

                # Load required R libraries for tidymodels prediction
                ro.r('library(stats)')
                ro.r('library(tidymodels)')
                ro.r('library(tune)')

                # Convert pandas DataFrame to R DataFrame using localconverter
                with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
                    r_input = conversion.py2rpy(input_data)

                    # Make prediction using the S3 predict method
                    predict_func = ro.r('predict')
                    r_pred = predict_func(occupancy_model, new_data=r_input)

                    # Convert back to Python
                    pred_values = conversion.rpy2py(r_pred)

                # Extract occupancy prediction
                if isinstance(pred_values, pd.DataFrame):
                    occupancy_pred = float(pred_values.iloc[0, 0])
                elif isinstance(pred_values, (list, np.ndarray)):
                    occupancy_pred = float(pred_values[0])
                else:
                    occupancy_pred = float(pred_values)

                # Display results
                st.divider()
                st.subheader("📈 Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Predicted Occupancy Rate",
                        f"{occupancy_pred*100:.1f}%",
                        delta=f"{occupancy_pred*100-50:.1f}% vs 50%"
                    )

                with col2:
                    annual_revenue = price * occupancy_pred * 365
                    st.metric(
                        "Annual Revenue",
                        f"€{annual_revenue:,.0f}",
                        delta=f"at €{price}/night"
                    )

                with col3:
                    platform_fee_pct = 0.03
                    net_profit = annual_revenue * (1 - platform_fee_pct)
                    st.metric(
                        "Annual Profit (after 3% fee)",
                        f"€{net_profit:,.0f}"
                    )

                # Detailed metrics table
                st.subheader("Key Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Nightly Price',
                        'Occupancy Rate',
                        'Days Booked/Year',
                        'Gross Revenue',
                        'Platform Fee (3%)',
                        'Annual Profit'
                    ],
                    'Value': [
                        f'€{price}',
                        f'{occupancy_pred*100:.1f}%',
                        f'{occupancy_pred*365:.0f} days',
                        f'€{annual_revenue:,.0f}',
                        f'€{annual_revenue*0.03:,.0f}',
                        f'€{net_profit:,.0f}'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        else:
            st.error("❌ Models not loaded. Cannot make predictions.")

# ============================================================================
# PAGE 2: PRICE OPTIMIZATION
# ============================================================================

elif page == "💰 Price Optimization":
    st.header("Find Optimal Price")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Optimization Parameters")

        opt_price_min = st.number_input("Minimum Price (€):", value=50, min_value=20, max_value=500)
        opt_price_max = st.number_input("Maximum Price (€):", value=300, min_value=50, max_value=1000)
        opt_step = st.number_input("Price Step (€):", value=5, min_value=1, max_value=50)

        st.divider()

        platform_fee = st.number_input("Platform Fee (%):", value=3, min_value=0, max_value=50)
        variable_cost = st.number_input("Variable Cost per Night (€):", value=10, min_value=0, max_value=100)
        fixed_costs = st.number_input("Fixed Annual Costs (€):", value=2000, min_value=0, max_value=50000)

    with col2:
        st.subheader("📊 Use Property Details From Predictor")
        st.info(
            "The price optimization uses the same property details from the predictor tab. "
            "Make sure to set those first, then come back here to optimize!"
        )

    if st.button("🎯 Find Optimal Price", use_container_width=True, type="primary"):
        if models_loaded:
            try:
                from rpy2.robjects import conversion, pandas2ri
                from rpy2 import robjects as ro

                # Get values from session state
                maximum_nights = st.session_state['pred_maximum_nights']
                minimum_nights = st.session_state['pred_minimum_nights']
                number_of_reviews = st.session_state['pred_number_of_reviews']
                neighbourhood = st.session_state['pred_neighbourhood']
                property_type = st.session_state['pred_property_type']
                accommodates = st.session_state['pred_accommodates']
                bathrooms = st.session_state['pred_bathrooms']
                bedrooms = st.session_state['pred_bedrooms']
                beds = st.session_state['pred_beds']
                amenities_count = st.session_state['pred_amenities_count']
                calculated_host_listings_count = st.session_state['pred_calculated_host_listings_count']
                host_has_profile_pic = st.session_state['pred_host_has_profile_pic']
                host_is_superhost = st.session_state['pred_host_is_superhost']
                host_phone_verification = st.session_state['pred_host_phone_verification']
                host_work_email_verification = st.session_state['pred_host_work_email_verification']
                host_email_verification = st.session_state['pred_host_email_verification']
                host_identity_verified = st.session_state['pred_host_identity_verified']
                review_scores_value = st.session_state['pred_review_scores_value']
                review_scores_cleanliness = st.session_state['pred_review_scores_cleanliness']
                review_scores_location = st.session_state['pred_review_scores_location']
                review_scores_communication = st.session_state['pred_review_scores_communication']
                review_scores_rating = st.session_state['pred_review_scores_rating']
                quality_rating = st.session_state['pred_quality_rating']

                # Generate price range
                prices = np.arange(opt_price_min, opt_price_max + opt_step, opt_step)

                # Load R libraries
                ro.r('library(stats)')
                ro.r('library(tidymodels)')
                ro.r('library(tune)')

                results_data = []

                # Test each price point
                with st.spinner(f"Testing {len(prices)} price points..."):
                    for test_price in prices:
                        # Create prediction data with test price
                        pred_data = pd.DataFrame({
                            'price': [test_price],
                            'instant_bookable': [1],
                            'maximum_nights': [maximum_nights],
                            'minimum_nights': [minimum_nights],
                            'number_of_reviews': [number_of_reviews],
                            'neighbourhood_cleansed': [neighbourhood],
                            'property_type': [property_type],
                            'accommodates': [accommodates],
                            'bathrooms': [bathrooms],
                            'bedrooms': [bedrooms],
                            'beds': [beds],
                            'amenities_count': [amenities_count],
                            'calculated_host_listings_count': [calculated_host_listings_count],
                            'host_has_profile_pic': [int(host_has_profile_pic)],
                            'host_is_superhost': [int(host_is_superhost)],
                            'host_phone_verification': [int(host_phone_verification)],
                            'host_work_email_verification': [int(host_work_email_verification)],
                            'host_email_verification': [int(host_email_verification)],
                            'host_identity_verified': [int(host_identity_verified)],
                            'review_scores_value': [review_scores_value],
                            'review_scores_cleanliness': [review_scores_cleanliness],
                            'review_scores_location': [review_scores_location],
                            'review_scores_communication': [review_scores_communication],
                            'review_scores_rating': [review_scores_rating],
                            'quality_rating': [quality_rating]
                        })

                        # Make prediction
                        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
                            r_input = conversion.py2rpy(pred_data)
                            predict_func = ro.r('predict')
                            r_pred = predict_func(occupancy_model, new_data=r_input)
                            pred_values = conversion.rpy2py(r_pred)

                        # Extract occupancy
                        if isinstance(pred_values, pd.DataFrame):
                            occupancy = float(pred_values.iloc[0, 0])
                        elif isinstance(pred_values, (list, np.ndarray)):
                            occupancy = float(pred_values[0])
                        else:
                            occupancy = float(pred_values)

                        # Cap occupancy between 0 and 1
                        occupancy = max(0, min(1, occupancy))

                        # Calculate financials
                        days_booked = occupancy * 365
                        gross_revenue = test_price * days_booked
                        platform_fee_amount = gross_revenue * (platform_fee / 100)
                        variable_cost_total = variable_cost * days_booked
                        net_profit = gross_revenue - platform_fee_amount - variable_cost_total - fixed_costs

                        results_data.append({
                            'Price (€)': test_price,
                            'Occupancy Rate': occupancy,
                            'Days Booked': days_booked,
                            'Gross Revenue': gross_revenue,
                            'Platform Fee': platform_fee_amount,
                            'Variable Costs': variable_cost_total,
                            'Net Profit': net_profit
                        })

                results_df = pd.DataFrame(results_data)

                # Find optimal price
                optimal_idx = results_df['Net Profit'].idxmax()
                optimal_price = results_df.loc[optimal_idx, 'Price (€)']
                optimal_profit = results_df.loc[optimal_idx, 'Net Profit']
                optimal_occupancy = results_df.loc[optimal_idx, 'Occupancy Rate']
                optimal_revenue = results_df.loc[optimal_idx, 'Gross Revenue']

                # Display results
                st.divider()
                st.subheader("💰 Price Optimization Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Optimal Price", f"€{optimal_price:.0f}",
                              delta=f"€{optimal_price - opt_price_min:.0f} from minimum")

                with col2:
                    st.metric("Expected Occupancy", f"{optimal_occupancy*100:.1f}%")

                with col3:
                    st.metric("Annual Revenue", f"€{optimal_revenue:,.0f}")

                with col4:
                    st.metric("Annual Profit", f"€{optimal_profit:,.0f}")

                # Display detailed table
                st.subheader("📊 Price Sensitivity Analysis")

                # Format the results table for display
                display_df = results_df.copy()
                display_df['Price (€)'] = display_df['Price (€)'].astype(int)
                display_df['Occupancy Rate'] = (display_df['Occupancy Rate'] * 100).apply(lambda x: f'{x:.1f}%')
                display_df['Days Booked'] = display_df['Days Booked'].apply(lambda x: f'{x:.0f}')
                display_df['Gross Revenue'] = display_df['Gross Revenue'].apply(lambda x: f'€{x:,.0f}')
                display_df['Platform Fee'] = display_df['Platform Fee'].apply(lambda x: f'€{x:,.0f}')
                display_df['Variable Costs'] = display_df['Variable Costs'].apply(lambda x: f'€{x:,.0f}')
                display_df['Net Profit'] = display_df['Net Profit'].apply(lambda x: f'€{x:,.0f}')

                # Highlight the optimal row
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Plot profit curve
                st.subheader("📈 Profit vs Price")

                fig = st.line_chart(
                    results_df.set_index('Price (€)')['Net Profit'],
                    use_container_width=True,
                    height=400
                )

                st.success(f"✅ Optimal price is **€{optimal_price:.0f}/night** with an expected annual profit of **€{optimal_profit:,.0f}**")

            except Exception as e:
                st.error(f"❌ Optimization error: {str(e)}")
        else:
            st.error("❌ Models not loaded. Cannot optimize pricing.")

# ============================================================================
# PAGE 3: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.header("About This App")

    st.markdown("""
    ### 🤖 Model Information

    This app uses an **XGBoost regression model** trained on 7,600+ Airbnb listings in Milan to predict occupancy rates.

    #### 📋 Model Features (25 inputs)
    - **Property characteristics**: bedrooms, bathrooms, accommodates, beds
    - **Booking policies**: minimum/maximum nights, number of reviews
    - **Host information**: superhost status, verification flags, profile picture
    - **Review scores**: cleanliness, communication, location, value, overall rating
    - **Quality rating**: low/medium/high categorical rating
    - **Amenities**: count of available amenities

    #### 🎯 Performance Metrics
    """)

    if models_loaded and 'test_metrics' in model_info:
        metrics = model_info['test_metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        col2.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        col3.metric("R² Score", f"{metrics.get('rsq', 0):.4f}")

    st.markdown("""
    #### 💡 How to Use
    1. **Go to Predictor tab**: Enter your property details
    2. **Click Predict**: Get occupancy rate and revenue estimate
    3. **Go to Price Optimization**: Find the price that maximizes profit
    4. **Review Results**: See sensitivity analysis across price points

    #### 💰 Revenue Calculation
    ```
    Annual Revenue = Price × Occupancy Rate × 365 days
    Net Profit = Gross Revenue - Platform Fee - Fixed Costs
    ```

    Example: €100/night × 48% occupancy × 365 days = €17,520 annual revenue

    #### 📈 About the Models
    - **Algorithm**: XGBoost Regression
    - **Training Data**: 7,600+ Milan Airbnb listings
    - **Target**: Average occupancy rate (0-1)
    - **Framework**: tidymodels (R) → Converted to Python

    ---

    **Built with**: Streamlit • Python • XGBoost
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>🏠 Occupancy & Price Optimizer | Built for Airbnb Hosts</small>
    </div>
    """,
    unsafe_allow_html=True
)
