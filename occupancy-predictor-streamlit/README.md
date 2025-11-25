# 🏠 Occupancy & Price Optimizer

A Streamlit web application that predicts occupancy rates for Airbnb listings and finds optimal pricing strategies using machine learning.

## Features

### 🎯 Occupancy Predictor
- Predict occupancy rates based on property characteristics
- Input 25+ property features (bedrooms, bathrooms, amenities, reviews, host info, etc.)
- Get instant revenue projections at different price points

### 💰 Price Optimization
- Automatically test multiple price points
- Analyze profit sensitivity across different prices
- Find the optimal nightly rate that maximizes annual profit
- Account for platform fees, variable costs, and fixed costs

### 📊 About & Metrics
- View model performance metrics (RMSE, MAE, R²)
- Learn about the XGBoost model trained on 7,600+ Milan Airbnb listings
- Review revenue calculations

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python with rpy2 (R integration)
- **ML Model**: XGBoost (via tidymodels in R)
- **Data Processing**: pandas, numpy

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd occupancy-predictor-streamlit
   ```

2. **Install R** (required)
   - macOS: `brew install r`
   - Ubuntu/Debian: `sudo apt-get install r-base r-base-dev`
   - Windows: Download from https://cran.r-project.org/bin/windows/base/

3. **Install R packages**
   ```r
   install.packages("tidymodels")
   install.packages("tune")
   ```

4. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Place model files**
   - Create a `models/` directory
   - Add `occupancy_model_final.rds` to the `models/` folder
   - Add `occupancy_model_info.rds` to the `models/` folder

7. **Run the app**
   ```bash
   streamlit run app.py
   ```

### Deploy to Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add occupancy predictor"
   git push
   ```

2. **Deploy on Streamlit**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app" and select your repository
   - Set `app.py` as the main file
   - Deploy!

**Important**: Make sure `packages.txt` is in the **repository root** (not in a subdirectory). Streamlit Cloud uses this file to install system dependencies like R.

## Project Structure

```
occupancy-predictor-streamlit/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (R)
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── models/
│   ├── occupancy_model_final.rds  # Trained XGBoost model
│   └── occupancy_model_info.rds   # Model metadata
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## Model Details

- **Algorithm**: XGBoost Regression
- **Training Data**: 7,600+ Milan Airbnb listings
- **Features**: 25 property and host characteristics
- **Target**: Occupancy rate (0-1)
- **Performance**:
  - RMSE: 0.163
  - MAE: 0.131
  - R²: 0.504

## Model Inputs

### Property Characteristics
- Bedrooms, bathrooms, accommodates, beds
- Amenities count
- Instant bookable status

### Booking Policies
- Minimum/maximum stay nights
- Number of reviews

### Host Information
- Superhost status
- Profile picture, identity verification
- Phone, email, work email verification
- Number of listings

### Review Scores
- Value, cleanliness, location, communication
- Overall rating

### Quality Indicators
- Quality rating (low/medium/high)
- Price per night

## Usage Guide

### Step 1: Predictor Tab
1. Navigate to the **Predictor** tab
2. Fill in your property details
3. Click **Predict Occupancy**
4. Review results and revenue projections

### Step 2: Price Optimization Tab
1. After using Predictor, go to **Price Optimization**
2. Set optimization parameters:
   - Minimum and maximum price
   - Price step (testing interval)
   - Platform fees, variable costs, fixed costs
3. Click **Find Optimal Price**
4. Review the sensitivity analysis table and profit curve

### Step 3: Results
- **Key Metrics**: See optimal price and expected profit
- **Sensitivity Table**: View all price points and metrics
- **Profit Curve**: Visualize profit across price range

## Example Calculation

Property details:
- 1 bedroom, 1 bathroom, Duomo neighborhood
- €100/night price
- Predicted occupancy: 48%

Annual revenue: €100 × 0.48 × 365 = €17,520
Platform fee (3%): €525.60
Annual profit: €16,994.40

## Troubleshooting

### "R is not installed on this system"
- Install R using the commands shown in the error message
- Restart the app after installation

### "Models directory not found"
- Ensure `occupancy_model_final.rds` exists in the `models/` folder
- Check that the models folder is in the same directory as `app.py`

### On Streamlit Cloud: "No module named 'rpy2'"
- Ensure `packages.txt` is in the **repository root**
- Check that it contains:
  ```
  r-base
  r-base-dev
  ```
- Redeploy the app

## Performance Tips

- Model predictions are cached, so repeated predictions with same property details load instantly
- Price optimization with many price points may take a few seconds
- Use reasonable price ranges (e.g., €50-€300) for faster results

## License

This project is provided as-is for analysis and prediction purposes.

## Support

For issues or questions:
1. Check the diagnostic information in the error message
2. Verify R and dependencies are installed
3. Ensure all model files are in place

---

**Built with**: Streamlit • Python • XGBoost • tidymodels
