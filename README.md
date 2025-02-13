# Syarah-Price-Prediction - Faqih-Asshiddik
# Intelligent Car Pricing: Building Syarah's Market Predictor

## Project Overview
This project develops a machine learning model to predict used car prices for Syarah, a leading automotive marketplace in Saudi Arabia. The system aims to provide data-driven price recommendations to help sellers set competitive prices and give buyers confidence in their purchasing decisions.

## Business Problem
Syarah's platform currently allows sellers complete freedom in pricing, leading to market inefficiencies where prices often rely on limited market knowledge rather than data-driven insights. This creates several challenges:
- Sellers risk over or underpricing their vehicles
- Buyers struggle to evaluate fair market values
- Platform experiences reduced transaction efficiency due to misaligned price expectations

## Data Description
The dataset contains information about used cars listed for sale in Saudi Arabia, including:
- Vehicle characteristics (make, type, year, engine size)
- Usage details (mileage)
- Market positioning (region, options, gear type)
- Price information

## Methodology

### Data Preprocessing
1. Cleaned and standardized data by:
   - Handling unknown and other categories
   - Removing duplicate entries
   - Setting logical bounds for numerical features
   - Grouping low-frequency categories

2. Feature Engineering:
   - Transformed categorical variables using appropriate encoding methods
   - Scaled numerical features using RobustScaler
   - Applied log transformation to price values

### Modeling Approach
1. Evaluated multiple algorithms:
   - Linear Regression (baseline)
   - K-Nearest Neighbors
   - Decision Trees
   - Random Forest
   - XGBoost

2. Selected XGBoost as the final model based on performance metrics
3. Optimized model through extensive hyperparameter tuning
4. Validated results using cross-validation and residual analysis

## Results
The final model achieved:
- Mean Absolute Percentage Error (MAPE): 17.5%
- Root Mean Square Error (RMSE): 15,452 SAR

Key findings from feature importance analysis:
1. Manufacturing year is the strongest price predictor
2. Mileage shows significant but non-linear impact
3. Engine size serves as a reliable indicator of vehicle class
4. Make and Type demonstrate substantial influence on pricing

## Technical Implementation
The project uses the following technologies:
- Python 3.x
- Key libraries: pandas, numpy, scikit-learn, xgboost
- Feature engineering tools: category_encoders
- Analysis tools: SHAP, matplotlib, seaborn

## Project Structure
```
├── notebooks/
│   └── car_price_prediction.ipynb
├── models/
│   ├── xgb_tuned_model.pkl
│   └── transformer.pkl
├── data/
│   └── data_saudi_used_cars.csv
└── README.md
```

## Model Deployment
The final model is saved using pickle and can be loaded for predictions:
```python
import pickle

# Load model and transformer
with open('models/xgb_tuned_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('models/transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Make predictions
predictions = model.predict(X_test)
```

## Business Recommendations
1. Platform Enhancement:
   - Implement automated pricing tool using the predictive model
   - Standardize listing format to capture key pricing factors

2. User Experience:
   - Provide real-time pricing insights based on similar transactions
   - Display price trends based on year and mileage correlations

3. Process Optimization:
   - Restructure listing process to ensure complete information
   - Develop automated quality checks for unusual price combinations

## Future Development
- Incorporate additional data sources (maintenance history, accident reports)
- Develop more granular models for specific vehicle segments
- Implement real-time model updates based on market changes

## Author
Faqih Asshiddik

## License
[Include appropriate license information]
