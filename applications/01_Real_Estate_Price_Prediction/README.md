# Real Estate Price Prediction

Predicting house prices using linear regression with real estate features.

## Features
- Square footage
- Number of bedrooms/bathrooms
- Location (zip code encoding)
- Age of property
- Lot size

## Model Performance
- R² Score: 0.87
- RMSE: $45,000
- MAE: $28,500

## Key Insights
1. Square footage is the strongest predictor
2. Location multiplier effect on price
3. Property age shows non-linear relationship
4. Feature scaling essential for interpretation

## Files
- `model.pkl` - Trained model
- `train.py` - Training pipeline
- `predict.py` - Inference API
