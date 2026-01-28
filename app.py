
import streamlit as st
import pandas as pd
import joblib
import xgboost # Required to load XGBoost model

# Define the path to the model file
# This path assumes the model is in the same directory as app.py or accessible
model_path = 'xgboost_model.pkl'

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    return model

model = load_model()

st.title('NYC Property Sale Price Prediction')
st.write('Enter the property details to predict its adjusted sale price.')

# Define input fields for each feature used in the model
# These should match the features in X_train from the notebook
zip_code = st.number_input('ZIP CODE', min_value=10001, max_value=11697, value=10009)
block = st.number_input('BLOCK', min_value=1, value=374)
lot = st.number_input('LOT', min_value=1, value=46)
tax_class_at_time_of_sale = st.number_input('TAX CLASS AT TIME OF SALE', min_value=1, max_value=4, value=1)
sale_year = st.number_input('SALE YEAR', min_value=2000, max_value=2024, value=2022)
sale_month = st.number_input('SALE MONTH', min_value=1, max_value=12, value=9)
land_square_feet = st.number_input('LAND SQUARE FEET', min_value=0.0, value=2116.0)
gross_square_feet = st.number_input('GROSS SQUARE FEET', min_value=0.0, value=4400.0)
year_built = st.number_input('YEAR BUILT', min_value=1700, max_value=2024, value=1900)
residential_units = st.number_input('RESIDENTIAL UNITS', min_value=0, value=1)
commercial_units = st.number_input('COMMERCIAL UNITS', min_value=0, value=0)

# For encoded categorical features, assuming the user would input the encoded numerical values
# In a real-world app, you might re-implement the encoding logic or provide dropdowns with original categories
neighborhood_encoded = st.number_input('NEIGHBORHOOD_encoded', value=1.791735e+06, format="%.6e")
building_class_at_time_of_sale_encoded = st.number_input('BUILDING CLASS AT TIME OF SALE_encoded', value=5.551262e+06, format="%.6e")
building_class_category_description_encoded = st.number_input('BUILDING CLASS CATEGORY DESCRIPTION_encoded', value=9.375650e+05, format="%.6e")

# Create a button to make predictions
if st.button('Predict Sale Price'):
    # Prepare the input data as a DataFrame, ensuring correct order and types
    input_data = pd.DataFrame([{
        'ZIP CODE': zip_code,
        'BLOCK': block,
        'LOT': lot,
        'TAX CLASS AT TIME OF SALE': tax_class_at_time_of_sale,
        'SALE YEAR': sale_year,
        'SALE MONTH': sale_month,
        'LAND SQUARE FEET': land_square_feet,
        'GROSS SQUARE FEET': gross_square_feet,
        'YEAR BUILT': year_built,
        'RESIDENTIAL UNITS': residential_units,
        'COMMERCIAL UNITS': commercial_units,
        'NEIGHBORHOOD_encoded': neighborhood_encoded,
        'BUILDING CLASS AT TIME OF SALE_encoded': building_class_at_time_of_sale_encoded,
        'BUILDING CLASS CATEGORY DESCRIPTION_encoded': building_class_category_description_encoded
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Approximate current property value: ${prediction:,.2f}')
