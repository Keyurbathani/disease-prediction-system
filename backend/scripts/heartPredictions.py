import pandas as pd
import joblib
import sys
import json

# Load the trained Random Forest model
rf_model = joblib.load('./models/heart.pkl')

# Function to make predictions
def make_predictions(input_data):
    try:
        # Parse the input data from JSON
        input_dict = json.loads(input_data)

    
        input_data = {
            'age': input_dict.get('age'),
            'sex': input_dict.get('sex'),
            'chest pain type': input_dict.get('chestPain'),
            'resting bp s': input_dict.get('restingBp'),
            'cholesterol': input_dict.get('cholesterol'),
            'fasting blood sugar': input_dict.get('fastingBloodSugar'),
            'resting ecg' : input_dict.get('restingEcg'),
            'max heart rate': input_dict.get('maxHeartRate'),
            'exercise angina': input_dict.get('exerciseAngina'),
            'oldpeak': input_dict.get('oldPeak'),
            'ST slope': input_dict.get('STslope'),
        }
        
        # Prepare the input data as a DataFrame
        df = pd.DataFrame(input_data, index=[0])
        
        # Make predictions
        predictions = rf_model.predict(df)
        
        # Convert predictions to a JSON string and return
        return json.dumps(predictions.tolist())
    
    except Exception as e:
        # Return an error message if an exception occurs
        return str(e)

# Main function to handle command line arguments
if __name__ == "__main__":
    # Extract input data from command line arguments
    input_data = sys.argv[1]
    
    # Call make_predictions function and print results
    print(make_predictions(input_data))  # Print the predictions
    # 0:not heart disease
    # 1:heart disease
