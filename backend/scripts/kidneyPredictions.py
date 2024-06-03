import pandas as pd
import joblib
import sys
import json

# Load the trained Random Forest model
rf_model = joblib.load('./models/kidney.pkl')

# Function to make predictions
def make_predictions(input_data):
    try:
        # Parse the input data from JSON
        input_data = json.loads(input_data)

        for key, value in input_data.items():
            if value == '':
                input_data[key] = None 

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
    
    # Call make_predictions function and return results
    result = make_predictions(input_data)
    print(result)  # Return the predictions as a string
