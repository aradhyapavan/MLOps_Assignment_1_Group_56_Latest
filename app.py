from flask import Flask, request, render_template
from joblib import load
import pandas as pd


app = Flask(__name__)


# Load the trained model
model = load('models/model.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        X1_high_level = float(request.form['X1-high_level'])
        X2_low_level = float(request.form['X2-low_level'])

        # Create a DataFrame for the input
        input_data = pd.DataFrame(
            [[X1_high_level, X2_low_level]],
            columns=['X1-high_level', 'X2-low_level']
        )

        # Make predictions
        predictions = model.predict(input_data)

        # Return the predictions as a JSON response
        return render_template(
            'index.html', prediction_text=f'Prediction: {predictions[0]}'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
