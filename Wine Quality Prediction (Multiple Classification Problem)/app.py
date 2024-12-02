from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Render the HTML form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_data = {
            'fixed_acidity': float(request.form['fixed_acidity']),
            'volatile_acidity': float(request.form['volatile_acidity']),
            'citric_acid': float(request.form['citric_acid']),
            'residual_sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free_sulfur_dioxide': float(request.form['free_sulfur_dioxide']),
            'total_sulfur_dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }

        # Create a DataFrame from input data
        input_df = pd.DataFrame(input_data, index=[0])

        # Apply scaling using StandardScaler
        scaler = StandardScaler()
        scaled_input = scaler.fit_transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)

        return render_template('index.html', prediction_text=prediction[0])

    return render_template('index.html', prediction_text='')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
