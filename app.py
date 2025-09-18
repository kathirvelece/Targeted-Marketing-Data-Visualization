from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ✅ Load the model and ignore feature_names
with open('model.pkl', 'rb') as f:
    model_tuple = pickle.load(f)
    model = model_tuple[0]  # only extract the actual model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get values from form
        age = float(request.form.get('age', 0))
        income = float(request.form.get('income', 0))
        spending_score = float(request.form.get('spending_score', 0))
        membership_years = float(request.form.get('membership_years', 0))
        purchase_freq = float(request.form.get('purchase_frequency', 0))
        last_purchase_amount = float(request.form.get('last_purchase_amount', 0))

        # ✅ Ensure feature order matches training
        input_data = np.array([[age, income, spending_score, membership_years, purchase_freq, last_purchase_amount]])

        # ✅ Predict CLV
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=f"₹{prediction}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
