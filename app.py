from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
rf_model = joblib.load('rf_model.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get inputs from the frontend
            close_lag1 = float(request.form['close_lag1'])
            high_lag1 = float(request.form['high_lag1'])
            low_lag1 = float(request.form['low_lag1'])
            volume_lag1 = float(request.form['volume_lag1'])
            open_price = float(request.form['open_price'])

            # Prepare the input data
            input_data = np.array([[close_lag1, high_lag1, low_lag1, volume_lag1, open_price]])

            # Make prediction
            predicted_close = rf_model.predict(input_data)[0]

            return render_template("index.html", predicted_close=round(predicted_close, 2))
        
        except ValueError as e:
            return f"Invalid input: {e}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
