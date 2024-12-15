# from flask import Flask, request, jsonify
# import joblib

# #acts as a central part my web server, handling incoming requests and sending back responses.
# app = Flask(__name__)

# #Loads a pre-trained machine learning model stored in the file "churn_model.pkl"
# # This allows the model to be used for predictions when the API receives a request.
# model = joblib.load("churn_model.pkl")


# # This decorator defines a route for the API.
# # It specifies that when the server receives a POST request at the /predict URL, the predict() function should handle it.

# @app.route('/predict', methods=['POST'])



# def predict():
#     data = request.json
#     prediction = model.predict([data])
#     return jsonify({'churn': prediction[0]})

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("churn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input
    data = request.json
    
    # Convert input to NumPy array (ensure it matches model input format)
    features = np.array([data['features']])
    
    # Make prediction
    prediction = model.predict(features)
    prob = model.predict_proba(features)[:, 1]  # Probability of churn (if supported)

    # Return prediction as JSON
    return jsonify({
        "prediction": int(prediction[0]),
        "churn_probability": float(prob[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
