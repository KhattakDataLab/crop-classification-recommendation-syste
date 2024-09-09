from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template("index.html")

# Route for handling the prediction
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extracting data from form fields
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare the feature array for prediction
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply MinMaxScaler and StandardScaler as done during training
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Make prediction using the loaded model
        prediction = model.predict(final_features)

        # Mapping prediction to the corresponding crop
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Get the crop name based on the model prediction
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated in the given conditions.".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        
    except ValueError:
        result = "Please enter valid numeric values for all fields."
    except Exception as e:
        result = f"An error occurred: {str(e)}"

    # Render the result in the HTML template
    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
