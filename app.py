from flask import Flask, render_template, request
from my_prediction import Prediction
from my_preprocessing import Preprocessing
import pickle

app = Flask(__name__)

# Load the saved Random Forest model from the file
with open('rf_model.pkl', 'rb') as f:
    loaded_rf_model = pickle.load(f)

# Initialize Preprocessing object
preprocessor = Preprocessing()

# Route for the home page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/whatisit')
def defination():
    return render_template('box1.html')

@app.route('/need')
def necessity():
    return render_template('box2.html')

@app.route('/withoutit')
def without():
    return render_template('box3.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')
# Route for making predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    # Get the data from the request
    data = request.form.get('text', '')

    # Preprocess the data
    preprocessed_data = preprocessor.text_preprocessing_user(data)

    # Initialize Prediction object
    predictor = Prediction(preprocessed_data)

    # Make predictions
    result = predictor.prediction_model(loaded_rf_model)

    # Render the result page with the prediction result
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
