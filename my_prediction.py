import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Prediction:
    
    def __init__(self, pred_data):
        if isinstance(pred_data, str):  # If pred_data is a string
            self.pred_data = [pred_data]  # Wrap it in a list
        elif isinstance(pred_data, list) and all(isinstance(item, str) for item in pred_data):  # If pred_data is a list of strings
            self.pred_data = pred_data
        else:
            # Log a warning message here if needed
            self.pred_data = None  # Set pred_data to None if it's not a string or list of strings
        self.loaded_rf_model = self.load_rf_model()
        self.tf = self.load_tf()
        self.stopwords = set(stopwords.words('english'))
        self.lm = WordNetLemmatizer()
    
    def load_tf(self):
        with open('tf.pkl', 'rb') as f:
            loaded_tf = pickle.load(f)
        return loaded_tf
       
    def load_rf_model(self):
        # Load the saved Random Forest model from the file
        with open('rf_model.pkl', 'rb') as f:
            loaded_rf_model = pickle.load(f)
        return loaded_rf_model
     
    def text_preprocessing_user(self, data):
        if isinstance(data, str):  # Check if input is a string
            review = data.lower()
            review = review.split()
            review = [self.lm.lemmatize(x) for x in review if x not in self.stopwords]
            review = " ".join(review)
            return review
        elif isinstance(data, list):  # Check if input is a list
            preprocess_data = []
            for item in data:
                review = item.lower()
                review = review.split()
                review = [self.lm.lemmatize(x) for x in review if x not in self.stopwords]
                review = " ".join(review)
                preprocess_data.append(review)
            return preprocess_data
        else:
            # Log a warning message here if needed
            return None  # Return None if input data is not a string or list of strings
     
    def prediction_model(self, model):   
        if self.pred_data is None:  # Check if pred_data is None
            return "Invalid input data", None
        
        if isinstance(self.pred_data, str):  # If pred_data is a string
            preprocessed_data = self.text_preprocessing_user(self.pred_data)  # Don't wrap it in a list
            preprocessed_data = [preprocessed_data]  # Wrap it in a list for consistency
        elif isinstance(self.pred_data, list) and all(isinstance(item, str) for item in self.pred_data):  # If pred_data is a list of strings
            preprocessed_data = self.text_preprocessing_user(self.pred_data)
        else:
            # Log a warning message here if needed
            return "Invalid input data", None
    
        data = self.tf.transform(preprocessed_data)
        prediction_prob = model.predict_proba(data)
        prediction = model.predict(data)
        
        if prediction[0] == 0:
            result = "The News Is Fake"
        else:
            result = "The News Is Real"
        
        probability_real = prediction_prob[0][1]  # Probability of the news being real
        return result, probability_real  # Return both the prediction and the probability of being real
