
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessing:
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lm = WordNetLemmatizer()
    
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