# src/model.py

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

class MedicalQABot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def train(self, questions, answers):
        X = self.vectorizer.fit_transform(questions)
        self.classifier.fit(X, answers)

    def answer(self, question):
        X = self.vectorizer.transform([question])
        prediction = self.classifier.predict(X)[0]
        confidence = self.classifier.predict_proba(X).max()
        return prediction, confidence

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(model_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)

    def load_model(self, model_dir):
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            self.classifier = pickle.load(f)
