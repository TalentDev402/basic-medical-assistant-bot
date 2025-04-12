from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class MedicalQABot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.questions = []
        self.answers = []
        self.question_vectors = None

    def train(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.question_vectors = self.vectorizer.fit_transform(questions)

    def save_model(self, model_path="model"):
        joblib.dump((self.vectorizer, self.questions, self.answers, self.question_vectors), model_path)

    def load_model(self, model_path="model"):
        self.vectorizer, self.questions, self.answers, self.question_vectors = joblib.load(model_path)

    def get_answer(self, user_query):
        query_vector = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        best_match_index = similarities.argmax()
        return self.answers[best_match_index], similarities[best_match_index]
