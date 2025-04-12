from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_model(bot, test_questions, test_answers):
    correct = 0
    for q, a in zip(test_questions, test_answers):
        predicted_answer, _ = bot.get_answer(q)
        if a.lower() in predicted_answer.lower():  # Loose match
            correct += 1
    accuracy = correct / len(test_questions)
    return accuracy
