from sklearn.metrics import accuracy_score

def evaluate_model(bot, questions, true_answers):
    predicted_answers = []

    for q in questions:
        predicted_answer, _ = bot.answer(q)  # ✅ fixed here
        predicted_answers.append(predicted_answer)

    return accuracy_score(true_answers, predicted_answers)