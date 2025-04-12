from src.model import MedicalQABot
from src.data_preprocessing import clean_text

def interactive_chat():
    bot = MedicalQABot()
    bot.load_model("model")

    print("Medical Assistant Bot. Ask me anything about diseases. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        cleaned_input = clean_text(user_input)
        response, score = bot.get_answer(cleaned_input)
        print(f"Bot (Score: {score:.2f}): {response}\n")
