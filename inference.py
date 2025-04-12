# inference.py

from src.model import MedicalQABot
import os

def run_chat():
    # Load model
    if not os.path.exists("model/"):
        print("❌ Model not found. Please run train_bot.py first to train and save the model.")
        return

    bot = MedicalQABot()
    bot.load_model("model")

    print("🩺 Medical Assistant Bot")
    print("Type your medical question below (or 'exit' to quit):")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! 👋")
            break
        answer, score = bot.answer(user_input)
        print(f"Bot (Score: {score:.2f}): {answer}")

if __name__ == "__main__":
    run_chat()
