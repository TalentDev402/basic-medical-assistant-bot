# train_bot.py

from src.data_preprocessing import load_and_preprocess_data
from src.model import MedicalQABot
from src.evaluate import evaluate_model

# Load and preprocess data
train_df, test_df = load_and_preprocess_data("data/medical_dataset.csv")

# Train model
bot = MedicalQABot()
bot.train(train_df["Question"].tolist(), train_df["Answer"].tolist())
bot.save_model("model")

# Evaluate model
accuracy = evaluate_model(bot, test_df["Question"].tolist(), test_df["Answer"].tolist())
print(f"Validation Accuracy: {accuracy:.2f}")
