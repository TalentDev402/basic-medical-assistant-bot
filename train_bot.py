# train_bot.py

import pandas as pd
from src.model import MedicalQABot
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def auto_detect_columns(df):
    col_names = [col.lower().strip() for col in df.columns]
    question_col = None
    answer_col = None
    for col in df.columns:
        lower = col.lower().strip()
        if "question" in lower and not question_col:
            question_col = col
        if "answer" in lower and not answer_col:
            answer_col = col
    if not question_col or not answer_col:
        raise ValueError("Could not detect question and answer columns. Please rename manually.")
    return question_col, answer_col

# Load dataset
df = pd.read_csv("data/medical_dataset.csv")

# Auto-detect and rename question/answer columns
q_col, a_col = auto_detect_columns(df)
df = df.rename(columns={q_col: "Question", a_col: "Answer"})

# Drop nulls if any
df.dropna(subset=["Question", "Answer"], inplace=True)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train model
bot = MedicalQABot()
bot.train(train_df["Question"].tolist(), train_df["Answer"].tolist())
bot.save_model("model")

# Evaluate model
accuracy = evaluate_model(bot, test_df["Question"].tolist(), test_df["Answer"].tolist())
print(f"Validation Accuracy: {accuracy:.2f}")
