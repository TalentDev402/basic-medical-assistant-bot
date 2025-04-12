import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=["Question", "Answer"], inplace=True)
    df["Question"] = df["Question"].apply(clean_text)
    df["Answer"] = df["Answer"].apply(str)
    return train_test_split(df, test_size=0.2, random_state=42)