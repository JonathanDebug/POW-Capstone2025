import torch
import os
from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import pandas as pd
from bs4 import BeautifulSoup  # only needed if the emails have HTML
import kagglehub
import shutil
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


# Checks that CUDA is available
# Install torch from the torch website with the appropiate CUDA, for the AI to use your GPU instead of CPU (way faster).
# I used this (I have a RTX 2070): pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# If you get memory errors try this: pip install torch --index-url https://download.pytorch.org/whl/cu126
# pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
def check_cuda():
    print("CUDA Available?", torch.cuda.is_available(), "\nGPU:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")  # Should be True


def download_dataset():
    # Get the absolute path of the folder where this script is running
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Create a "Datasets" folder in the same directory if it doesn't exist
    DATASET_DIR = os.path.join(BASE_DIR, "Datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Download dataset from KaggleHub
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")

    # Move or link it into our Datasets folder
    print(f"Downloaded dataset to: {path}")
    print(f"Copying dataset into: {DATASET_DIR}")

    # If KaggleHub returns a folder, copy it inside "Datasets"

    if os.path.isdir(path):
        dest = os.path.join(DATASET_DIR, os.path.basename(path))
        if not os.path.exists(dest):
            shutil.copytree(path, dest)
    else:
        shutil.copy(path, DATASET_DIR)


def load_data():
    dfs = []
    dataset_paths = ["Datasets/1/CEAS_08.csv",  "Datasets/1/Enron.csv"]
    for path in dataset_paths:
        print(f"Loading dataset from: {path}")

        df = pd.read_csv(path, encoding="utf-8", on_bad_lines='skip')

        # Check columns
        print("Columns found:", df.columns.tolist())

        # Combine subject + body into one text column
        if "subject" in df.columns and "body" in df.columns:
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        elif "body" in df.columns:
            df["text"] = df["body"].fillna("")
        else:
            raise ValueError("Dataset must contain either 'body' or 'text' column.")

        # Keep only text + label for training
        df = df[["text", "label"]]
        print("Loaded", len(df), "emails.")
        dfs.append(df)

    # Combine all datasets
    full_df = pd.concat(dfs, ignore_index=True)
    print("Total emails loaded from all datasets:", len(full_df))
    return full_df




def train_model(df):
    # Split train/test 80% train 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to HF datasets
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir="./bert-phishing-model",
        do_eval=True,
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # 3 epochs (3 times running through the same dataset)
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()
    trainer.save_model("./bert-phishing-final")
    tokenizer.save_pretrained("./bert-phishing-final")
    print("Training complete. Model saved to ./bert-phishing-final")



def evaluate_model():
    # Load your fine-tuned pipeline
    pipe = pipeline("text-classification",
                    model="./bert-phishing-final",
                    tokenizer="./bert-phishing-final",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
    )

    # Load test dataset
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    y_true = test_df["label"].tolist()
    y_pred = []

    print("Running predictions on test set...")
    for text in test_df["text"]:
        result = pipe(text)[0]  # pipeline returns a list of dicts
        label = result["label"]
        # Convert 'LABEL_0'/'LABEL_1' to 0/1
        if label in ["LABEL_0", "Safe"]:
            y_pred.append(0)
        else:
            y_pred.append(1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n Accuracy: {acc * 100:.2f}%")
    print("\n Confusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred))


# =============================
# Main
# =============================
if __name__ == "__main__":
    evaluate_model()

# Optional: function to clean HTML
def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()


