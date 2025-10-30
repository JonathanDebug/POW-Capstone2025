import torch
import os
from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import pandas as pd
from bs4 import BeautifulSoup  # only needed if the emails have HTML
import kagglehub
import shutil

#Checks that CUDA is available
print("CUDA Available?",torch.cuda.is_available() , "\nGPU:",  torch.cuda.get_device_name(0)if torch.cuda.is_available() else "CUDA not available" )       # Should be True


# Install torch from the torch website with the appropiate CUDA, for the AI to use your GPU instead of CPU (way faster).
# I used this (I have a RTX 2070): pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# If you get memory errors try this: pip install torch --index-url https://download.pytorch.org/whl/cu126
# pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


classifier = pipeline(
    task="text-classification", # There are multiple tasks but for the project text-classification will be most usefull
    model="distilbert-base-uncased",
    dtype=torch.float16,
    device=0,
)



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


id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe" : 1}


def load_data():
    dfs = []
    for file in os.listdir("Datasets/1"):
        if file.endswith(".csv"):
            path = os.path.join("Datasets/1",file)
            print(f"Loading {path}...")
            df = pd.read_csv(path)

            if "body" in df.columns and "subject" in df.columns and "label" in df.columns:
                if "subject" in df.columns:
                    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
                else:
                    df["text"] = df["body"].fillna("")

                dfs.append(df[["text","label"]])
            else:
                print(f"Skipped {file}, missing required columns.")
    final_df = pd.concat(dfs,ignore_index=True)
    print("Total emails: ", len(final_df))
    return final_df



# Optional: function to clean HTML
def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()

# download_dataset()
# load_data()

result = classifier("I love using Hugging Face Transformers!")
print(result)