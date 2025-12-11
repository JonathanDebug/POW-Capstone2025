import torch
import os
import time ## REQUIRED FOR TIME MEASUREMENT
from thop import clever_format, profile # REQUIRED FOR FLOPS MEASUREMENT
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

#SEBAS NOTES - I had to do a different pip install for pytorch for cuda 12.9 I think, but that depends on the GPU and the driver that you have installed. 
# If the code might not run, update GPU drivers and install the pip install correctly from pytorch. 

def check_cuda():
    print("CUDA Version: ", torch.version.cuda)
    print("CUDA Available?", torch.cuda.is_available(), "\nGPU:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")  # Should be True



#---------------------MEASUREMENT FUNCTIONS---------------------#
def measure_model_complexity(model_path="./bert-phishing-final", num_runs=100, sequence_length=256):
    """
    Measure time complexity and FLOPs of the trained model.
    
    Args:
        model_path: Path to the saved model
        num_runs: Number of inference runs for timing
        sequence_length: Input sequence length for measurements
    """
    
    # Load model and tokenizer
    print("Loading model for complexity analysis...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input for measurements
    dummy_text = """subject: Updating how Meta personalizes experiences
                    from: Instagram<no-reply@mail.instagram.com>
                    body: You can learn more about how we’re updating how we use your info in our Privacy Policy. Updating how Meta personalizes experiences Hi sorah350, On December 16, 2025, we’re making changes to our Privacy Policy. Here are some details. Personalizing your experiences We’ll start using your interactions with AIs to personalize your experiences and ads. What this means Personalizing your experiences includes suggesting content like posts that you may find interesting and reels to watch. It also includes showing ads that are more relevant to you. Thanks,
                    Meta Privacy © Meta. Meta Platforms, Inc., Attention: Community Support, 1 Meta Way, Menlo Park, CA 94025
                    This email was sent to btgmns@outlook.com. To help keep your account secure, please don’t forward this email.
                    Learn more """ * 10
    inputs = tokenizer(dummy_text, 
                      return_tensors="pt", 
                      truncation=True, 
                      padding="max_length", 
                      max_length=sequence_length)
    
    print(f"\n=== Model Complexity Analysis ===")
    print(f"Model: {model_path}")
    print(f"Input sequence length: {sequence_length}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Measure FLOPs
    print(f"\n--- FLOPs Measurement ---")
    try:
        macs, params = profile(model, 
                              inputs=(inputs['input_ids'], inputs['attention_mask']),
                              verbose=False)
        flops = macs * 2  # Convert MACs to FLOPs (approximately)
        
        # Format for better readability
        macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
        flops_formatted = clever_format([flops], "%.3f")
        
        print(f"MACs (Multiply-Accumulate Operations): {macs_formatted}")
        print(f"FLOPs (Floating Point Operations): {flops_formatted}")
        print(f"Parameters: {params_formatted}")
        
    except Exception as e:
        print(f"FLOPs measurement failed: {e}")
        print("Install thop for FLOPs measurement: pip install thop")
    
    # Measure inference time
    print(f"\n--- Time Complexity Measurement ---")
    
    # Warm-up runs
    print("Running warm-up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)
    
    # CPU timing
    if torch.cuda.is_available():
        model = model.cpu()
        print("Measuring CPU inference time...")
        cpu_times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                _ = model(**inputs)
                end_time = time.time()
                cpu_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        cpu_avg = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)
        print(f"CPU Inference Time: {cpu_avg:.2f} ± {cpu_std:.2f} ms")
        print(f"CPU Throughput: {1000/cpu_avg:.2f} samples/second")
    
    # GPU timing (if available)
    if torch.cuda.is_available():
        print("\nMeasuring GPU inference time...")
        model = model.cuda()
        inputs = {key: value.cuda() for key, value in inputs.items()}
        
        # GPU warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)
        torch.cuda.synchronize()  # Wait for GPU to finish
        
        gpu_times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()  # Ensure proper timing
                end_time = time.time()
                gpu_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        gpu_avg = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)
        print(f"GPU Inference Time: {gpu_avg:.2f} ± {gpu_std:.2f} ms")
        print(f"GPU Throughput: {1000/gpu_avg:.2f} samples/second")
        print(f"Speedup (GPU vs CPU): {cpu_avg/gpu_avg:.2f}x")
    
    # Memory usage analysis
    print(f"\n--- Memory Usage ---")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    print(f"Model size: {total_size / 1024**2:.2f} MB")
    print(f"Parameters size: {param_size / 1024**2:.2f} MB")
    print(f"Buffers size: {buffer_size / 1024**2:.2f} MB")
    
    # Layer-wise analysis (simplified)
    print(f"\n--- Model Architecture ---")
    print(f"Model type: {model.__class__.__name__}")
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'num_hidden_layers'):
            print(f"Number of layers: {config.num_hidden_layers}")
        if hasattr(config, 'hidden_size'):
            print(f"Hidden size: {config.hidden_size}")
        if hasattr(config, 'num_attention_heads'):
            print(f"Attention heads: {config.num_attention_heads}")



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


def load_synth_data():
    dfs = []
    dataset_paths = ["D:\github\POW-Capstone2025\GenDatasets\Phishing_Emails_test2\CSVs\Synth_POW.csv"]
    for path in dataset_paths:
        print(f"Loading dataset from: {path}")

        df = pd.read_csv(path, encoding="utf-8", on_bad_lines='skip')

        # Check columns
        print("Columns found:", df.columns.tolist())

        # Combine subject + body into one text column
        if "subject" in df.columns and "body" in df.columns:
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        elif "email_subject" in df.columns and "body" in df.columns:
            df["text"] = df["email_subject"].fillna("") + " " + df["body"].fillna("")
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
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

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



def evaluate_synth_model():
    # Load your fine-tuned pipeline
    pipe = pipeline("text-classification",
                    model="./bert-phishing-final",
                    tokenizer="./bert-phishing-final",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
    )

    # Load test dataset
    df = load_synth_data()
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

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
    check_cuda()
    
    # df = load_data()
    # train_model(df)
    evaluate_synth_model()
    

    # measure_model_complexity()


    # pipe = pipeline("text-classification",
    #                model="./bert-phishing-final",
    #                tokenizer="./bert-phishing-final",)
    # result = pipe("Helllo! we are excited to offer you a free iPhone! Click the link below to claim your prize.")
    # print(f"Prediction result: {result}")

# Optional: function to clean HTML
def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()


