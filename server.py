from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import math
from fastapi.middleware.cors import CORSMiddleware


# Command to run server: uvicorn server:app --reload --host 0.0.0.0 --port 8000
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Email(BaseModel):
    subject: str
    sender: str
    body: str

print("Running AI Model")
classifier = pipeline("text-classification",
                model="./bert-phishing-final",
                tokenizer="./bert-phishing-final",
                truncation=True,
                max_length=512,
                padding="max_length"
                )

@app.post("/analyze-email")
def analyze_email(email:Email):
    print("Analyzing Email")
    text = f"{email.subject} {email.body}"
    sender = email.sender
    domain = sender.split("@")[1]
    isUPR = True
    print("Domain: " , domain)
    if domain != "upr.edu":
        print("Email not from UPR")
        isUPR = False
    else:
        print("Email is from UPR")
        isUPR = True

    output = classifier(text)[0]  # this is a dict: {"label": "...", "score": 0.xx}


    score = output["score"]
    formatted_score = math.floor(score * 100) / 100
    label = ""
    if output["label"] == "LABEL_0":
        label = False
    else:
        label = True



    return{
        "phish": label,
        "accuracy": formatted_score,
        "upr": isUPR
    }

# test
