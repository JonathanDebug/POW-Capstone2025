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
    print("Analyzing Email...")
    text = f"{email.subject} {email.body}"
    sender = email.sender
    domain = "Not found"
    isUPR = False
    if "<" in sender and "@" in sender:
        domain = sender.split("<")[1]
        domain = domain.split(">")[0]
        domain = domain.split("@")[1]
        if domain != "upr.edu":
            print("Email not from UPR")
            isUPR = False
        else:
            isUPR = True

    output = classifier(text)[0]  # this is a dict: {"label": "...", "score": 0.xx}

    score = output["score"]
    formatted_score = math.floor(score * 100)
    label = ""
    if output["label"] == "LABEL_0":
        label = False
    else:
        label = True


    print("Sender: " + sender)
    print("Content: " + text)
    print("#######################")
    print("Domain: " + domain)
    print("Domain is from UPR: " + str(isUPR))
    print("Label: " + output["label"])

    if label:
        print("AI Model has detected phishing in this email.")
    else:
        print("Phishing not detected")
    print("#######################")

    return{
        "phish": label,
        "accuracy": formatted_score,
        "upr": isUPR
    }

# test
