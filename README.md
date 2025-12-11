POW-Capstone2025
Repository for the Phish Outta Water (POW) Capstone Project — Fall 2025, UPRM

Phish Outta Water (POW) is a browser extension designed to help users detect phishing emails using machine learning. It analyzes email content and sender information to identify potentially malicious messages without compromising user privacy.

How to Run:
Install all required dependencies using the provided requirements.txt file:
pip install -r requirements.txt

It is recommended to install the dependencies inside a Python virtual environment.

Execute the DistilBERT.py file to fine-tune the model.

Once the model has been fine-tuned, run the backend server using the following command:
uvicorn server:app --reload --host 0.0.0.0 --port 8000

Finally, install the browser extension in Google Chrome using Developer Mode and use it within an Outlook email to scan messages.
