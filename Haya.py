import subprocess
import sys
import os

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required dependencies are installed
required_packages = [
    "flask", "flask-cors", "firebase-admin", "psycopg2", "torch", "transformers",
    "langchain", "langchain-openai", "requests", "faiss-cpu", "chromadb",
    "textblob", "deep-translator", "cryptography", "gtts", "summarizer", "apscheduler"
]
for package in required_packages:
    install_package(package)

from flask import Flask, request, jsonify, Response
import firebase_admin
from firebase_admin import credentials, firestore
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
import requests
from textblob import TextBlob
from deep_translator import GoogleTranslator
from cryptography.fernet import Fernet
from gtts import gTTS
from summarizer import Summarizer
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Firebase using Environment Variables
firebase_json = os.getenv("FIREBASE_CREDENTIALS")
if firebase_json:
    with open("firebase_credentials.json", "w") as f:
        f.write(firebase_json)
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    raise ValueError("Firebase credentials not set in environment variables")

app = Flask(__name__)

# Load AI Model (Using OpenAI API for GPT-4)
def generate_response(user_input):
    api_key = os.getenv("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")
    if not api_key:
        return "Error: OpenAI API Key not set"
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)
    conversation_chain = ConversationChain(llm=llm)
    return conversation_chain.run(user_input)

# Emotion Detection System
def detect_emotion(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2:
        return "positive"
    elif analysis.sentiment.polarity < -0.2:
        return "negative"
    return "neutral"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    detected_emotion = detect_emotion(user_input)
    if detected_emotion == "positive":
        user_input += " 😊"
    elif detected_emotion == "negative":
        user_input += " 😔"

    raw_response = generate_response(user_input)
    return jsonify({"response": raw_response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)