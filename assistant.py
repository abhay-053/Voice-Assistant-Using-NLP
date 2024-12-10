import os
import webbrowser
import pyttsx3
import datetime
import time
import pyautogui
import speech_recognition as sr
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Load JSON file
with open("intents.json") as file:
    data = json.load(file)

model = load_model("chats.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Initialize TTS engine globally
engine = pyttsx3.init("nsss")
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[1].id)
engine.setProperty('rate', engine.getProperty('rate') - 50)
engine.setProperty('volume', engine.getProperty('volume') + 0.25)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening...")
        audio = r.listen(source, phrase_time_limit=10)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
    except Exception:
        print("Sorry, I didn't catch that.")
        return "None"
    return query.lower()

def cal_day():
    day_dict = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    day = datetime.datetime.today().weekday() + 1
    return day_dict.get(day, "Unknown")

def wishMe():
    hour = int(datetime.datetime.now().hour)
    t = time.strftime("%I:%M:%p")
    day = cal_day()
    if 0 <= hour < 12 and "AM" in t:
        speak(f"Good Morning Abhay, it's {day} and the time is {t}")
    elif 12 <= hour < 16 and "PM" in t:
        speak(f"Good Afternoon Abhay, it's {day} and the time is {t}")
    else:
        speak(f"Good Evening Abhay, it's {day} and the time is {t}")

def social_media(command):
    urls = {
        'facebook': "https://www.facebook.com/",
        'whatsapp': "https://web.whatsapp.com/",
        'linkedin': "https://www.linkedin.com/",
        'instagram': "https://www.instagram.com/",
        'reddit': "https://www.reddit.com/"
    }
    for key, url in urls.items():
        if key in command:
            speak(f"Opening {key}")
            webbrowser.open(url)
            return
    speak("I am not sure which social media platform you want to open.")

def openApp(application):
    app_dict = {
        'safari': '/Applications/Safari.app',
        'chrome': '/Applications/Google Chrome.app',
        'messages': '/System/Applications/Messages.app',
        'mail': '/System/Applications/Mail.app',
        'system settings': '/System/Applications/System Settings.app',
        'visual studio code': '/Applications/Visual Studio Code.app'
    }
    app_name = application.replace('open', '').strip()
    app_path = app_dict.get(app_name.lower())
    if app_path:
        speak(f"Opening {app_name}")
        os.system(f"open \"{app_path}\"")
    else:
        speak(f"Sorry, I cannot find the application {app_name}.")

def closeApp(application):
    process_dict = {
        'safari': 'Safari',
        'chrome': 'Google Chrome',
        'messages': 'Messages',
        'mail': 'Mail',
        'system settings': 'System Settings',
        'visual studio code': 'Visual Studio Code'
    }
    app_name = application.replace('close', '').strip()
    process_name = process_dict.get(app_name.lower())
    if process_name:
        is_running = os.system(f"pgrep -x \"{process_name}\" > /dev/null 2>&1") == 0
        if is_running:
            speak(f"Closing {app_name}")
            exit_code = os.system(f"killall \"{process_name}\" 2>/dev/null")
            if exit_code == 0:
                speak(f"{app_name} has been closed successfully.")
            else:
                speak(f"Failed to close {app_name}. Please try again.")
        else:
            speak(f"{app_name} is not currently running.")
    else:
        speak(f"Sorry, I cannot find the application {app_name}.")

def browsing(query):
    if 'chrome' in query or 'search on chrome' in query:
        speak("What should I search for on Chrome?")
        search_query = command().lower()
        if search_query != "none":
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        else:
            speak("No input received for the search.")
    elif 'youtube' in query or 'search on youtube' in query:
        speak("What should I search for on YouTube?")
        search_query = command().lower()
        if search_query != "none":
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        else:
            speak("No input received for the search.")

if __name__ == "__main__":
    wishMe()
    while True:
        query = input("Enter your command: ").lower()  # Replace with `command()` for voice input
        if 'exit' in query or 'quit' in query:
            speak("Goodbye Abhay!")
            break
        elif any(platform in query for platform in ['facebook', 'whatsapp', 'linkedin', 'instagram', 'reddit']):
            social_media(query)
        elif 'open' in query:
            openApp(query)
        elif 'close' in query:
            closeApp(query)
        elif 'search on chrome' in query or 'search on youtube' in query:
            browsing(query)
        else:
            # Predict intent using the trained model
            padding_sequences = pad_sequences(tokenizer.texts_to_sequences([query]), maxlen=20, truncating='post')
            prediction = model.predict(padding_sequences)
            confidence = np.max(prediction)  # Get confidence level
            tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Handle low confidence
            if confidence < 0.5:
                speak("I'm not sure I understood that. Could you rephrase?")
                continue

            # Match the intent tag
            for intent in data['intents']:
                if intent['tag'] == tag:
                    # Check for any special cases or context handling
                    if tag == 'datetime':
                        now = datetime.datetime.now()
                        response = f"Today's date is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p')}."
                        speak(response)
                    elif tag == 'jokes':
                        response = random.choice(intent['responses'])
                        speak(response)
                    else:
                        response = random.choice(intent['responses'])
                        speak(response)
                    break
            else:
                speak("Sorry, I didn't understand that. Please try again.")
