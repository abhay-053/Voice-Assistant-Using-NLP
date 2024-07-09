import speech_recognition as sr
from openai import OpenAI

client = OpenAI(api_key='your api key')
from gtts import gTTS
import os

# Set up OpenAI API key

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError:
            print("Sorry, the service is down.")
        return None

# Function to generate responses using OpenAI GPT model
def generate_response(text):
    response = client.completions.create(
        model="text-davinci-003",  # Or use "gpt-4" if available
        prompt=text,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to convert text to speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

# Main function to run the voice assistant
def main():
    while True:
        user_input = recognize_speech()
        if user_input:
            response = generate_response(user_input)
            print("Assistant: " + response)
            speak(response)

if __name__ == "__main__":
    main()
