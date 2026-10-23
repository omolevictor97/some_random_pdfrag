import os
import io
import base64
import pygame
import time
from groq import Groq
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

if os.path.exists('.env'):
    load_dotenv(find_dotenv())

# create a groq client
client = Groq()


def generate_voice(input_text, speech_file_path):
    
    response = client.audio.speech.create(
        model = "canopylabs/orpheus-v1-english",
        voice= 'daniel', 
        response_format = 'wav',
        input = input_text
    )
    return response

