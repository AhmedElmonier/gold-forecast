import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN")
url = f"https://api.telegram.org/bot{token}/getUpdates"

print(url)
response = requests.get(url).json()
print("Chat History:")
for result in response.get("result", []):
    message = result.get("message", {})
    chat = message.get("chat", {})
    if chat:
        print(f"Chat ID: {chat.get('id')}, Type: {chat.get('type')}, User/Title: {chat.get('username') or chat.get('title') or chat.get('first_name')}")
