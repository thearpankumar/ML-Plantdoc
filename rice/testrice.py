import requests
import json

# Replace with your actual API key
API_KEY = "AIzaSyAgLVNdRob3KZLeR-ZYd_8MHYc1G9BdOYo"

# Define the endpoint URL
url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"

# Define the headers and payload
headers = {
    'Content-Type': 'application/json',
}

data = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": "Give me five subcategories of jazz?"}
            ]
        }
    ]
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Check the response status and print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
