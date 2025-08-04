from google import genai
from os import environ
from dotenv import load_dotenv

load_dotenv()
cli = genai.Client(api_key=environ.get("GEMINI_API_KEY"))
File = cli.files.upload(file=input("Enter the file path with root dir: "))
response = cli.models.generate_content(
    model="gemini-1.5-flash",
    contents=[File, "\n\n",
              input("Enter your query: ")])

print(response.text)