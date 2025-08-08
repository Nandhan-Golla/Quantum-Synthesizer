import Input_loader
import dotenv
import os

dotenv.load_dotenv()
take = Input_loader.load_data(dir=input("enter the dir to be chosen: "), api_key=os.environ.get("GEMINI_API_KEY_2"))
