import google.genai as gai
import os
from IPython.display import Markdown
import dotenv as env

class load_data:
    def __init__(self, dir, api_from_base=True, api_custom_model=False):
        self.dir = dir
        #self.api = api
        if api_from_base:
            env.load_dotenv()
            self.api_key_dir = os.environ.get("GEMINI_API_KEY")
        elif api_custom_model:
            self.api_key_dir = input("Enter your Custom API Key: ") 
    
    def process(self):
        ind_paths = []
        try:
            for items in os.listdir(dir):
                path = os.path.join(dir, items)
                ind_paths.append([*map(str, items)])
        except Exception:
            print("Exception occured: Please change the directory")
       # inp = map(str, dir.append(input("enter your directory: ")))
        prompt = os.environ.get("SYSTEM_PROMPT")
        client = gai.Client(api_key=self.api_key_dir)
        uploads = [client.files.upload(x) for x in ind_paths]
        report = client.models.generate_content(model='gemini-1.5-flash', contents=[[prompt] + uploads])
        return Markdown(report.text)


        