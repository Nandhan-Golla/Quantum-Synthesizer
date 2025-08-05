import google.genai as gai
import os
#from IPython.display import Markdown
import dotenv as env


class load_data:
    def __init__(self, dir, api_from_base=None, api_custom_model=None):
        self.dir = dir
        self.api_key = None
        
        # Always load environment variables
        env.load_dotenv()
        
        if api_from_base:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        elif api_custom_model:
            # Try multiple API key environment variables
            self.api_key = (os.environ.get("GEMINI_API_KEY_2") or 
                           os.environ.get("GEMINI_API_KEY_3"))
        else:
            # Default to primary API key
            self.api_key = os.environ.get("GEMINI_API_KEY")
            
        if not self.api_key:
            raise ValueError("Missing Gemini API key! Please set one of the following environment variables: GEMINI_API_KEY, GEMINI_API_KEY_2, or GEMINI_API_KEY_3")
    
    def process(self):
        #console = Console()
        ind_paths = []
        try:
            for items in os.listdir(self.dir):
                path = os.path.join(self.dir, items)
                ind_paths.append(path)
        except Exception as e:
            print(f"Exception occured: Please change the directory {e}")
       # inp = map(str, dir.append(input("enter your directory: ")))
        prompt = """You are a highly capable medical intelligence assistant.

You will be provided with multiple medical files in varying formats. These may include lab reports, ECG summaries, physician notes, discharge summaries, imaging reports, or other clinical data. These files may be scanned documents, typed notes, or structured reports.

Your task is to:
1. **Read, parse, and understand each document individually**, regardless of formatting or order.
2. **Merge** their information into a **single, well-organized master report**.
3. **Correct and resolve** inconsistencies or duplication in the data.
4. Use clear, clinical, and concise language suitable for both professionals and patients.

---

### ⛑ Output Format:
Please organize your output using the following structured sections:

1. **Patient Overview**
   - Name (if available)
   - Age / Sex
   - Known Conditions / History
   - Summary of uploaded report sources

2. **Vital Signs**
   - BP, HR, SpO2, Temperature, Respiratory Rate, etc.

3. **Lab Findings**
   - Highlight abnormal values with units (e.g., Glucose: 180 mg/dL — High)

4. **Diagnosis Summary**
   - List potential or confirmed diagnoses with supporting evidence

5. **Medications**
   - Existing medications (from reports)
   - Any medications suggested based on findings

6. **Clinical Recommendations**
   - Follow-up tests
   - Referrals
   - Lifestyle suggestions

7. **Flagged Concerns (if any)**
   - Critical or urgent issues
   - Contradictions or missing data in the reports

---

### 🧠 Behavior Expectations:
- Be cautious in interpreting unclear or ambiguous data.
- Use clinical reasoning to infer connections between different files.
- Do NOT hallucinate or assume information not provided.
- If important data is missing, state that clearly.

---

Now process all attached files and generate the master report as per the above format."""
        client = gai.Client(api_key=self.api_key)
        uploads = [client.files.upload(file=x) for x in ind_paths]
        report = client.models.generate_content(model='gemini-1.5-flash', contents=[[prompt] + uploads])
        return report.text


        