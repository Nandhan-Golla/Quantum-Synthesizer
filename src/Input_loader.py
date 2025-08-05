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
            self.api_key = os.environ.get("GEMINI_API_KEY_2") 
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

        prompt = """
You are a highly capable AI medical assistant designed for multi-modal, multi-document clinical reasoning.

You will be provided with multiple medical reports in various formats, including (but not limited to) lab reports, ECG summaries, physician notes, discharge summaries, imaging data, and other clinical diagnostics. These may be partially complete, inconsistent, or presented in different formats and quality.

---

### 🧠 Your Objectives:

1. **Ingest, parse, and understand** all uploaded files.
2. **Merge the information** into a single, unified, and precise master report.
3. **Correct inconsistencies**, deduplicate content, and highlight contradictions if present.
4. **Preserve all important detail** for downstream AI systems — especially for automated medication synthesis, drug generation, and pattern recognition.
5. **Critically evaluate** whether the information provided is sufficient to generate a clinically accurate and complete profile. If not, state it explicitly.

---

### 📋 Output Format (Structured Sections):

1. **Patient Overview**
   - Name (if available), Age, Sex
   - Known medical history or conditions
   - Origin and type of each uploaded report

2. **Vital Signs**
   - Include only clinically relevant vitals
   - Mention date/time if available

3. **Lab Findings**
   - Present abnormal and critical values
   - Maintain original units and references
   - Highlight possible interrelated issues (e.g., high creatinine + proteinuria)

4. **Diagnosis Summary**
   - List confirmed diagnoses
   - List suspected diagnoses with confidence level
   - Support each diagnosis with source evidence

5. **Medications**
   - Existing medications from input
   - Suggested medications (if confident)
   - Drug interactions or contraindications (if any)

6. **Clinical Recommendations**
   - Follow-up tests
   - Specialist referrals
   - Lifestyle / dietary suggestions (only if mentioned)

7. **Flagged Concepts or Urgencies**
   - Critical issues needing immediate attention
   - Contradictory or unclear information
   - Unexpected findings

8. **AI Drug-Agent Integration Layer**
   > This section is to be used by downstream AI agents (RLHF + Quantum Drug Simulation Agents).

   - **Medication Insight Summary**  
     Summarize the treatment direction suggested by the report and diagnostic evidence.

   - **Drug Generation Triggers**  
     Identify biomarker patterns, clusters, or unresolved conditions that may require custom or optimized drug generation.

   - **Missing Drug Targets**  
     If the patient's data suggests gaps in existing medications or if disease control is suboptimal, list those.

   - **Pattern Recognition Observations**  
     Include any novel patterns, unexplained symptom clusters, or combinations that are statistically or biologically interesting for an AI model to investigate further.

9. **Data Sufficiency Statement**
   - Clearly state if the uploaded reports were insufficient, contradictory, outdated, or missing key data (e.g., no vitals, no lab results, unclear prescriptions).
   - Use this structure:  
     > “⚠️ Warning: The provided data is insufficient to generate a medically complete report. Missing elements include: [list]. Please upload more complete records.”

10. **Next Steps / Continuation Prompt**
   - End the report with this friendly reminder:

     > “✅ Master report generation complete. To improve accuracy, insights, and medication recommendations, please upload additional medical reports or historical data when available.”

---

### 🚫 Safety and Grounding Rules

- DO NOT guess or hallucinate medical conditions.
- If information is missing or ambiguous, state it directly.
- If there is insufficient data for a diagnosis or drug recommendation, say so.
- Never suggest experimental treatments unless explicitly mentioned in the reports.

"""
        client = gai.Client(api_key=self.api_key)
        uploads = [client.files.upload(file=x) for x in ind_paths]
        report = client.models.generate_content(model='gemini-1.5-flash', contents=[[prompt] + uploads])
        return report.text


        