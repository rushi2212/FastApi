import googleapiclient.errors
import googleapiclient.discovery
import os
import re
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from phi.tools.pubmed import PubmedTools
import tempfile


# Environment configuration
os.environ['TAVILY_API_KEY'] = "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"
os.environ['GOOGLE_API_KEY'] = "AIzaSyCRKWB-XFsghy70mD6f9tPYOZklWXZiTLQ"

app = FastAPI(
    title="Medical Report Image Analyzer API",
    description="API for analyzing medical report images and providing patient-friendly explanations",
    version="1.0.0"
)

# CORS middleware
origins = [
    "http://localhost",
    "https://curehub-bfrb.onrender.com/",
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1",

    # Add other allowed origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class MedicalReportImageAnalyzer:
    def __init__(self):
        self.report_agent = self._create_report_agent()
        self.research_agent = self._create_research_agent()

    def _create_report_agent(self) -> Agent:
        """Create the primary medical report image analysis agent"""
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            system_prompt="""You are a senior medical specialist with 15 years experience analyzing various medical reports. Your tasks:
1. Analyze medical report images (blood tests, scans, doctor's notes, etc.) with clinical precision
2. Identify 3-5 key findings needing further explanation
3. Mark each with [[RESEARCH:TERM]] notation
4. Create patient-friendly explanations using:
   - Simple analogies (e.g. "your HDL cholesterol is like a cleanup crew")
   - Visual emojis/icons when appropriate
   - Clear, reassuring language
5. For complex findings, use Tavily to find layperson explanations
6. Structure output with:
   - ✅ Normal Findings
   - 🔍 Notable Results
   - ❓ Terms Needing Clarification
   - 💡 Health Recommendations""",
            instructions="""Analysis Protocol:
1. First pass: Extract all medical data from images with 100% accuracy
2. Second pass: Convert to patient-friendly language
3. Research terms: Mark any needing literature support
4. Web search: Use Tavily for patient education materials
5. Final output: Combine all elements with professional tone""",
            tools=[TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))],
            markdown=True,
            debug_mode=True,
            show_tool_calls=False
        )

    def _create_research_agent(self) -> Agent:
        """Create specialized research agent for medical literature"""
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            system_prompt="""You are a medical research librarian. Your tasks:
1. For each [[RESEARCH:TERM]] from medical reports:
   - Find 2-3 most relevant PubMed articles
   - Include recent studies (priority to last 3 years)
   - Extract key conclusions in plain language
2. Format each finding with:
   - 🏥 Medical Term
   - 📚 Research Summary (1-2 sentences)
   - 🔗 DOI Link
   - 💡 Patient Implications""",
            tools=[PubmedTools(
                email="rushikesh220703@gmail.com",
                max_results=3,
            )],
            markdown=True,
            debug_mode=True,
            show_tool_calls=False
        )

    async def analyze(self, image_files: List[UploadFile]) -> dict:
        """Complete analysis pipeline for report images with error handling"""
        try:
            # Save uploaded files to temporary files
            temp_files = []
            image_paths = []

            for image_file in image_files:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{image_file.filename.split('.')[-1]}")
                content = await image_file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file)
                image_paths.append(temp_file.name)

            # Step 1: Report image analysis
            analysis_task = """Analyze these medical report images with attention to:
1. Accurate medical interpretation of all values and findings
2. Marking terms needing research [[RESEARCH:TERM]] 
3. Using Tavily to find patient-friendly explanations when needed
4. Creating clear, supportive output"""

            medical_report = self.report_agent.run(
                analysis_task,
                images=image_paths
            ).content

            # Step 2: Extract research terms
            research_terms = self._extract_research_terms(medical_report)

            # Step 3: Conduct targeted research
            research_report = ""
            if research_terms:
                research_report = self.research_agent.run(
                    f"Research these medical terms: {', '.join(research_terms)}\n"
                    "Prioritize recent studies (last 3 years) with clear patient implications"
                ).content

            # Step 4: Generate final report
            final_report = self._generate_final_report(
                report=medical_report,
                research=research_report
            )

            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            return {
                "status": "success",
                "analysis": final_report,
                "research_terms": research_terms
            }

        except Exception as e:
            # Clean up any temporary files if error occurs
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            return {
                "status": "error",
                "message": str(e),
                "analysis": self._generate_error_report(e)
            }

    def _extract_research_terms(self, report: str) -> List[str]:
        """Extract research terms with deduplication"""
        terms = re.findall(r'\[\[RESEARCH:(.*?)\]\]', report)
        return list(set(terms))[:5]  # Limit to 5 most important terms

    def _generate_final_report(self, report: str, research: str) -> str:
        """Professional report assembly with safety checks"""
        # Remove research markers from patient-facing content
        clean_report = re.sub(r'\[\[RESEARCH:.*?\]\]', '', report)

        report_template = f"""
# 📋 Your Medical Report Analysis

{clean_report}

{self._format_research_section(research) if research else ""}

## Next Steps
- Discuss these results with your doctor
- Remember many findings require clinical context
- Your healthcare team can answer specific questions

Wishing you good health!
"""
        return report_template.strip()

    def _format_research_section(self, research: str) -> str:
        """Format research findings for patient understanding"""
        return f"""
## 📚 Medical Research Summary

Based on current medical literature:

{research}

Note: Always consult your physician about your specific case
"""

    def _generate_error_report(self, error: Exception) -> str:
        """User-friendly error handling"""
        return f"""
⚠️ We encountered an issue generating your report

Our team has been notified. Please try again later.

For immediate assistance:
- Contact your healthcare provider
- Call support at 1-800-EXAMPLE

Error details (for support staff):
{str(error)}
"""


# Initialize the analyzer
analyzer = MedicalReportImageAnalyzer()


@app.post("/analyze-reports/")
async def analyze_reports(files: List[UploadFile] = File(...)):
    """
    Analyze medical report images and provide a patient-friendly explanation

    Parameters:
    - files: List of medical report images (blood tests, scans, etc.)

    Returns:
    - JSON response with analysis results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    result = await analyzer.analyze(files)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return JSONResponse(content=result)


# Add these imports at the top with your other imports

# YouTube API configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def _find_medical_videos_internal(query: str, api_key: str) -> str:
    """
    Internal function to search YouTube for top 5 medical-related videos.
    Returns a string with video URLs separated by newlines or an error message.
    """
    if not query:
        return "Error: No search query provided."

    try:
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key)
        request = youtube.search().list(
            part="snippet",
            # Adding "medical" to ensure medical relevance
            q=f"{query} medical",
            type="video",
            maxResults=5,
            order="relevance"
        )
        response = request.execute()

        video_urls = []
        if response and 'items' in response:
            for item in response['items']:
                if 'videoId' in item.get('id', {}):
                    video_id = item['id']['videoId']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    video_urls.append(video_url)

        if video_urls:
            return video_urls  # Return as list instead of joined string
        else:
            return []

    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403:
            raise HTTPException(
                status_code=429, detail="YouTube API quota exceeded")
        elif e.resp.status == 400:
            raise HTTPException(
                status_code=400, detail="Invalid YouTube API request")
        else:
            raise HTTPException(
                status_code=500, detail=f"YouTube API error ({e.resp.status})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-medical-videos/")
async def search_medical_videos(query: str):
    """
    Search YouTube for medical educational videos related to a query

    Parameters:
    - query: Medical term or symptom to search for

    Returns:
    - JSON response with video URLs or error message
    """
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=500, detail="YouTube API key not configured")

    try:
        video_urls = _find_medical_videos_internal(
            query=query, api_key=YOUTUBE_API_KEY)
        return {
            "status": "success",
            "query": query,
            "videos": video_urls if isinstance(video_urls, list) else [],
            "error": None if isinstance(video_urls, list) else video_urls
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MedicineImageAnalyzer:
    def __init__(self):
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Initialize and return the AI agent"""
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", temperature=0.2),
            system_prompt="""You are a Medical Expert specializing in analyzing medications and providing health advice.  
Your role is to analyze medicine photos, determine their safety, and provide scientifically-backed alternatives.  

You simplify complex medical information into easy-to-understand advice and suggest safer home remedies when possible.  
Ensure responses are clear, empathetic, and structured in Markdown format.  

Critical Rules for Responses:  
1. Always mention tools/sources (e.g., "According to Tavily").  
2. Present all findings as direct medical advice, not research results.  
3. Hide technical processes—assume all recommendations are evidence-based without citing methods.  
4. Use simple, everyday language anyone can easily understand.  

When researching medical information:  
1. Use Tavily for medical information and home remedies  
2. Share final, actionable conclusions.  

Output Format:  
- Use only the original INSTRUCTIONS structure (no deviations).  
- Write as if giving doctor-approved advice.  
- For sources: Add a small italic note at section ends like:  
  [Supported by clinical studies and medical guidelines]  

Language Guidelines:  
- Replace medical terms with simple words:  
  "analgesic" → "pain reliever"  
  "contraindicated" → "not recommended"  
- Use conversational but professional tone  
- Break complex information into bullet points""",
            instructions="""## 💊 Medicine Analysis

### Step 1: Identify the Medicine
- Extract medicine name and details from the uploaded image
- Explain it in simple terms

### Step 2: Safety Check
- General safety considerations  
- Potential risks or side effects  
- Important warnings  

### Step 3: Home Remedies & Natural Alternatives
- Suggest evidence-based home remedies  
- Herbal alternatives (with safety notes)  
- Lifestyle changes that might help  

### Step 4: Nutritional Support
- Foods/nutrients that might help  
- Vitamins or supplements that could be alternatives  

### Step 5: Doctor Consultation Advice
- When to consult a doctor  
- Red flag symptoms to watch for  
- Questions to ask healthcare provider  

### Additional Notes
- Use Markdown formatting  
- Be clear and professional  
- Always recommend consulting with a healthcare provider""",
            tools=[TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))],
            markdown=True,
            debug_mode=True,
            show_tool_calls=False
        )

    async def analyze(self, image_file: UploadFile) -> dict:
        """Analyze medicine image with error handling"""
        try:
            # Save uploaded file to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{image_file.filename.split('.')[-1]}")
            content = await image_file.read()
            temp_file.write(content)
            temp_file.close()

            # Run analysis
            analysis = self.agent.run(
                "Analyze this medicine with safety recommendations and alternatives",
                images=[temp_file.name]
            ).content

            # Clean up temp file
            os.unlink(temp_file.name)

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            # Clean up temp file if error occurs
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            return {
                "status": "error",
                "message": str(e),
                "analysis": self._generate_error_report(e)
            }

    def _generate_error_report(self, error: Exception) -> str:
        """User-friendly error handling"""
        return f"""
⚠ We encountered an issue analyzing your medicine

Error details: {str(error)}

Please try again or contact support.
"""


# Initialize the medicine analyzer
medicine_analyzer = MedicineImageAnalyzer()


@app.post("/analyze-medicine/")
async def analyze_medicine(file: UploadFile = File(...)):
    """
    Analyze a medicine image and provide safety information and alternatives

    Parameters:
    - file: Image of medicine (pill bottle, package, etc.)

    Returns:
    - JSON response with analysis results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image")

    result = await medicine_analyzer.analyze(file)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return JSONResponse(content=result)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
