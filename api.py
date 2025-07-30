
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

app = FastAPI(
    title="Domain Name Suggestion API",
    description="AI-powered domain name generation for businesses",
    version="1.0.0"
)

class DomainRequest(BaseModel):
    business_description: str
    num_suggestions: int = 5
    max_length: int = 50

class DomainResponse(BaseModel):
    suggestions: List[str]
    business_description: str
    safety_filtered: bool

# Initialize components (would load actual model in production)
model = None
safety_filter = None

@app.post("/suggest-domains", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """Generate domain name suggestions for a business description"""
    try:
        # Input validation
        if not request.business_description.strip():
            raise HTTPException(status_code=400, detail="Business description cannot be empty")

        # Safety filtering
        if not safety_filter.filter_input(request.business_description):
            raise HTTPException(status_code=400, detail="Input contains inappropriate content")

        # Generate suggestions using correct method name
        suggestions = model.generate_suggestions(
            request.business_description,
            num_suggestions=request.num_suggestions
        )

        # Filter suggestions for safety
        filtered_suggestions = safety_filter.filter_output(suggestions)

        return DomainResponse(
            suggestions=filtered_suggestions[:request.num_suggestions],
            business_description=request.business_description,
            safety_filtered=len(filtered_suggestions) < len(suggestions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
