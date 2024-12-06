from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarizer import summarize_large_text

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Content Summarization API (MVP)")

# Optional: Add CORS if you're accessing from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CHAR_LENGTH = 50000

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

@app.post("/summarize")
def generate_summary(request: SummarizationRequest):
    text = request.text.strip()
    text = "".join(ch for ch in text if ch.isprintable())

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="No valid text provided.")

    if len(text) > MAX_CHAR_LENGTH:
        text = text[:MAX_CHAR_LENGTH]

    try:
        summaries = summarize_large_text(
            text=text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return {"summaries": summaries}
    except Exception as e:
        print("Error during summarization:", e)
        raise HTTPException(status_code=500, detail="Internal server error during summarization.")
