from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.core.analysis import analyze_video

router = APIRouter()

class VideoAnalysisRequest(BaseModel):
    match_id: str
    video_url: str

@router.post("/analyze")
async def analyze_match(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    # Offload to background task
    background_tasks.add_task(analyze_video, request.video_url, request.match_id)
    return {"status": "processing", "match_id": request.match_id}
