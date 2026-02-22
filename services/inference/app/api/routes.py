from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.core.analysis import analyze_video

router = APIRouter()

class VideoAnalysisRequest(BaseModel):
    match_id: str
    video_url: str
    start_time:   Optional[float] = None
    end_time:     Optional[float] = None
    language:     str = "english"
    aspect_ratio: str = "16:9"

@router.post("/analyze")
async def analyze_match(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    # Offload to background task
    background_tasks.add_task(
        analyze_video, 
        request.video_url, 
        request.match_id,
        start_time=request.start_time,
        end_time=request.end_time,
        language=request.language,
        aspect_ratio=request.aspect_ratio
    )
    return {"status": "processing", "match_id": request.match_id}
