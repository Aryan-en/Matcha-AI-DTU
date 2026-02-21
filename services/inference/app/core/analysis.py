import cv2
import logging
import requests
import os
import numpy as np
import torch

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Base paths (work for both Docker and native Windows) ─────────────────────
# When running in Docker: /app is the workdir
# When running natively: services/inference is the workdir
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # services/inference/
UPLOADS_DIR = BASE_DIR.parent.parent / "uploads"  # workspace/uploads/
MUSIC_DIR = BASE_DIR / "app" / "music"  # services/inference/app/music/

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "VIDEO_SAMPLE_FPS": 1.0,
    "VIDEO_PROCESS_FPS": 1.0,
    "MOTION_WINDOW_SECS": 5.0,
    "TARGET_FRAME_HEIGHT": 480,
    "MAX_FRAME_WIDTH": 800,
    "YOLO_DOWNSCALE_HEIGHT": 480,
    "MOTION_PEAK_THRESHOLD": 0.45,
    "MOTION_FALLBACK_THRESHOLD": 0.55,
    "MOTION_MIN_GAP_SECS": 20.0,
    "MOTION_FALLBACK_MIN_GAP": 20.0,
    "CANDIDATE_MIN_MOTION": 0.65,
    "MAX_MOTION_BASED_EVENTS": 8,
    "HIGHLIGHT_CLIP_DURATION": 30.0,
    "HIGHLIGHT_CLIP_PRE_PCT": 0.35,
    "HIGHLIGHT_CLIP_POST_PCT": 0.65,
    "HIGHLIGHT_COUNT": 5,
    "HIGHLIGHT_MIN_SPREAD_PCT": 0.15,
    "AUDIO_PEAK_PERCENTILE": 90,
    "AUDIO_MOTION_DIVISOR": 40.0,
    "VISION_FAILURE_THRESHOLD": 5,
    "YOLO_SKIP_MOTION_THRESHOLD": 0.15,
    "YOLO_SKIP_MOTION_INTERVAL": 3,
    "LATE_GAME_PCT": 0.85,
    "EARLY_GAME_PCT": 0.08,
    "FRAME_STEP_FOR_1FPS": 30,
    "COMPRESS_SIZE_THRESHOLD_MB": 100,
    "COMPRESS_OUTPUT_HEIGHT": 480,
    "COMPRESS_OUTPUT_FPS": 1,
    "FFMPEG_CLIP_TIMEOUT": 120,
    "FFMPEG_CONCAT_TIMEOUT": 60,
    "FFMPEG_COMPRESS_TIMEOUT": 300,
    "MAX_SUMMARY_CHARS": 5000,
    "MAX_COMMENTARY_CHARS": 1000,
    "MAX_HIGHLIGHT_COMMENTARY_CHARS": 500,
    "MAX_EVENTTYPE_CHARS": 50,
    # ── Goal Detection Parameters ─────────────────────────────────────────
    "GOAL_DETECTION_ENABLED": True,
    "GOAL_DETECTION_MIN_FRAMES": 3,  # Frames ball must be in goal area
    "GOAL_DETECTION_MIN_SIZE": 4,  # Min ball bbox size in pixels
    "GOAL_DETECTION_MAX_SIZE": 200,  # Max ball bbox size in pixels
    "GOAL_DETECTION_CONFIDENCE_THRESHOLD": 0.35,  # Goal confidence threshold
}

# ── Goal Detection (vision-based goal line crossing) ──────────────────────────
try:
    from app.core.goal_detection import GoalDetectionEngine
    GOAL_DETECTION_AVAILABLE = True
    logger.info("Goal Detection engine loaded ✓")
except ImportError as e:
    logger.warning(f"Goal Detection not available: {e}")
    GOAL_DETECTION_AVAILABLE = False
    GoalDetectionEngine = None

# ── SoccerNet (football-specific event detection) ────────────────────────────
try:
    from app.core.soccernet_detector import detect_football_events
    SOCCERNET_AVAILABLE = True
    logger.info("SoccerNet detector loaded ✓")
except ImportError as e:
    logger.warning(f"SoccerNet detector not available: {e}")
    SOCCERNET_AVAILABLE = False
    detect_football_events = None

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    # PyTorch 2.6+ requires adding safe globals for model loading
    if hasattr(torch.serialization, "add_safe_globals"):
        import torch.nn.modules.container
        import torch.nn.modules.conv
        import torch.nn.modules.batchnorm
        import torch.nn.modules.activation
        import torch.nn.modules.pooling
        import torch.nn.modules.upsampling
        torch.serialization.add_safe_globals([
            DetectionModel,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.upsampling.Upsample,
        ])
except Exception as e:
    logger.warning(f"Could not add safe globals: {e}")
    from ultralytics import YOLO

# ── Gemini (used only for NLP: commentary text generation & match summary) ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set — Gemini features will be unavailable")
_gemini_model = None
_gemini_vision_model = None


def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured")
            _gemini_model = False
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini 2.0 Flash loaded ✓")
        except Exception as e:
            logger.warning(f"Gemini unavailable: {e}")
            _gemini_model = False
    return _gemini_model if _gemini_model else None


def _get_gemini_vision():
    """Get Gemini model configured for vision tasks."""
    global _gemini_vision_model
    if _gemini_vision_model is None:
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured")
            _gemini_vision_model = False
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_vision_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini Vision loaded ✓")
        except Exception as e:
            logger.warning(f"Gemini Vision unavailable: {e}")
            _gemini_vision_model = False
    return _gemini_vision_model if _gemini_vision_model else None


def _frame_to_pil(frame):
    """Convert OpenCV BGR frame to PIL RGB Image."""
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def analyze_frame_with_vision(frame, timestamp: float, context: str = "") -> dict:
    """
    Use Gemini Vision to analyze a video frame and detect football events.
    
    Returns: {"event_type": "GOAL"|"SAVE"|"TACKLE"|"FOUL"|"CELEBRATION"|"NONE", 
              "confidence": 0.0-1.0, "description": str}
    """
    gemini = _get_gemini_vision()
    if not gemini:
        return {"event_type": "NONE", "confidence": 0.0, "description": "Vision AI unavailable"}
    
    try:
        pil_image = _frame_to_pil(frame)
        
        prompt = """Analyze this football/soccer video frame (Note: this may be a wide-angle tactical camera view, so players and the ball may appear very small). Determine if a significant game event is happening.

IMPORTANT: Be STRICT. Only classify as an event if you're confident it's actually happening in this frame.

Event types to look for:
- GOAL: Ball clearly entering/in the goal net, or immediate celebration after scoring
- SAVE: Goalkeeper making a save, diving, catching or deflecting the ball
- TACKLE: Clear physical challenge between players for the ball
- FOUL: Player being fouled, falling unnaturally, referee intervention
- CELEBRATION: Players clearly celebrating (arms raised, hugging, jumping)
- NONE: Regular gameplay, nothing significant, unclear, or can't determine

Respond in this exact format (one line):
EVENT_TYPE|CONFIDENCE|DESCRIPTION

Where:
- EVENT_TYPE is one of: GOAL, SAVE, TACKLE, FOUL, CELEBRATION, NONE
- CONFIDENCE is a number from 0.0 to 1.0 (be conservative - use < 0.5 if unsure)
- DESCRIPTION is a brief 5-10 word description of what you see

Examples:
GOAL|0.9|Ball in net, goalkeeper beaten, players celebrating nearby
NONE|0.8|Regular midfield play, ball being passed
TACKLE|0.7|Defender sliding in to win the ball
"""
        
        response = gemini.generate_content([prompt, pil_image])
        text = response.text.strip()
        
        # Parse response
        parts = text.split("|")
        if len(parts) >= 3:
            event_type = parts[0].strip().upper()
            if event_type not in ["GOAL", "SAVE", "TACKLE", "FOUL", "CELEBRATION", "NONE"]:
                event_type = "NONE"
            try:
                confidence = float(parts[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.3
            description = "|".join(parts[2:]).strip()
            
            return {
                "event_type": event_type,
                "confidence": confidence,
                "description": description
            }
        else:
            logger.warning(f"Vision AI returned unexpected format: {text}")
            return {"event_type": "NONE", "confidence": 0.0, "description": text[:100]}
            
    except Exception as e:
        logger.warning(f"Vision analysis failed: {e}")
        return {"event_type": "NONE", "confidence": 0.0, "description": str(e)[:50]}


def validate_candidate_moment(cap, timestamp: float, fps: float, duration: float) -> dict:
    """
    Validate a candidate moment by analyzing multiple frames around it.
    Uses majority voting across frames for reliable event classification.
    
    Returns: {"event_type": str, "confidence": float, "timestamp": float, "description": str}
             or None if no significant event detected.
    """
    # Sample 3 frames: 0.5s before, at timestamp, and 0.5s after
    offsets = [-0.5, 0.0, 0.5]
    results = []
    
    for offset in offsets:
        t = timestamp + offset
        if t < 0 or t > duration:
            continue
            
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Downscale for faster API calls
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        
        result = analyze_frame_with_vision(frame, t)
        if result["event_type"] != "NONE" and result["confidence"] >= 0.5:
            results.append(result)
    
    if not results:
        return None
    
    # Majority voting: pick the most common event type
    from collections import Counter
    event_counts = Counter(r["event_type"] for r in results)
    most_common_event, count = event_counts.most_common(1)[0]
    
    # Require at least 2 frames to agree for non-GOAL events
    # Goals are dramatic enough that 1 confident detection is enough
    if most_common_event != "GOAL" and count < 2:
        return None
    
    # Average confidence of matching results
    matching = [r for r in results if r["event_type"] == most_common_event]
    avg_confidence = sum(r["confidence"] for r in matching) / len(matching)
    
    # Use the best description
    best_result = max(matching, key=lambda r: r["confidence"])
    
    return {
        "event_type": most_common_event,
        "confidence": round(avg_confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": best_result["description"],
        "frame_votes": count,
    }


# Track consecutive Vision AI failures for fallback
_vision_failures = 0
_MAX_VISION_FAILURES = 5  # After this many failures, use fallback


def fallback_heuristic_event(motion_score: float, timestamp: float, duration: float) -> dict:
    """
    Fallback event detection when Vision AI is unavailable.
    Uses motion score + temporal position to generate generic highlights.
    Much less accurate but ensures some highlights are generated.
    """
    # Very high motion in key periods suggests something important
    late_game = duration > 0 and (timestamp / duration) > 0.75
    
    # Conservative thresholds - only flag extremely high motion moments
    if motion_score >= 0.7:
        event_type = "HIGHLIGHT"  # Generic highlight
        confidence = min(0.6, motion_score * 0.8)
        desc = "High action moment detected (Vision AI fallback)"
    elif motion_score >= 0.55 and late_game:
        event_type = "HIGHLIGHT"
        confidence = 0.5
        desc = "Late-game action moment (Vision AI fallback)"
    else:
        return None
    
    return {
        "event_type": event_type,
        "confidence": round(confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": desc,
    }


def validate_candidate_with_fallback(cap, timestamp: float, fps: float, duration: float, motion_score: float) -> dict:
    """
    Validate a candidate moment, with fallback to heuristics if Vision AI fails.
    """
    global _vision_failures
    
    # If too many Vision AI failures, use fallback immediately
    if _vision_failures >= _MAX_VISION_FAILURES:
        logger.warning("Vision AI unavailable - using motion-based fallback")
        return fallback_heuristic_event(motion_score, timestamp, duration)
    
    result = validate_candidate_moment(cap, timestamp, fps, duration)
    
    if result is None:
        # Check if this was due to Vision AI failure (no frames analyzed)
        # vs. genuinely no event detected
        # We can't easily distinguish, so just try the fallback for high-motion moments
        if motion_score >= 0.65:
            _vision_failures += 1
            if _vision_failures >= _MAX_VISION_FAILURES:
                logger.warning(f"Vision AI failed {_vision_failures} times - switching to fallback mode")
            return fallback_heuristic_event(motion_score, timestamp, duration)
        return None
    
    # Vision AI worked - reset failure counter
    _vision_failures = 0
    return result


def find_motion_peaks(motion_windows: list, threshold: float = None, min_gap: float = None) -> list:
    if threshold is None:
        threshold = CONFIG["MOTION_PEAK_THRESHOLD"]
    if min_gap is None:
        min_gap = CONFIG["MOTION_MIN_GAP_SECS"]
        
    if not isinstance(motion_windows, list) or not motion_windows:
        return []
    if threshold < 0 or threshold > 1:
        logger.warning(f"Invalid threshold {threshold}, using default")
        threshold = CONFIG["MOTION_PEAK_THRESHOLD"]
    
    candidates = []
    last_peak = -999
    
    for w in motion_windows:
        if not isinstance(w, dict) or "motionScore" not in w or "timestamp" not in w:
            continue
        if w["motionScore"] >= threshold:
            t = w["timestamp"]
            if t - last_peak >= min_gap:
                candidates.append(t)
                last_peak = t
    
    return candidates


# ── FFmpeg helpers ───────────────────────────────────────────────────────────
def _generate_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    """Generate silent stereo MP3 audio using FFmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logger.warning(f"FFmpeg silent audio generation failed: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to generate silent audio: {e}")
        return False


# ── edge-tts (Microsoft neural TTS — no API key, Python 3.14 compatible) ──────
_EDGE_TTS_VOICE = "en-US-AriaNeural"   # natural female sports commentator voice

def _get_tts():
    """Check edge-tts is importable."""
    try:
        import edge_tts  # type: ignore   # noqa: F401
        return True
    except Exception as e:
        logger.error(f"edge-tts not available: {e}")
        return False


def tts_generate(text: str, output_path: str) -> bool:
    """Generate TTS audio using Microsoft edge-tts neural voice."""
    if not _get_tts():
        return False
    try:
        import asyncio
        import edge_tts  # type: ignore

        async def _synth():
            communicate = edge_tts.Communicate(text, _EDGE_TTS_VOICE)
            await communicate.save(output_path)

        asyncio.run(_synth())
        logger.info(f"edge-tts generated: {output_path}")
        return True
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return False

import subprocess
import tempfile

def create_highlight_reel(video_path: str, highlights: list, match_id: str, output_dir: str) -> str:
    """Generate TTS audio, mix with music/crowd, add overlays, and concatenate into one reel."""
    temp_clips = []
    
    music_path = str(MUSIC_DIR / "music.mp3")
    crowd_path = str(MUSIC_DIR / "crowd.mp3")
    
    # Ensure dummy audio files exist if not mounted
    if not os.path.exists(music_path):
        os.makedirs(os.path.dirname(music_path), exist_ok=True)
        if not _generate_silent_audio(music_path, duration=10.0):
            logger.warning(f"Could not generate music file at {music_path}")
    if not os.path.exists(crowd_path):
        if not _generate_silent_audio(crowd_path, duration=10.0):
            logger.warning(f"Could not generate crowd audio file at {crowd_path}")

    for i, h in enumerate(highlights):
        start = h["startTime"]
        end = h["endTime"]
        text = h.get("commentary", "")
        event_type = h.get("eventType", "Highlight")
        duration = end - start
        
        clip_path = os.path.join(output_dir, f"temp_clip_{match_id}_{i}.mp4")
        audio_path = os.path.join(output_dir, f"temp_audio_{match_id}_{i}.wav")
        
        has_audio = False
        if text:
            has_audio = tts_generate(text, audio_path)

        # Build complex filter
        # Video: drawtext scrolling right to left, fade in, fade out
        v_filter = f"[0:v]drawtext=text='Matcha AI Broadcast - {event_type}':fontcolor=white:fontsize=32:box=1:boxcolor=black@0.6:boxborderw=5:x=w-mod(t*150\\,w+tw):y=40,fade=t=in:st=0:d=1,fade=t=out:st={duration-1}:d=1[v];"
        
        # Check if video has audio (skip if ffprobe unavailable)
        has_video_audio = False
        try:
            # Validate ffprobe is available
            subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", video_path]
            probe_out = subprocess.check_output(probe_cmd, timeout=5).decode("utf-8").strip()
            has_video_audio = "audio" in probe_out
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("ffprobe unavailable, assuming no audio")
        except Exception as e:
            logger.debug(f"Audio detection failed: {e}")

        # Build ffmpeg command inputs
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end), "-i", video_path
        ]
        
        input_idx = 1
        audio_inputs = []
        a_filter = ""
        
        if has_video_audio:
            a_filter += f"[0:a]volume=0.3[a0];"
            audio_inputs.append("[a0]")
        else:
            # Add silent audio input
            cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}"])
            a_filter += f"[{input_idx}:a]volume=0.0[a0];"
            audio_inputs.append("[a0]")
            input_idx += 1
            
        if has_audio:
            cmd.extend(["-i", audio_path])
            a_filter += f"[{input_idx}:a]volume=1.5[a1];"
            audio_inputs.append("[a1]")
            input_idx += 1
            
        cmd.extend(["-stream_loop", "-1", "-i", crowd_path])
        a_filter += f"[{input_idx}:a]volume=0.3[a2];"
        audio_inputs.append("[a2]")
        input_idx += 1
        
        cmd.extend(["-stream_loop", "-1", "-i", music_path])
        a_filter += f"[{input_idx}:a]volume=0.15[a3];"
        audio_inputs.append("[a3]")
        input_idx += 1
        
        # Mix all audio inputs
        inputs_str = "".join(audio_inputs)
        a_filter += f"{inputs_str}amix=inputs={len(audio_inputs)}:duration=first:normalize=0,afade=t=in:st=0:d=1,afade=t=out:st={duration-1}:d=1[a]"

        cmd.extend([
            "-filter_complex", v_filter + a_filter,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-ac", "2", "-shortest",
            clip_path
        ])
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if result.returncode == 0:
                temp_clips.append(clip_path)
            else:
                stderr_msg = result.stderr.decode('utf-8', errors='ignore')[:300]
                logger.error(f"FFmpeg clip {i} failed (code {result.returncode}): {stderr_msg}")
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg clip {i} timeout after 120s")
        except Exception as e:
            logger.error(f"FFmpeg clip {i} error: {e}")
            
        if has_audio and os.path.exists(audio_path):
            os.remove(audio_path)

    if not temp_clips:
        return None

    # Concatenate all clips
    reel_filename = f"highlight_reel_{match_id}.mp4"
    reel_path = os.path.join(output_dir, reel_filename)
    list_path = os.path.join(output_dir, f"concat_list_{match_id}.txt")
    
    with open(list_path, "w") as f:
        for clip in temp_clips:
            f.write(f"file '{clip}'\n")
            
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", reel_path
    ]
    
    try:
        result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode != 0:
            stderr_msg = result.stderr.decode('utf-8', errors='ignore')[:300]
            logger.error(f"FFmpeg concat failed (code {result.returncode}): {stderr_msg}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg concat timeout after 60s")
        return None
    except Exception as e:
        logger.error(f"FFmpeg concat error: {e}")
        return None
        
    # Cleanup temp files
    for clip in temp_clips:
        if os.path.exists(clip):
            os.remove(clip)
    if os.path.exists(list_path):
        os.remove(list_path)
        
    return f"http://localhost:4000/uploads/{reel_filename}"


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")

# Use YOLOv8s (small) instead of nano - better GPU parallelization, faster on RTX GPUs
# Downloads automatically on first run (~22MB)
model = YOLO("yolov8s.pt")

# ── COCO classes to track for visualization ──────────────────────────────────
# NOTE: YOLO is now ONLY used for ball/player tracking visualization on canvas
# Events are detected via Gemini Vision analysis of candidate moments
YOLO_TRACK_CLASSES = {"sports ball", "person"}

# Legacy mapping - kept for compatibility but NO LONGER used for event creation
YOLO_TYPE_MAP = {"sports ball": "GOAL", "person": "TACKLE"}

# Per-class minimum YOLO confidence thresholds (for tracking visualization)
MIN_CONF: dict[str, float] = {
    "sports ball": 0.30,   # small + blurry → lower threshold
    "person":      0.50,   # for player tracking
}

# ── Minimum gap between events of the same type (seconds) ────────────────────
# Used by Vision AI event detection phase
MODEL_MIN_GAP: dict[str, float] = {
    "GOAL":        5.0,
    "TACKLE":     45.0,
    "SAVE":       20.0,
    "FOUL":       15.0,
    "Celebrate":  30.0,
    "PENALTY":    10.0,
    "RED_CARD":   10.0,
    "YELLOW_CARD": 10.0,
    "CORNER":     10.0,
    "OFFSIDE":    10.0,
}
DEFAULT_MIN_GAP = 20.0

# ── Event weight table (out of 10) ───────────────────────────────────────────
EVENT_WEIGHTS = {
    "GOAL":         10.0,
    "PENALTY":       9.5,
    "RED_CARD":      9.0,
    "SAVE":          8.0,
    "YELLOW_CARD":   7.0,
    "CELEBRATION":   6.5,   # Vision AI detected celebration
    "FOUL":          6.0,
    "HIGHLIGHT":     5.5,   # Fallback generic highlight from motion
    "TACKLE":        5.0,
    "CORNER":        4.0,
    "OFFSIDE":       2.5,
}
W1, W2, W3, W4 = 0.40, 0.20, 0.25, 0.15


# ── Scoring helpers ───────────────────────────────────────────────────────────
def time_context_weight(timestamp, duration):
    """Late-game moments carry more weight."""
    if duration <= 0:
        return 0.70
    pct = timestamp / duration
    if pct > 0.92:   return 1.00   # injury time / dying minutes
    if pct > 0.85:   return 0.95   # final 10 min
    if pct > 0.70:   return 0.85   # last quarter
    if pct > 0.50:   return 0.75   # second half
    if pct > 0.45:   return 0.60   # around half-time
    return 0.65                    # first half


def compute_context_score(event_type, motion_score, timestamp, duration, confidence):
    ew    = EVENT_WEIGHTS.get(event_type, 4.0) / 10.0
    audio = min(motion_score * 1.3, 1.0)
    tw    = time_context_weight(timestamp, duration)
    base  = (ew * W1) + (audio * W2) + (motion_score * W3) + (tw * W4)
    score = base * (0.5 + 0.5 * confidence)
    if duration > 0 and (timestamp / duration) > 0.85 and event_type == "GOAL":
        score *= 2.0   # late goals doubled
    if duration > 0 and (timestamp / duration) < 0.08 and event_type in ("SAVE", "TACKLE"):
        score *= 1.3   # frantic early-game bump
    return round(min(score * 10.0, 10.0), 2)


# ── Fallback commentary ───────────────────────────────────────────────────────
_FALLBACK = {
    "GOAL":        {"high": "GOOOAL! Sensational — the crowd erupts!", "mid": "Goal! Crucial finish puts them ahead!", "low": "Goal scored."},
    "TACKLE":      {"high": "FEROCIOUS TACKLE! Incredible commitment!", "mid": "Strong challenge wins the ball back.", "low": "Tackle wins possession."},
    "FOUL":        {"high": "DEFINITE FOUL! Referee steps in immediately!", "mid": "Free kick awarded — bodies flying here.", "low": "Foul given."},
    "SAVE":        {"high": "UNBELIEVABLE SAVE! Superhuman goalkeeping!", "mid": "Good stop from the keeper — keeping them in it.", "low": "Save made."},
    "CELEBRATION": {"high": "INCREDIBLE SCENES! The players are losing their minds!", "mid": "Celebrations break out on the pitch!", "low": "The players celebrate."},
    "HIGHLIGHT":   {"high": "WHAT A MOMENT! Crucial action in this match!", "mid": "Important moment of play here.", "low": "Key moment of play."},
}


def _fallback_commentary(event_type, final_score, timestamp, duration):
    minute = max(1, int(timestamp / 60))
    late   = duration > 0 and (timestamp / duration) > 0.85
    energy = "high" if final_score >= 7.5 else ("mid" if final_score >= 5 else "low")
    text   = _FALLBACK.get(event_type, {}).get(energy, f"{event_type} at minute {minute}.")
    if "minute" not in text.lower():
        text = text.rstrip("!.") + f" at minute {minute}."
    if late and final_score >= 7:
        text = "LATE DRAMA! " + text
    return text


# ── Gemini commentary ─────────────────────────────────────────────────────────
def generate_commentary(event_type, final_score, timestamp, duration, context_events=None):
    """Single vivid commentary sentence via Gemini, fallback to template."""
    minute   = max(1, int(timestamp / 60))
    late     = duration > 0 and (timestamp / duration) > 0.85
    energy   = "HIGH INTENSITY" if final_score >= 7.5 else ("MODERATE" if final_score >= 5 else "low key")
    late_str = "in the dying minutes (CRUCIAL late-game moment!)" if late else f"at minute {minute}"
    ctx_str  = ""
    if context_events:
        near = [e for e in context_events if abs(e["timestamp"] - timestamp) < 60]
        if near:
            ctx_str = " Nearby events: " + ", ".join(e["type"] for e in near[:3]) + "."

    prompt = (
        f"You are an incredibly passionate and energetic football/soccer commentator. "
        f"Write a thrilling, continuous commentary script (around 40-60 words) describing a {event_type} "
        f"{late_str}. The intensity is {energy} (score {final_score:.1f}/10).{ctx_str} "
        f"Make it sound like a live broadcast, building up excitement, describing the build-up, the moment itself, "
        f"and the immediate aftermath. No quotes, no attribution, just the spoken words."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = gemini.generate_content(prompt)
            text = resp.text.strip().strip('"').strip("'")
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini commentary error: {e}")
    return _fallback_commentary(event_type, final_score, timestamp, duration)


# ── Gemini match summary ──────────────────────────────────────────────────────
def generate_match_summary(scored_events, highlights, duration):
    """3-5 sentence AI narrative summary of the whole match."""
    if not scored_events:
        return "No significant events were detected in this match footage."

    by_type: dict = {}
    for e in scored_events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    top5  = sorted(scored_events, key=lambda x: x["finalScore"], reverse=True)[:5]
    tdesc = "; ".join(
        f"{e['type']} at {int(e['timestamp']//60)}:{int(e['timestamp']%60):02d} (score {e['finalScore']:.1f})"
        for e in top5
    )
    stats = ", ".join(f"{v} {k.lower()}s" for k, v in by_type.items())
    dur_m = int(duration // 60)

    prompt = (
        f"You are a football analyst AI. Write a 3-5 sentence match summary for a {dur_m}-minute match.\n"
        f"Event breakdown: {stats}.\n"
        f"Top 5 moments: {tdesc}.\n"
        f"Total events: {len(scored_events)} | Highlights: {len(highlights)}.\n"
        f"Use present tense, analytical but engaging. Describe match narrative — intense phases, "
        f"key moments, overall character. Don't invent player names or exact scorelines."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = gemini.generate_content(prompt)
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini summary error: {e}")

    goals   = by_type.get("GOAL", 0)
    saves   = by_type.get("SAVE", 0)
    tackles = by_type.get("TACKLE", 0)
    fouls   = by_type.get("FOUL", 0)
    half    = "second half" if duration > 0 and top5[0]["timestamp"] / duration > 0.5 else "first half"
    return (
        f"A {dur_m}-minute match featuring {len(scored_events)} detected events. "
        f"The game produced {goals} goal(s), {saves} save(s), {tackles} tackle(s), and {fouls} foul(s). "
        f"The top moment scored {top5[0]['finalScore']:.1f}/10 — "
        f"a {top5[0]['type']} at minute {int(top5[0]['timestamp']//60)}. "
        f"Overall intensity peaked in the {half}."
    )


# ── Highlight selection ───────────────────────────────────────────────────────
def select_highlights(scored_events, duration, top_n=5, clip_secs=30.0):
    """
    Top-N non-overlapping highlights, spread across the full video.
    Two rules:
      1. No time-window overlap between clips.
      2. Clip centres must be at least 15% of duration apart (prevents same-scene
         from different YOLO frames appearing twice).
    """
    if not scored_events:
        return []
    sorted_evs = sorted(scored_events, key=lambda x: x["finalScore"], reverse=True)
    min_spread = max(30.0, duration * 0.15)   # at least 15% of the video length
    used, highlights = [], []

    for ev in sorted_evs:
        if len(highlights) >= top_n:
            break
        ts    = ev["timestamp"]
        start = max(0.0, ts - clip_secs * 0.35)
        end   = min(duration if duration > 0 else ts + 60.0, ts + clip_secs * 0.65)

        # Rule 1 – no overlap
        if any(not (end <= w[0] or start >= w[1]) for w in used):
            continue

        # Rule 2 – clips must be spread out
        if any(abs(ts - (h["startTime"] + (h["endTime"] - h["startTime"]) / 2)) < min_spread
               for h in highlights):
            continue

        used.append((start, end))
        highlights.append({
            "startTime":  round(start, 1),
            "endTime":    round(end, 1),
            "score":      ev["finalScore"],
            "eventType":  ev["type"],
            "commentary": ev.get("commentary", ""),
        })

    return sorted(highlights, key=lambda x: x["startTime"])


# ── Team colour clustering ────────────────────────────────────────────────────
def _crop_jersey(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Return the torso crop (middle 40% height, inner 60% width) of a person box."""
    h, w = frame.shape[:2]
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = int(x2 * w), int(y2 * h)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 4 or bh < 10:
        return np.array([])
    # Torso: rows 30-70%, cols 20-80%
    cy1 = by1 + int(bh * 0.30)
    cy2 = by1 + int(bh * 0.70)
    cx1 = bx1 + int(bw * 0.20)
    cx2 = bx1 + int(bw * 0.80)
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop.size else np.array([])


def _dominant_colour(crop: np.ndarray) -> list[int] | None:
    """Return [R, G, B] dominant colour of a BGR crop via median."""
    if crop is None or crop.size < 3:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    return [int(v) for v in np.median(pixels, axis=0).tolist()]


def _cluster_teams(colours: list[list[int]], n: int = 2) -> tuple[list[list[int]], list[int]]:
    """
    K-means cluster colours into n teams.
    Returns (centroids, labels) where centroids = [[R,G,B], ...]
    Uses numpy-only mini K-means (no sklearn needed).
    """
    if len(colours) < n:
        defaults = [[220, 50, 50], [50, 100, 220]]
        return defaults[:n], [i % n for i in range(len(colours))]
    data = np.array(colours, dtype=np.float32)
    # Initialise centres from extremes
    centres = data[np.random.choice(len(data), n, replace=False)]
    for _ in range(20):
        dists  = np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1)
        labels = np.argmin(dists, axis=1)
        new_c  = np.stack([
            data[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
            for k in range(n)
        ])
        if np.allclose(centres, new_c, atol=1.0):
            break
        centres = new_c
    final_labels = np.argmin(
        np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1), axis=1
    )
    return [[int(v) for v in c.tolist()] for c in centres], final_labels.tolist()


def _get_motion_at(windows, timestamp):
    if not windows:
        return 0.3
    return min(windows, key=lambda w: abs(w["timestamp"] - timestamp))["motionScore"]


def _report_failure(match_id):
    try:
        requests.post(f"{ORCHESTRATOR_URL}/matches/{match_id}/progress", json={"progress": -1}, timeout=3)
    except Exception:
        pass


def emit_live_event(match_id: str, event: dict):
    """
    POST one event immediately to the orchestrator so it can be broadcast via
    WebSocket to any frontend clients watching this match in real-time.
    Failures are silently swallowed (best-effort).
    """
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/live-event",
            json=event,
            timeout=2,
        )
    except Exception:
        pass


def _precompress_video(video_path: str, match_id: str) -> str:
    if not video_path or not isinstance(video_path, str):
        raise ValueError("Invalid video_path")
    if not match_id or not isinstance(match_id, str):
        raise ValueError("Invalid match_id")
        
    try:
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        threshold_mb = CONFIG["COMPRESS_SIZE_THRESHOLD_MB"]
        
        if file_size_mb < threshold_mb:
            return video_path
            
        logger.info(f"Pre-compressing video ({file_size_mb:.0f}MB) for faster analysis...")
        
        output_dir = Path(video_path).parent
        compressed_path = str(output_dir / f"compressed_{match_id}.mp4")
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"scale=-2:{CONFIG['COMPRESS_OUTPUT_HEIGHT']},fps={CONFIG['COMPRESS_OUTPUT_FPS']}",
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-an",
            compressed_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=CONFIG["FFMPEG_COMPRESS_TIMEOUT"])
        
        if result.returncode == 0 and os.path.exists(compressed_path):
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
            reduction_pct = 100 - (compressed_size / file_size_mb * 100) if file_size_mb > 0 else 0
            logger.info(f"Compressed: {file_size_mb:.0f}MB → {compressed_size:.0f}MB ({reduction_pct:.0f}% reduction)")
            return compressed_path
        else:
            logger.warning(f"FFmpeg compression failed, using original")
            return video_path
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Pre-compression timeout, using original")
        return video_path
    except Exception as e:
        logger.warning(f"Pre-compression failed: {e}")
        return video_path


# ── Goal Detection Pipeline ────────────────────────────────────────────────────
def detect_goals_in_video(video_path: str) -> list:
    """
    Detect goals in video using vision-based goal-line crossing detection.
    Returns list of goal events: [{"timestamp": float, "type": "GOAL", ...}]
    """
    if not GOAL_DETECTION_AVAILABLE or not GoalDetectionEngine:
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for goal detection: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize goal detector
        goal_engine = GoalDetectionEngine()
        goal_engine.init(frame_width, frame_height, fps)
        
        logger.info(f"Goal detection: {frame_width}×{frame_height} @ {fps:.1f}fps")
        
        goals_detected = []
        frame_idx = 0
        
        # Process every Nth frame to speed up (goal detection doesn't need every frame)
        frame_step = max(1, int(fps / 5.0))  # Process at ~5 FPS
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames
            if frame_idx % frame_step != 0:
                continue
            
            # Resize for faster processing if very large
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))
            
            # Process frame
            goal_event = goal_engine.process_frame(frame)
            
            if goal_event:
                goals_detected.append({
                    "timestamp": round(goal_event.timestamp, 2),
                    "type": "GOAL",
                    "confidence": round(goal_event.confidence, 3),
                    "description": f"Goal detected ({goal_event.direction})",
                    "source": "goal_detection"
                })
                logger.info(f"⚽ GOAL at {goal_event.timestamp:.1f}s | confidence: {goal_event.confidence:.2f}")
        
        cap.release()
        
        logger.info(f"Goal detection completed: {len(goals_detected)} goals found")
        return goals_detected
        
    except Exception as e:
        logger.error(f"Goal detection failed: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_video(video_path: str, match_id: str):
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        _report_failure(match_id)
        return {"error": "Video file not found"}
    
    if not match_id or not isinstance(match_id, str):
        logger.error("Invalid match_id")
        return {"error": "Invalid match_id"}
        
    logger.info(f"Starting analysis: match={match_id}")
    
    # Pre-compress large videos for faster processing
    original_video_path = video_path
    video_path = _precompress_video(video_path, match_id)
    compressed = (video_path != original_video_path)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            _report_failure(match_id)
            return {"error": "Could not open video"}

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        duration     = total_frames / fps if fps > 0 else 0.0
        
        # If we pre-compressed to 1fps, adjust settings
        if compressed:
            process_fps = fps
            frame_step = 1
        else:
            process_fps = CONFIG["VIDEO_PROCESS_FPS"]
            frame_step = max(1, int(fps / process_fps))
            
        track_interval = 1
        window_frames = max(1, int(process_fps * CONFIG["MOTION_WINDOW_SECS"]))
        target_height = CONFIG["YOLO_DOWNSCALE_HEIGHT"] if frame_h > 720 else frame_h

        logger.info(f"Video: {total_frames}f @ {fps:.1f}fps = {duration:.1f}s [{frame_w}×{frame_h}] → {target_height}p @ {process_fps}fps")

        frame_count  = 0
        processed_count = 0
        prev_gray    = None
        window_diffs = []

        motion_windows: list = []
        track_frames:   list = []
        jersey_colours: list = []   # [[R,G,B], ...] one per sampled person crop
        frame_person_rows: list = []  # parallel to track_frames, raw persons before team assignment

        # ── GoalDetectionEngine (Kalman + Homography + FSM) ───────────────────────
        _goal_engine = None
        if GOAL_DETECTION_AVAILABLE and GoalDetectionEngine:
            try:
                _goal_engine = GoalDetectionEngine()
                _goal_engine.init(frame_w, frame_h, fps)
                logger.info("GoalDetectionEngine initialised ✓")
            except Exception as _ge:
                logger.warning(f"GoalDetectionEngine init failed: {_ge}")
                _goal_engine = None

        # NOTE: raw_events and last_seen are no longer used here
        # Events are now detected via Vision AI after the main loop

        while cap.isOpened():
            # Fast-forward: skip decoding frames we don't need
            for _ in range(frame_step - 1):
                cap.grab()
                frame_count += 1
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            processed_count += 1
            timestamp = frame_count / fps

            # Downscale frame if it's very large (e.g., 1080p/4K) to speed up YOLO and motion diff
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                # Update frame_w and frame_h for normalized coordinates
                frame_w, frame_h = 800, int(h * scale)

            # ── Motion window ────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                window_diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray

            if len(window_diffs) >= window_frames:
                raw_max = float(np.percentile(window_diffs, 90)) if len(window_diffs) > 1 else window_diffs[0]
                m_score = round(min(raw_max / 40.0, 1.0), 3)
                motion_windows.append({
                    "timestamp":   round(timestamp - 5.0, 1),
                    "motionScore": m_score,
                    "audioScore":  round(min(m_score * 1.2, 1.0), 3),
                })
                window_diffs = []

            # ── Progress update ──────────────────────────────────────────────
            if processed_count % 5 == 0 and total_frames > 0:
                try:
                    requests.post(
                        f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                        json={"progress": int((frame_count / total_frames) * 99)},
                        timeout=1,
                    )
                except Exception:
                    pass

            # ── YOLO detection & tracking ────────────────────────────────────
            current_motion = _get_motion_at(motion_windows, timestamp) if motion_windows else 0.3
            
            # OPTIMIZATION: Skip YOLO on very static scenes
            skip_yolo = current_motion < CONFIG["YOLO_SKIP_MOTION_THRESHOLD"] and processed_count % CONFIG["YOLO_SKIP_MOTION_INTERVAL"] != 0
            
            frame_balls:   list = []
            frame_persons: list = []
            
            if not skip_yolo:
                # Downscale frame for faster YOLO inference
                yolo_frame = frame
                if frame.shape[0] > target_height:
                    scale = target_height / frame.shape[0]
                    yolo_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                try:
                    results = model(yolo_frame, verbose=False)
                except Exception as e:
                    logger.warning(f"YOLO inference failed: {e}")
                    results = []

                # Scale factor for converting YOLO coords back to original frame size
                scale_factor = frame.shape[0] / yolo_frame.shape[0] if frame.shape[0] != yolo_frame.shape[0] else 1.0

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls   = int(box.cls[0])
                        conf  = float(box.conf[0])
                        label = model.names[cls]

                        # Only track sports ball and person for visualization
                        if label not in YOLO_TRACK_CLASSES:
                            continue

                        x1, y1, x2, y2 = [v * scale_factor for v in box.xyxy[0].tolist()]

                        # Store normalised coords for canvas overlay
                        nx = round(x1 / frame_w, 4)
                        ny = round(y1 / frame_h, 4)
                        nw = round((x2 - x1) / frame_w, 4)
                        nh = round((y2 - y1) / frame_h, 4)

                        tid = -1
                        if hasattr(box, "id") and box.id is not None:
                            tid = int(box.id[0])

                        if label == "sports ball":
                            frame_balls.append([nx, ny, nw, nh, round(conf, 3)])
                        elif label == "person":
                            # Extract jersey colour → stored temporarily as [nx,ny,nw,nh,tid, r,g,b]
                            # After the loop we replace r,g,b with team index
                            crop = _crop_jersey(frame, nx, ny, nx + nw, ny + nh)
                            col  = _dominant_colour(crop) or [128, 128, 128]
                            jersey_colours.append(col)
                            frame_persons.append([nx, ny, nw, nh, tid, col[0], col[1], col[2]])
                        
                        # NOTE: We no longer create events from YOLO detections
                        # Events are now detected using Gemini Vision analysis
                        # YOLO is only used for ball/player tracking visualization

            # ── GoalDetectionEngine per-frame ────────────────────────────────────
            if _goal_engine is not None:
                try:
                    _goal_engine.process_frame(frame)
                except Exception as _gfe:
                    logger.debug(f"goal_engine.process_frame error: {_gfe}")

            # Store tracking frame on every detection tick (max every track_interval)
            if (processed_count % track_interval == 0) and (frame_balls or frame_persons):
                track_frames.append({
                    "t": round(timestamp, 2),
                    "b": frame_balls[:4],      # ≤4 balls (sports balls)
                    "p": frame_persons[:25],   # ≤25 players (full pitch)
                })

        cap.release()

        # Flush remaining motion window
        if window_diffs:
            raw_max = float(np.percentile(window_diffs, 90)) if len(window_diffs) > 1 else window_diffs[0]
            m_score = round(min(raw_max / 40.0, 1.0), 3)
            motion_windows.append({
                "timestamp":   round((frame_count - len(window_diffs)) / fps, 1),
                "motionScore": m_score,
                "audioScore":  round(min(m_score * 1.2, 1.0), 3),
            })

        # ── Team colour clustering ────────────────────────────────────────────
        # Cluster all jersey colours into 2 teams, then replace the temporary
        # [r,g,b] stored per person with its team index {0, 1}.
        team_colors = [[220, 60, 60], [60, 100, 220]]   # fallback: red / blue
        if len(jersey_colours) >= 4:
            try:
                centroids, _ = _cluster_teams(jersey_colours, n=2)
                team_colors  = centroids
                logger.info(f"Team colours detected: {team_colors}")
            except Exception as e:
                logger.warning(f"Team clustering failed: {e}")

        def _assign_team(r: int, g: int, b: int) -> int:
            """Return 0 or 1 — whichever centroid [r,g,b] is closest to."""
            col = np.array([r, g, b], dtype=float)
            dists = [np.linalg.norm(col - np.array(c)) for c in team_colors]
            return int(np.argmin(dists))

        # Replace [nx,ny,nw,nh,tid, r,g,b] → [nx,ny,nw,nh,tid, team]
        for tf in track_frames:
            labelled = []
            for p in tf.get("p", []):
                if len(p) == 8:   # has colour channels
                    team = _assign_team(int(p[5]), int(p[6]), int(p[7]))
                    labelled.append([p[0], p[1], p[2], p[3], p[4], team])
                elif len(p) >= 5:
                    labelled.append(list(p[:5]) + [0])
                else:
                    labelled.append(p)
            tf["p"] = labelled

        # ══════════════════════════════════════════════════════════════════════
        # ██ SOCCERNET EVENT DETECTION (football-specific trained model) ██
        # ══════════════════════════════════════════════════════════════════════
        logger.info("Phase 2: SoccerNet football event detection...")
        
        raw_events = []
        
        # Add goals from GoalDetectionEngine (Kalman + Homography + FSM)
        if _goal_engine is not None:
            from app.core.goal_detection import goal_events_to_raw
            _raw_goals = goal_events_to_raw(_goal_engine.goals)
            if _raw_goals:
                logger.info(f"GoalDetectionEngine found {len(_raw_goals)} goal(s)")
                raw_events.extend(_raw_goals)
        
        # Primary: Use SoccerNet (trained on football footage)
        if SOCCERNET_AVAILABLE and detect_football_events:
            try:
                logger.info("Running SoccerNet analysis on original video...")
                soccernet_events = detect_football_events(original_video_path, sensitivity=1.0)
                
                if soccernet_events:
                    for ev in soccernet_events:
                        raw_events.append({
                            "timestamp": ev["timestamp"],
                            "type": ev["type"],
                            "confidence": ev["confidence"],
                            "description": f"SoccerNet detected {ev['type'].lower()}",
                            "source": "soccernet"
                        })
                    logger.info(f"SoccerNet detected {len(raw_events)} events")
                else:
                    logger.warning("SoccerNet returned no events, falling back to motion analysis")
                    
            except Exception as e:
                logger.error(f"SoccerNet analysis failed: {e}")
        
        # Fallback: Motion-based highlights if SoccerNet fails or returns too few
        if len(raw_events) < 3:
            logger.info("Supplementing with motion-based highlight detection...")
            
            # Find high-motion peaks as generic highlights
            candidate_timestamps = find_motion_peaks(
                motion_windows, 
                threshold=CONFIG["MOTION_FALLBACK_THRESHOLD"],
                min_gap=CONFIG["MOTION_FALLBACK_MIN_GAP"]
            )
            
            # Filter out timestamps that are too close to existing events
            existing_times = {ev["timestamp"] for ev in raw_events}
            for candidate_t in candidate_timestamps:
                # Skip if within 15s of an existing event
                if any(abs(candidate_t - et) < 15 for et in existing_times):
                    continue
                    
                motion_score = _get_motion_at(motion_windows, candidate_t) if motion_windows else 0.5
                
                # Only add very high motion moments as generic highlights
                # Validate the motion peak with Gemini Vision AI instead of blindly accepting it
                # Tactical cameras cause massive motion spikes during pans. Vision AI will reject these as "NONE".
                val_result = validate_candidate_with_fallback(cap, candidate_t, fps, duration, motion_score)
                if val_result and val_result["event_type"] != "NONE":
                    raw_events.append({
                        "timestamp": val_result["timestamp"],
                        "type": val_result["event_type"],
                        "confidence": val_result["confidence"],
                        "description": val_result["description"],
                        "source": "vision_ai_fallback"
                    })
                    existing_times.add(candidate_t)
                    
                    if len(raw_events) >= CONFIG["MAX_MOTION_BASED_EVENTS"]:  # Cap total events
                        break
                        
            logger.info(f"Total events after fallback: {len(raw_events)}")
        
        # Sort by timestamp
        raw_events.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Final event detection: {len(raw_events)} events")
        
        # ── Score events & emit live ─────────────────────────────────────────
        scored_events: list = []
        for ev in raw_events:
            m_score   = _get_motion_at(motion_windows, ev["timestamp"])
            fs        = compute_context_score(ev["type"], m_score, ev["timestamp"], duration, ev["confidence"])
            scored_ev = {**ev, "finalScore": fs}
            scored_events.append(scored_ev)

            # Fire-and-forget: sends to NestJS → WebSocket → browser
            emit_live_event(match_id, scored_ev)

        # ── Gemini commentary per event ──────────────────────────────────────
        for i, ev in enumerate(scored_events):
            ctx = scored_events[max(0, i - 3):i] + scored_events[i + 1:i + 3]
            ev["commentary"] = generate_commentary(
                ev["type"], ev["finalScore"], ev["timestamp"], duration, ctx
            )

        # ── Highlights (spread, non-overlapping) ─────────────────────────────
        highlights = select_highlights(scored_events, duration)

        # ── Generate Highlight Video Clips with TTS ──────────────────────────
        logger.info("Generating highlight reel with TTS, music, and crowd noise...")
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        # Use original video for highlight reel (compressed is 1fps)
        highlight_reel_url = create_highlight_reel(
            video_path=original_video_path,
            highlights=highlights,
            match_id=match_id,
            output_dir=str(UPLOADS_DIR)
        )

        # ── Emotion scores ────────────────────────────────────────────────────
        emotion_scores = [
            {
                "timestamp":     w["timestamp"],
                "audioScore":    w["audioScore"],
                "motionScore":   w["motionScore"],
                "contextWeight": round(time_context_weight(w["timestamp"], duration), 3),
                "finalScore":    round(
                    (w["audioScore"] * 0.3 + w["motionScore"] * 0.5 +
                     time_context_weight(w["timestamp"], duration) * 0.2) * 10, 2
                ),
            }
            for w in motion_windows
        ]

        # ── Gemini match summary ──────────────────────────────────────────────
        logger.info("Generating Gemini match summary…")
        summary = generate_match_summary(scored_events, highlights, duration)
        logger.info(f"Summary: {len(summary)} chars")

        logger.info(
            f"Done: {len(scored_events)} events | {len(highlights)} highlights | "
            f"{len(track_frames)} tracking frames | {duration:.1f}s"
        )

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        payload = {
            "events":        convert_numpy(scored_events),
            "highlights":    convert_numpy(highlights),
            "emotionScores": convert_numpy(emotion_scores),
            "duration":      round(float(duration), 1),
            "summary":       summary,
            "highlightReelUrl": highlight_reel_url,
            "trackingData":  convert_numpy(track_frames),
            "teamColors":    convert_numpy(team_colors),   # [[R,G,B],[R,G,B]] team0 / team1
        }

        try:
            resp = requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/complete",
                json=payload,
                timeout=30,
            )
            logger.info(f"Complete sent — HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to send completion: {e}")
            try:
                requests.post(
                    f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                    json={"progress": 100}, timeout=5
                )
            except Exception:
                pass

        # Cleanup compressed file if we created one
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        return {"status": "completed", "match_id": match_id}

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        # Cleanup compressed file on error too
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
        _report_failure(match_id)
        return {"error": str(e)}

