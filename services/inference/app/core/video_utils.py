import os
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from app.core.tts import tts_generate

logger = logging.getLogger(__name__)

# ── Goal celebration overlay asset ──────────────────────────────────────────
# Path to the green-screen goal animation video.
# Resolved relative to the workspace root (two levels up from this file).
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]          # …/Matcha-AI-DTU
GOAL_OVERLAY_PATH = str(_PROJECT_ROOT / "uploads" / "Goal_1.mp4")


def remove_greenscreen(frame: np.ndarray,
                       lower_green: np.ndarray = np.array([35, 80, 80]),
                       upper_green: np.ndarray = np.array([85, 255, 255])) -> tuple:
    """
    Remove green-screen from a BGR frame.
    Returns (rgb_frame, alpha_mask) where alpha_mask is 0-255 single-channel.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # Invert: foreground = non-green
    alpha = cv2.bitwise_not(green_mask)
    # Soften edges with a small blur to avoid harsh outlines
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return frame, alpha


def overlay_goal_animation(clip_path: str, output_path: str,
                           overlay_video: str = GOAL_OVERLAY_PATH,
                           position: str = "center",
                           scale: float = 0.40) -> bool:
    """
    Composite a green-screen goal animation on top of an existing highlight clip.

    Parameters
    ----------
    clip_path   : path to the base highlight clip (mp4)
    output_path : where to write the composited clip
    overlay_video: path to the goal animation with green background
    position    : "center" | "top-right" | "bottom-center"
    scale       : overlay size relative to clip width (0.0-1.0)

    Returns True on success.
    """
    if not os.path.exists(overlay_video):
        logger.warning(f"Goal overlay not found at {overlay_video} – skipping")
        return False

    base_cap = cv2.VideoCapture(clip_path)
    ovl_cap  = cv2.VideoCapture(overlay_video)

    if not base_cap.isOpened() or not ovl_cap.isOpened():
        logger.error("Failed to open base or overlay video")
        return False

    fps    = base_cap.get(cv2.CAP_PROP_FPS) or 30.0
    bw     = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bh     = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (bw, bh))

    ovl_total = int(ovl_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ovl_fps   = ovl_cap.get(cv2.CAP_PROP_FPS) or 30.0
    ovl_dur   = ovl_total / ovl_fps if ovl_fps else 0

    # Pre-calculate overlay dimensions
    ow = int(bw * scale)
    # We'll compute oh per-frame to preserve aspect ratio

    frame_idx = 0
    while True:
        ret, base_frame = base_cap.read()
        if not ret:
            break

        # Read overlay frame (loop if overlay is shorter)
        ret_o, ovl_frame = ovl_cap.read()
        if not ret_o:
            # If overlay ended, just write base frames as-is
            writer.write(base_frame)
            frame_idx += 1
            continue

        # ── Chroma-key removal ──────────────────────────────────────────
        ovl_bgr, alpha = remove_greenscreen(ovl_frame)

        # Resize overlay to target width, keep aspect ratio
        orig_h, orig_w = ovl_bgr.shape[:2]
        oh = int(ow * orig_h / orig_w) if orig_w else ow
        ovl_resized = cv2.resize(ovl_bgr, (ow, oh), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (ow, oh), interpolation=cv2.INTER_AREA)

        # ── Position calculation ────────────────────────────────────────
        if position == "top-right":
            x_off = bw - ow - 20
            y_off = 20
        elif position == "bottom-center":
            x_off = (bw - ow) // 2
            y_off = bh - oh - 20
        else:  # center
            x_off = (bw - ow) // 2
            y_off = (bh - oh) // 2

        # Clamp to frame bounds
        x_off = max(0, min(x_off, bw - ow))
        y_off = max(0, min(y_off, bh - oh))

        # ── Alpha-blend overlay onto base frame ─────────────────────────
        roi = base_frame[y_off:y_off + oh, x_off:x_off + ow]
        alpha_f = alpha_resized.astype(np.float32) / 255.0
        if len(alpha_f.shape) == 2:
            alpha_f = alpha_f[:, :, np.newaxis]  # (H, W, 1)

        blended = (ovl_resized.astype(np.float32) * alpha_f +
                   roi.astype(np.float32) * (1.0 - alpha_f))
        base_frame[y_off:y_off + oh, x_off:x_off + ow] = blended.astype(np.uint8)

        writer.write(base_frame)
        frame_idx += 1

    base_cap.release()
    ovl_cap.release()
    writer.release()

    # Re-encode with libx264 so the result is compatible with later ffmpeg stages
    re_encoded = output_path.replace(".mp4", "_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-an", re_encoded
    ]
    if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
        os.replace(re_encoded, output_path)
    else:
        logger.warning("H264 re-encode failed; keeping mp4v output")
        if os.path.exists(re_encoded):
            os.remove(re_encoded)

    logger.info(f"Goal overlay composited → {output_path} ({frame_idx} frames)")
    return True

def generate_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to generate silent audio: {e}")
        return False

def create_highlight_reel(video_path: str, highlights: list, match_id: str, output_dir: str, music_dir: Path, tracking_data: Optional[list] = None, aspect_ratio: str = "16:9", language: str = "english") -> Optional[str]:
    """
    Professional highlight reel generation:
    1. Extracts clips with xfade transitions.
    2. Overlays continuous background music and crowd noise.
    3. Auto-ducks background audio during TTS commentary.
    """
    if not highlights: return None
    
    # ── Preparation ──────────────────────────────────────────────────────────
    logger.info(f"Generating professional reel for match {match_id} ({len(highlights)} highlights)")
    music_path = str(music_dir / "music.mp3")
    crowd_path = str(music_dir / "crowd.mp3")
    roar_path = str(music_dir / "roar.mp3")
    
    for p in [music_path, crowd_path, roar_path]:
        if not os.path.exists(p): generate_silent_audio(p, duration=10.0)

    # ── Phase 1: Individual Clips & Audio Extraction ────────────────────────
    # We first extract raw video clips and generate TTS audio segments
    clip_details = []
    transition_duration = 1.0  # 1 second crossfade
    
    for i, h in enumerate(highlights):
        start, end = h["startTime"], h["endTime"]
        text, event_type = h.get("commentary", ""), h.get("eventType", "Highlight")
        duration = end - start
        
        # Paths for temporary assets
        v_clip = os.path.join(output_dir, f"raw_v_{match_id}_{i}.mp4")
        a_tts = os.path.join(output_dir, f"raw_a_tts_{match_id}_{i}.wav")
        a_game = os.path.join(output_dir, f"raw_a_game_{match_id}_{i}.wav")
        
        # 1. Video extraction with Smart Zoom if 9:16
        v_filters = [f"drawtext=text='Matcha AI Broadcast - {event_type}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.6:boxborderw=5:x=w-mod(t*100\\,w+tw):y=40"]
        
        if aspect_ratio == "9:16" and tracking_data:
            # Find ball for this segment
            seg_balls = []
            for tf in tracking_data:
                if start <= tf["t"] <= end and tf["b"]:
                    # Take first ball's center
                    bx, by, bw, bh = tf["b"][0][:4]
                    seg_balls.append((bx + bw/2, by + bh/2))
            
            if seg_balls:
                # Calculate median ball X for this clip
                avg_x = np.median([b[0] for b in seg_balls])
                # We want a 9:16 crop. If input is 16:9, width is 1.77 * height.
                # Crop width should be height * (9/16).
                # Normalised crop width = (1.0/nR) * (9/16)
                cw_norm = (9/16) / (16/9) # approx 0.3125
                x_start = max(0, min(1.0 - cw_norm, avg_x - cw_norm/2))
                v_filters.append(f"crop=iw*{cw_norm}:ih:{x_start}*iw:0,scale=720:1280")
            else:
                # Fallback to center crop
                v_filters.append(f"crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=720:1280")
        elif aspect_ratio == "9:16":
            v_filters.append(f"crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=720:1280")
        
        v_filter_str = ",".join(v_filters)
        extract_v = [
            "ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path,
            "-vf", v_filter_str, "-an", "-c:v", "libx264", "-preset", "ultrafast", v_clip
        ]
        
        # 2. TTS Generation (Multi-language)
        has_tts = tts_generate(text, a_tts, language=language) if text else False
        
        # 3. Game Audio extraction
        has_game_audio = False
        extract_a_game = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path, "-vn", "-c:a", "pcm_s16le", a_game]
        if subprocess.run(extract_a_game, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            has_game_audio = True

        if subprocess.run(extract_v).returncode == 0:
            # ── Goal Overlay: composite green-screen Goal_1.mp4 on GOAL clips ──
            if event_type.upper() == "GOAL":
                overlaid = os.path.join(output_dir, f"goal_ovl_{match_id}_{i}.mp4")
                if overlay_goal_animation(v_clip, overlaid, GOAL_OVERLAY_PATH,
                                          position="center", scale=0.45):
                    os.replace(overlaid, v_clip)   # replace raw clip with composited version
                    logger.info(f"Goal overlay applied to clip {i}")

            clip_details.append({
                "video": v_clip, "tts": a_tts if has_tts else None, 
                "game_audio": a_game if has_game_audio else None,
                "duration": duration, "event_type": event_type
            })

    if not clip_details: return None

    # ── Phase 2: Build Filter Complex ────────────────────────────────────────
    # Video: Chain xfades
    # Audio: Mix continuous streams + segmented tracks
    inputs = []
    for c in clip_details: inputs.extend(["-i", c["video"]])
    for c in clip_details: 
        if c["tts"]: inputs.extend(["-i", c["tts"]])
        if c["game_audio"]: inputs.extend(["-i", c["game_audio"]])
    
    inputs.extend(["-stream_loop", "-1", "-i", music_path])
    inputs.extend(["-stream_loop", "-1", "-i", crowd_path])
    inputs.extend(["-stream_loop", "-1", "-i", roar_path])

    # Complexity Warning: Building a massive xfade chain
    filter_parts = []
    
    # Video xfade chain
    last_v = "[0:v]"
    current_offset = clip_details[0]["duration"] - transition_duration
    for i in range(1, len(clip_details)):
        next_v = f"[{i}:v]"
        out_v = f"[v{i}]"
        filter_parts.append(f"{last_v}{next_v}xfade=transition=fade:duration={transition_duration}:offset={current_offset}{out_v}")
        last_v = out_v
        current_offset += clip_details[i]["duration"] - transition_duration

    # Audio Mixing: This is complex due to varying starts. 
    # For now, we use a simpler approach of mixing global layers + ducking logic.
    total_reel_duration = current_offset + transition_duration
    
    # Final global audio mix: Music (0.1) + Crowd (0.2) + Roar (if any)
    # Plus ducking during TTS (implemented via volume filter segments)
    # For brevity in this script, we'll perform a robust amix
    aud_idx_start = len(clip_details)
    tts_inputs = []
    game_inputs = []
    curr_idx = aud_idx_start
    
    for c in clip_details:
        if c["tts"]:
            tts_inputs.append(f"[{curr_idx}:a]")
            curr_idx += 1
        if c["game_audio"]:
            game_inputs.append(f"[{curr_idx}:a]")
            curr_idx += 1
            
    music_idx, crowd_idx, roar_idx = curr_idx, curr_idx + 1, curr_idx + 2
    
    # Construct final filter
    # v_out is last_v
    filter_parts.append(f"[{music_idx}:a]volume=0.1[bg_m];")
    filter_parts.append(f"[{crowd_idx}:a]volume=0.2[bg_c];")
    filter_parts.append(f"[{roar_idx}:a]volume=0.4[bg_r];")
    
    # Combine bg tracks and duck them if TTS is playing (simplified as a mix for now)
    filter_parts.append(f"[bg_m][bg_c][bg_r]amix=inputs=3[ambient];")
    
    # Mix TTS and Game Audio with offsets (adhoc concatenation/mix)
    # NOTE: In a perfect world, we'd use 'adelay' for each segment. 
    # For this implementation, we'll use a reliable concat/mix strategy for the segments.
    
    final_reel = os.path.join(output_dir, f"highlight_reel_{match_id}.mp4")
    
    # Execution: Due to FFmpeg filter limits, we'll use a more robust multi-pass for the actual mix
    # Pass 1: Video Transitions
    v_output = os.path.join(output_dir, f"v_trans_{match_id}.mp4")
    v_cmd = ["ffmpeg", "-y"]
    for c in clip_details: v_cmd.extend(["-i", c["video"]])
    v_cmd.extend(["-filter_complex", ";".join(filter_parts[:len(clip_details)-1]), "-map", last_v, "-c:v", "libx264", "-preset", "ultrafast", v_output])
    
    if subprocess.run(v_cmd).returncode == 0:

        # Pass 2: Merge with all audio layers
        # Inputs: 0:v_trans, 1:music, 2:crowd, 3:roar, 4..:tts, ..:game_audio
        final_cmd = ["ffmpeg", "-y", "-i", v_output]
        bg_inputs = [music_path, crowd_path, roar_path]
        for p in bg_inputs: final_cmd.extend(["-stream_loop", "-1", "-i", p])
        
        # Collect all tts and game audio segments
        seg_inputs = []
        for c in clip_details:
            if c["tts"]: seg_inputs.append(c["tts"])
            if c["game_audio"]: seg_inputs.append(c["game_audio"])
        
        for p in seg_inputs: final_cmd.extend(["-i", p])
        
        # Audio filter logic with enhanced mixing
        # Music @ 0.08, Crowd @ 0.35, Roar @ 0.25 (ducked during TTS)
        filter_complex = []
        
        # Background Mix (Ambient) - Both music and crowd clearly audible
        filter_complex.append("[1:a]volume=0.08[m]")     # Music (subtle)
        filter_complex.append("[2:a]volume=0.35[c]")     # Crowd (prominent, as requested)
        filter_complex.append("[3:a]volume=0.25[r]")     # Roar (for goal moments)
        filter_complex.append("[m][c][r]amix=inputs=3:duration=first[base_bg]")
        
        # Segments (TTS & Game Audio) with adelay
        # Calculation: i-th clip starts at sum of previous (durations - 1)
        start_offsets = [0.0]
        for i in range(len(clip_details) - 1):
            start_offsets.append(start_offsets[-1] + clip_details[i]["duration"] - transition_duration)
            
        seg_idx = 4
        mixed_seg_labels = []
        
        # Simple volume ducking for background: 
        # We'll build a volume filter string that ducks during TTS segments
        duck_volumes = []
        
        for i, c in enumerate(clip_details):
            offset_ms = int(start_offsets[i] * 1000)
            
            # Game Audio
            if c["game_audio"]:
                lbl = f"[ga{i}]"
                filter_complex.append(f"[{seg_idx}:a]adelay={offset_ms}|{offset_ms},volume=0.8{lbl}")
                mixed_seg_labels.append(lbl)
                seg_idx += 1
                
            # TTS (with ducking trigger)
            if c["tts"]:
                lbl = f"[tts{i}]"
                filter_complex.append(f"[{seg_idx}:a]adelay={offset_ms}|{offset_ms},volume=1.5{lbl}")
                mixed_seg_labels.append(lbl)
                # Duck background: volume=0.3 from target_start to target_end
                t_start = start_offsets[i]
                t_end = t_start + 5.0 # Assume TTS/Event focus is ~5s
                duck_volumes.append(f"between(t,{t_start},{t_end})")
                seg_idx += 1
        
        # Final Segment Mix
        filter_complex.append(f"{''.join(mixed_seg_labels)}amix=inputs={len(mixed_seg_labels)}:dropout_transition=0[segs]")
        
        # Apply ducking to base_bg with smooth fadeout during TTS
        if duck_volumes:
            # Create smooth ducking: volume fades from 1.0 to 0.25 during TTS, back to 1.0
            duck_expr = "+".join(duck_volumes)
            filter_complex.append(f"[base_bg]volume='if({duck_expr},0.25,1.0)':eval=frame[bg_ducked]")
        else:
            filter_complex.append("[base_bg]copy[bg_ducked]")
            
        # Mix everything: background + segments (both have equal importance)
        filter_complex.append("[bg_ducked][segs]amix=inputs=2:duration=first[final_a]")
        
        final_cmd.extend([
            "-filter_complex", ";".join(filter_complex),
            "-map", "0:v", "-map", "[final_a]",
            "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            final_reel
        ])
        
        subprocess.run(final_cmd)
        
        # Cleanup
        for c in clip_details:
            for p in [c["video"], c["tts"], c["game_audio"]]:
                if p and os.path.exists(p): os.remove(p)
        if os.path.exists(v_output): os.remove(v_output)
        
        return f"/uploads/highlight_reel_{match_id}.mp4"
    
    return None

def precompress_video(video_path: str, match_id: str, config: dict) -> str:
    try:
        if os.path.getsize(video_path) / (1024 * 1024) < config["COMPRESS_SIZE_THRESHOLD_MB"]:
            return video_path
        
        compressed_path = str(Path(video_path).parent / f"compressed_{match_id}.mp4")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"scale=-2:{config['COMPRESS_OUTPUT_HEIGHT']},fps={config['COMPRESS_OUTPUT_FPS']}", "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", "-an", compressed_path]
        
        if subprocess.run(cmd, timeout=config["FFMPEG_COMPRESS_TIMEOUT"]).returncode == 0:
            return compressed_path
    except: pass
    return video_path
