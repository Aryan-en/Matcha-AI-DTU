import os
import logging
import asyncio
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_KOKORO_MODEL   = "hexgrad/Kokoro-82M"
_KOKORO_VOICE   = "af_sky"
_EDGE_TTS_VOICE = "en-GB-RyanNeural"
_hf_token: Optional[str] = os.getenv("HF_TOKEN")

def _kokoro_tts(text: str, output_path: str) -> bool:
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=_hf_token or "hf_anonymous")
        audio_bytes = client.text_to_speech(text, model=_KOKORO_MODEL)
        if not audio_bytes:
            return False
        raw = audio_bytes if isinstance(audio_bytes, (bytes, bytearray)) else audio_bytes.read()
        if len(raw) < 100:
            return False
        with open(output_path, "wb") as f:
            f.write(raw)
        logger.info(f"[TTS Tier-1] Kokoro-82M generated: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"Kokoro TTS failed: {e}")
        return False

_VOICE_MAP = {
    "english": "en-GB-RyanNeural",
    "spanish": "es-ES-AlvaroNeural",
    "portuguese": "pt-BR-AntonioNeural",
    "arabic": "ar-SA-HamedNeural",
}

def _edge_tts(text: str, output_path: str, language: str = "english") -> bool:
    try:
        import edge_tts
        voice = _VOICE_MAP.get(language.lower(), _EDGE_TTS_VOICE)
        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        asyncio.run(_synth())
        logger.info(f"[TTS Tier-2] edge-tts generated ({language}): {output_path}")
        return True
    except Exception as e:
        logger.warning(f"edge-tts failed: {e}")
        return False

def tts_generate(text: str, output_path: str, language: str = "english") -> bool:
    # Kokoro is currently EN-only for af_sky, so we fallback to edge-tts for other languages
    if language.lower() == "english":
        if _kokoro_tts(text, output_path):
            return True
    
    if _edge_tts(text, output_path, language):
        return True
    logger.error("All TTS backends failed")
    return False

def get_tts_available() -> bool:
    try:
        import edge_tts
        return True
    except ImportError:
        return False
