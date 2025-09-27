# transcriber.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import io
import logging
import time
import sys

from pydub import AudioSegment
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcriber.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass
class TranscribeConfig:
    audio_path: Path                        # ruta del audio a transcribir
    language: str = "es"                    # idioma del audio (no traducción)
    model: str = "large-v3"                 # tamaño del modelo faster-whisper
    device: str = "auto"                    # "auto" | "cpu" | "cuda"
    compute_type_cpu: str = "int8"          # para CPU
    compute_type_cuda: str = "int8_float16" # para GPU NVIDIA
    target_sr: int = 16000                  # 16 kHz recomendado
    mono: bool = True                       # forzar 1 canal
    vad_filter: bool = True                 # Voice Activity Detection
    beam_size: int = 5
    best_of: int = 5
    no_speech_threshold: float = 0.2
    initial_prompt: Optional[str] = (
        "Transcribe en español rioplatense (Argentina). Conserva modismos y no sobrecorrijas."
    )

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TranscribeConfig":
        logger.debug(f"Creating TranscribeConfig from dict: {cfg}")
        cfg = dict(cfg or {})
        if "audio_path" in cfg and not isinstance(cfg["audio_path"], Path):
            original_path = cfg["audio_path"]
            cfg["audio_path"] = Path(cfg["audio_path"])
            logger.debug(f"Converted audio_path from {original_path} to Path object: {cfg['audio_path']}")

        config = cls(**cfg)
        logger.info(f"Created TranscribeConfig - audio_path: {config.audio_path}, model: {config.model}, language: {config.language}, device: {config.device}")
        return config


# -----------------------------
# Utilidades de audio
# -----------------------------
def _load_to_wav_buffer(path: Path, target_sr: int, mono: bool) -> io.BytesIO:
    """
    Carga cualquier formato soportado por ffmpeg y lo normaliza a WAV PCM 16-bit en memoria.
    """
    start_time = time.time()
    logger.info(f"Loading audio file: {path}")
    logger.debug(f"Audio processing parameters - target_sr: {target_sr}, mono: {mono}")

    try:
        # Load audio file
        logger.debug("Reading audio file with AudioSegment...")
        seg = AudioSegment.from_file(path)
        original_duration = len(seg) / 1000.0  # Convert to seconds
        original_channels = seg.channels
        original_frame_rate = seg.frame_rate

        logger.info(f"Original audio properties - duration: {original_duration:.2f}s, channels: {original_channels}, frame_rate: {original_frame_rate}Hz")

        # Convert to mono if needed
        if mono and seg.channels > 1:
            logger.debug("Converting audio to mono...")
            seg = seg.set_channels(1)
            logger.debug("Audio converted to mono")

        # Resample if needed
        if seg.frame_rate != target_sr:
            logger.debug(f"Resampling audio from {seg.frame_rate}Hz to {target_sr}Hz...")
            seg = seg.set_frame_rate(target_sr)
            logger.debug("Audio resampled successfully")

        # Set sample width to 16-bit PCM
        if seg.sample_width != 2:
            logger.debug(f"Converting sample width from {seg.sample_width * 8}-bit to 16-bit...")
            seg = seg.set_sample_width(2)  # 16-bit PCM
            logger.debug("Sample width converted to 16-bit")

        # Export to WAV buffer
        logger.debug("Exporting audio to WAV buffer...")
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)

        buffer_size = len(buf.getvalue())
        load_time = time.time() - start_time

        logger.info(f"Audio loading completed - buffer_size: {buffer_size} bytes, processing_time: {load_time:.2f}s")
        logger.debug(f"Final audio properties - channels: {seg.channels}, frame_rate: {seg.frame_rate}Hz, sample_width: {seg.sample_width * 8}-bit")

        return buf

    except Exception as e:
        logger.error(f"Error loading audio file {path}: {str(e)}")
        raise


# -----------------------------
# Transcriptor
# -----------------------------
class FasterWhisperTranscriber:
    def __init__(self, config: Dict[str, Any], run_manager=None):
        init_start_time = time.time()
        logger.info("Initializing FasterWhisperTranscriber...")

        # Store run manager for tracking
        self.run_manager = run_manager

        # Create config
        self.cfg = TranscribeConfig.from_dict(config)

        # Validate audio file
        logger.debug(f"Validating audio file exists: {self.cfg.audio_path}")
        if not self.cfg.audio_path.exists():
            logger.error(f"Audio file not found: {self.cfg.audio_path}")
            raise FileNotFoundError(f"No existe el archivo de audio: {self.cfg.audio_path}")

        file_size = self.cfg.audio_path.stat().st_size
        logger.info(f"Audio file validated - path: {self.cfg.audio_path}, size: {file_size} bytes")

        # Device resolution
        logger.debug("Resolving compute device...")
        device = (self.cfg.device or "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            logger.warning(f"Invalid device '{device}' specified, falling back to 'auto'")
            device = "auto"

        if device == "auto":
            logger.debug("Auto-detecting compute device...")
            from shutil import which
            nvidia_available = which("nvidia-smi") is not None
            device = "cuda" if nvidia_available else "cpu"
            logger.debug(f"NVIDIA GPU detected: {nvidia_available}, selected device: {device}")

        compute_type = self.cfg.compute_type_cuda if device == "cuda" else self.cfg.compute_type_cpu
        logger.info(f"Compute configuration - device: {device}, compute_type: {compute_type}")

        # Model handling
        logger.debug("Setting up Whisper model...")
        model_name = self.cfg.model
        repo_id = model_name
        local_dir = Path("models") / model_name

        logger.debug(f"Model configuration - name: {model_name}, local_dir: {local_dir}")

        model_download_start = time.time()
        if not local_dir.exists():
            logger.info(f"Model {repo_id} not found locally. Downloading to {local_dir}...")
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
                download_time = time.time() - model_download_start
                logger.info(f"Model download completed in {download_time:.2f}s")
            except Exception as e:
                logger.error(f"Failed to download model {repo_id}: {str(e)}")
                raise
        else:
            logger.info(f"Using cached model from {local_dir}")

        # Initialize Whisper model
        logger.debug("Initializing WhisperModel...")
        model_init_start = time.time()
        try:
            self._model = WhisperModel(str(local_dir), device=device, compute_type=compute_type)
            model_init_time = time.time() - model_init_start
            logger.info(f"WhisperModel initialized successfully in {model_init_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize WhisperModel: {str(e)}")
            raise

        total_init_time = time.time() - init_start_time
        logger.info(f"FasterWhisperTranscriber initialization completed in {total_init_time:.2f}s")

    def transcribe(self) -> tuple[str, Dict[str, Any]]:
        """
        Devuelve la transcripción como un único string.
        """
        transcribe_start_time = time.time()
        logger.info("Starting transcription process...")

        # Load and prepare audio
        logger.debug("Loading and preprocessing audio...")
        audio_load_start = time.time()
        try:
            wav_buf = _load_to_wav_buffer(self.cfg.audio_path, self.cfg.target_sr, self.cfg.mono)
            audio_load_time = time.time() - audio_load_start
            logger.info(f"Audio preprocessing completed in {audio_load_time:.2f}s")

            # Track timing in run manager
            if self.run_manager:
                self.run_manager.log_timing("audio_processing", audio_load_time)

        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise

        # Configure transcription parameters
        transcription_params = {
            "language": self.cfg.language,
            "vad_filter": self.cfg.vad_filter,
            "beam_size": self.cfg.beam_size,
            "best_of": self.cfg.best_of,
            "no_speech_threshold": self.cfg.no_speech_threshold,
            "initial_prompt": self.cfg.initial_prompt,
            "word_timestamps": False,
        }
        logger.debug(f"Transcription parameters: {transcription_params}")

        # Perform transcription
        logger.info("Executing Whisper transcription...")
        whisper_start_time = time.time()
        try:
            segments, info = self._model.transcribe(wav_buf, **transcription_params)
            whisper_time = time.time() - whisper_start_time
            logger.info(f"Whisper transcription completed in {whisper_time:.2f}s")

            # Track timing in run manager
            if self.run_manager:
                self.run_manager.log_timing("whisper_transcription", whisper_time)

            # Log transcription info
            logger.debug(f"Transcription info - language: {info.language}, language_probability: {info.language_probability:.4f}")
            if hasattr(info, 'duration'):
                logger.debug(f"Audio duration detected: {info.duration:.2f}s")

        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            raise

        # Process segments
        logger.debug("Processing transcription segments...")
        parts: List[str] = []
        segment_count = 0
        total_segment_duration = 0.0

        for s in segments:
            segment_count += 1
            text = (s.text or "").strip()
            segment_duration = s.end - s.start if hasattr(s, 'start') and hasattr(s, 'end') else 0
            total_segment_duration += segment_duration

            logger.debug(f"Segment {segment_count} - start: {s.start:.2f}s, end: {s.end:.2f}s, duration: {segment_duration:.2f}s, confidence: {getattr(s, 'avg_logprob', 'N/A')}")
            logger.debug(f"Segment {segment_count} text: '{text}'")

            if text:
                parts.append(text)

        logger.info(f"Processed {segment_count} segments, total audio duration: {total_segment_duration:.2f}s")

        # Finalize transcription
        logger.debug("Finalizing transcription output...")
        out = " ".join(parts).strip()

        # Add punctuation if missing
        if out and out[-1] not in ".!?…":
            logger.debug("Adding final punctuation to transcription")
            out += "."

        total_time = time.time() - transcribe_start_time
        char_count = len(out)
        word_count = len(out.split()) if out else 0

        logger.info(f"Transcription completed successfully!")
        logger.info(f"Performance metrics - total_time: {total_time:.2f}s, characters: {char_count}, words: {word_count}")
        logger.info(f"Transcription preview: '{out[:100]}{'...' if len(out) > 100 else ''}'")

        # Prepare additional information for run tracking
        transcription_info = {
            "audio_duration": total_segment_duration,
            "segment_count": segment_count,
            "language_detected": getattr(info, 'language', 'unknown'),
            "language_probability": getattr(info, 'language_probability', 0.0),
            "total_time": total_time,
            "whisper_time": self.run_manager.timings.get("whisper_transcription", 0.0) if self.run_manager else whisper_time,
            "audio_load_time": self.run_manager.timings.get("audio_processing", 0.0) if self.run_manager else audio_load_time,
            "character_count": char_count,
            "word_count": word_count
        }

        return out, transcription_info


# -----------------------------
# Ejemplo mínimo para IDE
# -----------------------------
if __name__ == "__main__":
    # Enable debug logging for demo
    logger.setLevel(logging.DEBUG)

    logger.info("Starting transcription demo...")

    cfg = {
        "audio_path": "data/001_audio.mp4",  # puede ser .mp3/.m4a/.ogg/.wav/.mp4
        "language": "es",
        "model": "large-v3",  # Systran/faster-whisper-large-v3
        "device": "auto",
        "target_sr": 16000,
        "mono": True,
        "vad_filter": True,
    }

    logger.info(f"Demo configuration: {cfg}")

    try:
        logger.info("Creating transcriber instance...")
        tx = FasterWhisperTranscriber(cfg)

        logger.info("Starting transcription...")
        text, info = tx.transcribe()

        logger.info("Transcription demo completed successfully!")
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT:")
        print("="*50)
        print(text)
        print("="*50)
        print("TRANSCRIPTION INFO:")
        print("="*50)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("="*50)

    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise
