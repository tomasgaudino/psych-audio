# run_manager.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import shutil
import uuid
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    run_id: str
    timestamp: str
    audio_file: str
    audio_size_bytes: int
    audio_duration_seconds: float
    transcription_config: Dict[str, Any]
    model_name: str
    device_used: str
    total_time_seconds: float
    audio_processing_time_seconds: float
    whisper_time_seconds: float
    transcription_length_chars: int
    transcription_length_words: int
    segment_count: int
    success: bool
    error_message: Optional[str] = None


class RunManager:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)

        # Generate unique run ID
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        # Initialize timing
        self.start_time = time.time()
        self.timings = {}

        logger.info(f"Created new run: {self.run_id}")
        logger.info(f"Run directory: {self.run_dir}")

    def log_timing(self, event: str, duration: float):
        """Log timing for specific events"""
        self.timings[event] = duration
        logger.debug(f"Run {self.run_id} - {event}: {duration:.2f}s")

    def save_transcription(self, transcription: str):
        """Save transcription to text file"""
        transcription_file = self.run_dir / "transcription.txt"
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logger.info(f"Transcription saved to: {transcription_file}")

    def copy_logs(self, log_file_path: str = "transcriber.log"):
        """Copy log files to run directory"""
        log_path = Path(log_file_path)
        if log_path.exists():
            dest_log = self.run_dir / f"transcriber_{self.run_id}.log"
            shutil.copy2(log_path, dest_log)
            logger.info(f"Log file copied to: {dest_log}")
        else:
            logger.warning(f"Log file not found: {log_path}")

    def save_metadata(self,
                     audio_file: str,
                     transcription: str,
                     config: Dict[str, Any],
                     audio_duration: float = 0.0,
                     segment_count: int = 0,
                     success: bool = True,
                     error_message: Optional[str] = None):
        """Save run metadata as JSON"""

        # Calculate final metrics
        total_time = time.time() - self.start_time
        audio_path = Path(audio_file)
        audio_size = audio_path.stat().st_size if audio_path.exists() else 0

        metadata = RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            audio_file=str(audio_path.resolve()),
            audio_size_bytes=audio_size,
            audio_duration_seconds=audio_duration,
            transcription_config=config,
            model_name=config.get("model", "unknown"),
            device_used=config.get("device", "unknown"),
            total_time_seconds=total_time,
            audio_processing_time_seconds=self.timings.get("audio_processing", 0.0),
            whisper_time_seconds=self.timings.get("whisper_transcription", 0.0),
            transcription_length_chars=len(transcription),
            transcription_length_words=len(transcription.split()) if transcription else 0,
            segment_count=segment_count,
            success=success,
            error_message=error_message
        )

        # Save metadata
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to: {metadata_file}")

        # Log summary
        if success:
            logger.info(f"Run {self.run_id} completed successfully!")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Transcription: {len(transcription)} chars, {len(transcription.split())} words")
        else:
            logger.error(f"Run {self.run_id} failed: {error_message}")

        return metadata

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of the current run"""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "elapsed_time": time.time() - self.start_time,
            "timings": self.timings
        }

    @classmethod
    def list_runs(cls, runs_dir: str = "runs") -> list[Dict[str, Any]]:
        """List all previous runs with their metadata"""
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return []

        runs = []
        for run_dir in runs_path.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        runs.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for run {run_dir.name}: {e}")

        # Sort by timestamp (newest first)
        return sorted(runs, key=lambda x: x.get('timestamp', ''), reverse=True)