#!/usr/bin/env python3
"""
Audio Transcription App with Run Tracking

This app transcribes audio files and saves all results, logs, and metadata
to organized run directories for easy tracking and analysis.
"""

import sys
import logging
from pathlib import Path
from modules.transcriber import FasterWhisperTranscriber
from modules.run_manager import RunManager

# Configure logging for the main app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main transcription workflow with run tracking"""

    # Configuration
    config = {
        "audio_path": "chunks/001_audio/chunk_003.wav",  # Change this to your audio file
        "language": "es",
        "model": "Systran/faster-whisper-large-v3",   # or "large-v2" for lighter model
        "device": "auto",  # "auto", "cpu", or "cuda"
        "target_sr": 16000,
        "mono": True,
        "vad_filter": True,
    }

    logger.info("Starting transcription workflow...")
    logger.info(f"Audio file: {config['audio_path']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")

    # Initialize run manager
    run_manager = RunManager()

    try:
        # Validate audio file exists
        audio_path = Path(config["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize transcriber with run tracking
        logger.info("Initializing transcriber...")
        transcriber = FasterWhisperTranscriber(config, run_manager=run_manager)

        # Perform transcription
        logger.info("Starting transcription...")
        transcription, info = transcriber.transcribe()

        # Save results to run directory
        logger.info("Saving transcription results...")
        run_manager.save_transcription(transcription)

        # Copy logs to run directory
        logger.info("Copying log files...")
        run_manager.copy_logs("transcriber.log")

        # Save metadata
        logger.info("Saving run metadata...")
        metadata = run_manager.save_metadata(
            audio_file=config["audio_path"],
            transcription=transcription,
            config=config,
            audio_duration=info["audio_duration"],
            segment_count=info["segment_count"],
            success=True
        )

        # Display results
        print("\n" + "="*60)
        print("TRANSCRIPTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Run ID: {run_manager.run_id}")
        print(f"Run Directory: {run_manager.run_dir}")
        print(f"Total Time: {metadata.total_time_seconds:.2f}s")
        print(f"Audio Duration: {info['audio_duration']:.2f}s")
        print(f"Segments: {info['segment_count']}")
        print(f"Characters: {info['character_count']}")
        print(f"Words: {info['word_count']}")
        print("="*60)
        print("TRANSCRIPTION:")
        print("="*60)
        print(transcription)
        print("="*60)

        logger.info(f"All results saved to: {run_manager.run_dir}")
        return True

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")

        # Save error metadata
        try:
            run_manager.save_metadata(
                audio_file=config.get("audio_path", "unknown"),
                transcription="",
                config=config,
                success=False,
                error_message=str(e)
            )
            run_manager.copy_logs("transcriber.log")
        except Exception as save_error:
            logger.error(f"Failed to save error metadata: {save_error}")

        print(f"\nERROR: {e}")
        print(f"Error details saved to: {run_manager.run_dir}")
        return False


def list_previous_runs():
    """List all previous transcription runs"""
    runs = RunManager.list_runs()

    if not runs:
        print("No previous runs found.")
        return

    print("\n" + "="*80)
    print("PREVIOUS RUNS")
    print("="*80)

    for i, run in enumerate(runs[:10], 1):  # Show last 10 runs
        status = "✓ SUCCESS" if run['success'] else "✗ FAILED"
        duration = run.get('total_time_seconds', 0)
        words = run.get('transcription_length_words', 0)

        print(f"{i:2d}. {run['run_id']}")
        print(f"    Status: {status}")
        print(f"    Time: {run['timestamp'][:19]}")
        print(f"    Audio: {Path(run['audio_file']).name}")
        print(f"    Duration: {duration:.1f}s | Words: {words}")
        if not run['success']:
            print(f"    Error: {run.get('error_message', 'Unknown error')}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_previous_runs()
    else:
        success = main()
        sys.exit(0 if success else 1)
