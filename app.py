#!/usr/bin/env python3
"""
Audio Transcription App with Run Tracking

This app transcribes audio files and saves all results, logs, and metadata
to organized run directories for easy tracking and analysis.
"""

import sys
import logging
from pathlib import Path
from modules.transcript_with_speakers_names import WhisperSpeakerTranscriber
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
        "input_path": "data/001_audio.mp4",  # Change this to your audio file
        "num_speakers": 2,
        "model_size": "large",
        "language": "any",
        "device": "cpu",  # "cpu", "cuda", or "mps"
        "output_dir": "runs",
        "write_script_format": True,
        "write_segments_json": True,
        "use_run_manager": True
    }

    logger.info("Starting transcription workflow...")
    logger.info(f"Audio file: {config['input_path']}")
    logger.info(f"Model: {config['model_size']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Speakers: {config['num_speakers']}")

    # Initialize run manager
    run_manager = RunManager()

    try:
        # Validate audio file exists
        audio_path = Path(config["input_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize transcriber with run tracking
        logger.info("Initializing transcriber...")
        transcriber = WhisperSpeakerTranscriber(run_manager=run_manager)

        # Perform transcription
        logger.info("Starting transcription with speaker diarization...")
        result = transcriber.process(config)

        # Copy logs to run directory (if any log files exist)
        logger.info("Copying log files...")
        if Path("transcriber.log").exists():
            run_manager.copy_logs("transcriber.log")

        # Display results
        print("\n" + "="*60)
        print("TRANSCRIPTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Run ID: {result['run_id']}")
        print(f"Run Directory: {result['run_dir']}")
        print(f"Audio Duration: {result['audio_duration']:.2f}s")
        print(f"Segments: {result['segment_count']}")
        print(f"Characters: {result['character_count']}")
        print(f"Words: {result['word_count']}")
        print("="*60)
        print("SCRIPT FORMAT TRANSCRIPTION:")
        print("="*60)
        if "text" in result:
            print(result["text"])
        else:
            # Read the script file directly
            script_path = result.get("script_path")
            if script_path and Path(script_path).exists():
                with open(script_path, "r", encoding="utf-8") as f:
                    print(f.read())
        print("="*60)

        logger.info(f"All results saved to: {result['run_dir']}")
        return True

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")

        # Copy logs to run directory for debugging
        try:
            if Path("transcriber.log").exists():
                run_manager.copy_logs("transcriber.log")
        except Exception as save_error:
            logger.error(f"Failed to copy logs: {save_error}")

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
