from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
from pydub import AudioSegment

SUPPORTED_EXTS = {".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav"}


@dataclass
class SplitConfig:
    input_dir: Path = Path("data")
    output_dir: Path = Path("chunks")
    chunk_ms: int = 3 * 60 * 1000           # 3 minutos
    overlap_ms: int = 0                     # 0 = sin solapamiento
    ensure_mono: bool = True
    sample_width_bytes: int = 2             # 16-bit PCM
    target_frame_rate: Optional[int] = None # ej: 16000 Hz para ASR
    export_format: str = "wav"
    normalize_dbfs: Optional[float] = None  # ej: -16.0
    fail_fast: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "SplitConfig":
        cfg = dict(cfg or {})
        if "input_dir" in cfg and not isinstance(cfg["input_dir"], Path):
            cfg["input_dir"] = Path(cfg["input_dir"])
        if "output_dir" in cfg and not isinstance(cfg["output_dir"], Path):
            cfg["output_dir"] = Path(cfg["output_dir"])
        return cls(**cfg)


class AudioSplitter:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = SplitConfig.from_dict(config)

    # ---------- utils ----------
    def _find_audio_files(self, input_dir: Path) -> Iterable[Path]:
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p

    def _normalize(self, seg: AudioSegment, target_dbfs: float) -> AudioSegment:
        if seg.dBFS == float("-inf"):
            return seg
        change = target_dbfs - seg.dBFS
        return seg.apply_gain(change)

    # ---------- core ----------
    def _split_one_file(self, src: Path) -> List[Path]:
        audio = AudioSegment.from_file(src)

        if self.cfg.ensure_mono:
            audio = audio.set_channels(1)
        if self.cfg.sample_width_bytes:
            audio = audio.set_sample_width(self.cfg.sample_width_bytes)
        if self.cfg.target_frame_rate:
            audio = audio.set_frame_rate(self.cfg.target_frame_rate)
        if self.cfg.normalize_dbfs is not None:
            audio = self._normalize(audio, self.cfg.normalize_dbfs)

        out_dir = self.cfg.output_dir / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        made: List[Path] = []
        step = max(1, self.cfg.chunk_ms - self.cfg.overlap_ms)
        total = len(audio)
        idx = 0
        pos = 0

        while pos < total:
            start = max(0, pos)
            end = min(start + self.cfg.chunk_ms, total)
            idx += 1
            chunk = audio[start:end]
            out_path = out_dir / f"chunk_{idx:03d}.{self.cfg.export_format}"
            chunk.export(out_path, format=self.cfg.export_format)
            made.append(out_path)
            if end == total:
                break
            pos += step

        return made

    def run(self) -> Dict[str, Any]:
        """Ejecuta el split y devuelve un resumen."""
        if not self.cfg.input_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de entrada: {self.cfg.input_dir}")

        files = list(self._find_audio_files(self.cfg.input_dir))
        if not files:
            raise FileNotFoundError(
                f"No encontré audios en {self.cfg.input_dir}. Extensiones: {', '.join(sorted(SUPPORTED_EXTS))}"
            )

        processed = []
        errors = []
        total_chunks = 0

        for src in files:
            try:
                made = self._split_one_file(src)
                total_chunks += len(made)
                processed.append({
                    "source": str(src),
                    "chunks": [str(p) for p in made],
                    "count": len(made)
                })
            except Exception as e:
                msg = f"{src.name}: {e}"
                errors.append(msg)
                if self.cfg.fail_fast:
                    raise

        return {
            "processed": processed,
            "errors": errors,
            "total_chunks": total_chunks,
            "output_dir": str(self.cfg.output_dir.resolve())
        }
