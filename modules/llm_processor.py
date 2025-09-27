from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import os

from dotenv import load_dotenv


class LLMProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        cfg keys:
            - transcript_path: str | Path
            - model: str = "gpt-4o-mini"
            - provider: str = "openai"
            - api_key_env: str = "OPENAI_API_KEY"
        """
        self.cfg = dict(config)
        self.transcript_path = Path(self.cfg.get("transcript_path"))
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {self.transcript_path}")

        self.model = self.cfg.get("model", "gpt-4o-mini")
        self.provider = self.cfg.get("provider", "openai")
        self.api_key_env = self.cfg.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(self.api_key_env)

        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

        if self.provider.lower() == "openai":
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    def process(self) -> str:
        """
        Sends transcript text to LLM, gets back a cleaned script
        with speakers labeled (Terapeuta / Paciente).
        Returns one single string.
        """
        transcript_text = self.transcript_path.read_text(encoding="utf-8")

        system_prompt = (
            "Eres un asistente que procesa sesiones terapéuticas. "
            "Toma el texto transcripto y clasifica cada intervención "
            "atribuyéndola a 'Terapeuta' o 'Paciente'. "
            "Identifica claramente el cambio de hablante incluso en el primer turno, "
            "evitando asignar todo el bloque inicial al mismo interlocutor. "
            "Devuelve un guion único, legible, con turnos bien diferenciados, "
            "mostrando cada intervención en una nueva línea con el formato: "
            "'Terapeuta: ...' o 'Paciente: ...'. "
            "No agregues explicaciones adicionales ni comentarios, solo el guion."
        )

        user_prompt = f"TRANSCRIPCIÓN:\n{transcript_text}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    load_dotenv()
    cfg = {
        "transcript_path": "transcription.txt",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    }
    processor = LLMProcessor(cfg)
    script = processor.process()
    print("== SCRIPT ==")
    print(script)
