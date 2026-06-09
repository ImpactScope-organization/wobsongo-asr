from typing import Protocol, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

# Enum for model selection
class ModelType(Enum):
    OMNILINGUAL = "omnilingual"

class TargetLanguage(Enum):
    ENGLISH = "en"
    FRENCH = "fr"

# This is the Object representation of the JSON that Replicate returns.
@dataclass
class TranscriptionOutput:
    transcript: str
    language_selected: str
    chunks: list[dict[str, Any]]
    final_text: str

class TranscriberProtocol(Protocol):
    def transcribe(
        self, 
        model: ModelType,
        target_lang: TargetLanguage,
        audio: Path | None = None,
        audio_url: str | None = None,
        human_transcription: str | None = None,
    ) -> TranscriptionOutput | dict[str, str]:
        ...