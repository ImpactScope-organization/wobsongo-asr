import os
import json
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExtractionOutput:
    lang_code: str = ""
    paraphrased_transcript: str = ""
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    key_points: List[Dict[str, Any]] = field(default_factory=list)
    raw_json_string: str = "{}"

class LLMExtractor:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def extract_data(self, text: str, target_lang: str = "English") -> ExtractionOutput:
        """Extracting intent, summary, and topic using OpenAI Function Calling"""
        if not text or len(str(text)) < 5:
            return ExtractionOutput()

        tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_transcript_data",
                "description": "Extract structured information from an ASR transcript including summary, topics, and key claims.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lang_code": {
                            "type": "string",
                            "description": "Language code (e.g., 'en', 'fr')"
                        },
                        "Paraphrased_Transcript": {
                            "type": "string",
                            "description": "A fluid, grammatically corrected paraphrased version of the raw machine transcript. Make it easy to read while keeping the original meaning."
                        },
                        "Summary": {
                            "type": "string",
                            "description": "Brief summary of the text."
                        },
                        "Topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of core topics discussed in the transcript."
                        },
                        "Key_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Claim": {"type": "string", "description": "The specific claim made in the text."},
                                    "Confidence": {"type": "number", "description": "Confidence score. Must be a float between 0.0 and 1.0 (e.g., 0.85)"},
                                    "Relevance": {"type": "number", "description": "Relevance score. Must be a float between 0.0 and 1.0 (e.g., 0.9)"}
                                },
                                "required": ["Claim", "Confidence", "Relevance"]
                            }
                        }
                    },
                    "required": ["lang_code", "Paraphrased_Transcript", "Summary", "Topics", "Key_points"]
                }
            }
        }
    ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a professional content analyst. Extract the intent and data from this {target_lang} transcript. Strictly use 0.0 to 1.0 scale for Confidence and Relevance scores."},
                    {"role": "user", "content": text}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "extract_transcript_data"}},
                temperature=0.1
            )
            
            json_output_str = response.choices[0].message.tool_calls[0].function.arguments
            parsed_data = json.loads(json_output_str)
            
            # Return as Object (Dataclass)
            return ExtractionOutput(
                lang_code=parsed_data.get("lang_code", ""),
                paraphrased_transcript=parsed_data.get("Paraphrased_Transcript", ""),
                summary=parsed_data.get("Summary", ""),
                topics=parsed_data.get("Topics", []),
                key_points=parsed_data.get("Key_points", []),
                raw_json_string=json_output_str
            )
        except Exception as e:
            print(f"Error in extraction: {e}")
            return ExtractionOutput()