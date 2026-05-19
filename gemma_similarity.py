import pandas as pd
import json
import os
import time
import numpy as np
from tqdm import tqdm
from huggingface_hub import InferenceClient 
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# Hugging Face setup
HF_TOKEN = os.getenv("HF_TOKEN")

hf_client = InferenceClient(token=HF_TOKEN)

# setup local model for similarity
print("Loading the local similarity measurement model (Multilingual)...")
similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
print("Similarity model successfully loaded!\n")


def calculate_semantic_similarity(text_human: str, text_machine: str) -> float:
    """ Measuring semantic proximity using Hugging Face Sentence Transformers. """
    if pd.isna(text_human) or pd.isna(text_machine) or not text_human or not text_machine: 
        return 0.0
    
    try:
        # Converting text to vector (Embeddings)
        vec_human = similarity_model.encode(str(text_human))
        vec_machine = similarity_model.encode(str(text_machine))
        
        # Calculating cosine distance (similarity)
        cos_sim = util.cos_sim(vec_human, vec_machine)
        
        return round(float(cos_sim[0][0]) * 100, 2)
    except Exception as e:
        print(f"Error in similarity: {e}")
        return 0.0


def extract_intent_function_calling(text: str):
    if pd.isna(text) or not text: return "{}"

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
                            "description": "Language code (e.g., 'en', 'id')"
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
                                    "Claim": {
                                        "type": "string", 
                                        "description": "The specific claim made in the text."
                                    },
                                    "Confidence": {
                                        "type": "number", 
                                        "description": "Confidence score of the claim extraction between 0.0 and 1.0."
                                    },
                                    "Relevance": {
                                        "type": "number", 
                                        "description": "Relevance score of the claim to the overall topic between 0.0 and 1.0."
                                    }
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
        # Using Hugging Face's native chat_completion method
        response = hf_client.chat_completion(
            model="google/gemma-4-31B-it", 
            messages=[
                {"role": "user", "content": f"Please extract the intent and data from this transcript: {text}"}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_transcript_data"}},
            max_tokens=1000
        )
        
        # Capture the response text
        message = response.choices[0].message
        
        if message.tool_calls:
            json_output = message.tool_calls[0].function.arguments
            
            json.loads(json_output) 
            
            return json_output
        else:
            print("Model did not return any tool calls.")
            return "{}"
            
    except json.JSONDecodeError:
        print("\nError: Gemma returned invalid JSON arguments.")
        return "{}"
    except Exception as e:
        print(f"\nError in Gemma tool calling: {e}")
        return "{}"

def process_local_file(file_path):
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path): return

    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error details: {e}")
        return

    total_rows = len(df)
    
    raw_similarities = []
    summary_similarities = []
    intent_similarities = []
    json_machine = []
    json_human = []
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Evaluation Process"):
        h_text = str(row.get('Human: Gemma English Translation', ''))
        m_text = str(row.get('Omnilingual: Gemma English Translation', ''))
        
        # calculate Raw Similarity
        raw_similarities.append(calculate_semantic_similarity(h_text, m_text))
        
        # extract JSON (Both Human and Machine)
        res_h = extract_intent_function_calling(h_text)
        res_m = extract_intent_function_calling(m_text)
        json_human.append(res_h)
        json_machine.append(res_m)
        
        # calculate Similarity Intent & Summary
        try:
            dict_h = json.loads(res_h)
            dict_m = json.loads(res_m)
            
            # calculate Summary Similarity
            sum_sim = calculate_semantic_similarity(dict_h.get('Summary', ''), dict_m.get('Summary', ''))
            summary_similarities.append(sum_sim)
            
            # calculate Intent Similarity (based on joined Topics)
            h_intents = ", ".join(dict_h.get('Topics', []))
            m_intents = ", ".join(dict_m.get('Topics', []))
            int_sim = calculate_semantic_similarity(h_intents, m_intents)
            intent_similarities.append(int_sim)
            
        except:
            summary_similarities.append(0.0)
            intent_similarities.append(0.0)

        time.sleep(2)
        
    # Build final DataFrame
    df['Raw Semantic Similarity (%)'] = raw_similarities
    df['Summary Similarity (%)'] = summary_similarities
    df['Intent (Topics) Similarity (%)'] = intent_similarities
    df['JSON Human'] = json_human
    df['JSON Machine'] = json_machine
    
    output_path = "gemma_deep_similarity_result4.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nDone, file saved as: {output_path}")

if __name__ == "__main__":
    file_name = "result_with_score_methode.xlsx" 
    process_local_file(file_name)