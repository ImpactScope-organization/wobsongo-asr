import pandas as pd
import json
import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm  
from dotenv import load_dotenv

#Setup Local OPENAI
load_dotenv()
client = OpenAI()

# Meaning Comparator
def calculate_semantic_similarity(text_human: str, text_machine: str) -> float:
    """
    Measuring semantic proximity (Distance 'd') using OpenAI Embeddings.
    """
    if pd.isna(text_human) or pd.isna(text_machine) or not text_human or not text_machine: 
        return 0.0
    
    try:
        response = client.embeddings.create(
            input=[str(text_human), str(text_machine)],
            model="text-embedding-3-small"
        )
        vec_human = np.array(response.data[0].embedding)
        vec_machine = np.array(response.data[1].embedding)
        
        dot_product = np.dot(vec_human, vec_machine)
        norm_a = np.linalg.norm(vec_human)
        norm_b = np.linalg.norm(vec_machine)
        similarity = dot_product / (norm_a * norm_b)
        
        return round(float(similarity) * 100, 2)
    except Exception as e:
        print(f"Error in similarity: {e}")
        return 0.0

# Intent / Meaning Extractor  - Function Calling
def extract_intent_function_calling(text: str):
    """
    Extracting claims using OpenAI Function Calling
    """
    if pd.isna(text) or not text or len(str(text)) < 5: return "{}"

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
                                    "Claim": {"type": "string", "description": "The specific claim made in the text."},
                                    "Confidence": {"type": "number", "description": "Confidence score 0.0 - 1.0."},
                                    "Relevance": {"type": "number", "description": "Relevance score 0.0 - 1.0."}
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional content analyst. Please extract the intent and data from this transcript."},
                {"role": "user", "content": text}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_transcript_data"}}
        )
        
        json_output = response.choices[0].message.tool_calls[0].function.arguments
        return json_output
    except Exception as e:
        print(f"Error in JSON extraction: {e}")
        return "{}"

# Local File Batch Process
def process_local_file(file_path):
    print(f"Reading file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read file. Error: {e}")
        return

    total_rows = len(df)
    print(f"Successfully loaded {total_rows} rows. Starting OpenAI Deep Evaluation...\n")
    
    raw_similarities = []
    summary_similarities = []
    intent_similarities = []
    json_machine = []
    json_human = []
    
    # Data iteration
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Evaluation Process"):
        h_text = str(row.get('Human: Gemma English Translation', ''))
        m_text = str(row.get('Omnilingual: Gemma English Translation', ''))
        
        # Raw Semantic Similarity
        raw_similarities.append(calculate_semantic_similarity(h_text, m_text))
        
        # Extract JSON (Both Human and Machine)
        res_h = extract_intent_function_calling(h_text)
        res_m = extract_intent_function_calling(m_text)
        json_human.append(res_h)
        json_machine.append(res_m)

        # Calculate Summary and Intent Similarity
        try:
            dict_h = json.loads(res_h)
            dict_m = json.loads(res_m)
            
            # Comparison Summary
            sum_sim = calculate_semantic_similarity(dict_h.get('Summary', ''), dict_m.get('Summary', ''))
            summary_similarities.append(sum_sim)
            
            # Comparison Intent (Topics)
            h_topics = ", ".join(dict_h.get('Topics', []))
            m_topics = ", ".join(dict_m.get('Topics', []))
            int_sim = calculate_semantic_similarity(h_topics, m_topics)
            intent_similarities.append(int_sim)
        except:
            summary_similarities.append(0.0)
            intent_similarities.append(0.0)
        
    df['OpenAI Raw Semantic Similarity (%)'] = raw_similarities
    df['OpenAI Summary Similarity (%)'] = summary_similarities
    df['OpenAI Intent Similarity (%)'] = intent_similarities
    df['Extracted JSON (Human)'] = json_human
    df['Extracted JSON (Machine)'] = json_machine

    output_path = "openai_deep_evaluation_results_11.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nDone! File saved as: {output_path}")

if __name__ == "__main__":
    file_name = "result_with_score_methode.xlsx" 
    process_local_file(file_name)