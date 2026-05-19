import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMComparator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def calculate_similarity(self, text_human: str, text_machine: str) -> float:
        """Calculating semantic similarity using OpenAI Embeddings"""
        if not text_human or not text_machine: 
            return 0.0
        
        try:
            response = self.client.embeddings.create(
                input=[str(text_human), str(text_machine)],
                model="text-embedding-3-small"
            )
            
            vec_human = np.array(response.data[0].embedding)
            vec_machine = np.array(response.data[1].embedding)
            
            # Calculate Cosine Similarity
            dot_product = np.dot(vec_human, vec_machine)
            norm_a = np.linalg.norm(vec_human)
            norm_b = np.linalg.norm(vec_machine)
            
            # Avoiding division by zero
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            return round(float(similarity) * 100, 2)
            
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return 0.0