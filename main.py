from tqdm import tqdm
import json
import ollama
import time
import os
import numpy as np
from numpy.linalg import norm
from alive_progress import alive_bar

def parse_file(file_path):
    quid_values = []
    
    try:
        with open(file_path, encoding="utf-8") as file:
            json_data = json.load(file)
            
            # Extract QUID values from the data array
            if "data" in json_data and isinstance(json_data["data"], list):
                for item in json_data["data"]:
                    if "QUID" in item:
                        quid_values.append(item["QUID"])
            
        return quid_values
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []

def save_embeddings(file_path, embeddings, indices):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{file_path}", "w") as file:
        json.dump({
            "embeddings": embeddings,
            "indices": indices
        }, file)

def load_embeddings(file_path):
    if not os.path.exists("embeddings"):
        return False
    try:
        with open(f"embeddings/{file_path}", "r") as file:
            data = json.load(file)
            return data.get("embeddings", []), data.get("indices", [])
    except FileNotFoundError:
        return False

def get_embeddings(file_path, modelname, chunks):
    existing_data = load_embeddings(file_path)
    
    if existing_data is not False:
        existing_embeddings, existing_indices = existing_data
    else:
        existing_embeddings, existing_indices = [], []

    # Find chunks that need processing
    chunks_to_process = []
    chunk_indices = []
    for i, chunk in enumerate(chunks):
        if i not in existing_indices:
            chunks_to_process.append(chunk)
            chunk_indices.append(i)

    if not chunks_to_process:
        # print("All chunks already processed")
        return existing_embeddings

    embeddings = []
    failed_chunks = []
    start_time = time.time()
    
    with tqdm(total=len(chunks_to_process), desc="Generating embeddings") as pbar:
        for proc_idx, chunk in enumerate(chunks_to_process):
            try:
                start = time.time()
                result = ollama.embeddings(model=modelname, prompt=chunk)
                duration = time.time() - start
                embeddings.append((chunk_indices[proc_idx], result["embedding"], duration))
            except Exception as e:
                failed_chunks.append((chunk_indices[proc_idx], str(e)))
            pbar.update(1)
    
    # Sort embeddings by index and extract only embedding values
    embeddings.sort(key=lambda x: x[0])
    new_embeddings = [emb[1] for emb in embeddings]
    new_indices = [emb[0] for emb in embeddings]

    # Merge with existing embeddings
    final_embeddings = existing_embeddings.copy()
    final_indices = existing_indices.copy()
    
    for i, (emb, idx) in enumerate(zip(new_embeddings, new_indices)):
        if idx not in final_indices:
            final_embeddings.append(emb)
            final_indices.append(idx)

    # Print statistics
    if embeddings:
        avg_time = sum(emb[2] for emb in embeddings) / len(embeddings)
        print(f"\nAverage time per chunk: {avg_time:.2f}s")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"New embeddings: {len(embeddings)}")
    print(f"Total embeddings: {len(final_embeddings)}")
    print(f"Failed chunks: {len(failed_chunks)}")
    
    if failed_chunks:
        print("\nFailed chunks:")
        for index, error in failed_chunks:
            print(f"Chunk {index}: {error}")
    
    if embeddings:
        save_embeddings(file_path, final_embeddings, final_indices)
    
    return final_embeddings

# def find_most_similar(needle, haystack):
#     needle_norm = norm(needle)
#     similarity_scores = [
#         np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
#     ]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def find_most_similar(needle: list, haystack: list, top_k: int = 5) -> list:
    """Calculate cosine similarity between vectors."""
    try:
        # Input validation
        if not haystack or not needle:
            return []
            
        # Convert lists to numpy arrays with explicit dimension check
        expected_dim = len(needle)
        
        # Filter haystack vectors to ensure consistent dimensions
        valid_vectors = [vec for vec in haystack if len(vec) == expected_dim]
        
        if not valid_vectors:
            print(f"No vectors with matching dimension {expected_dim} found")
            return []
            
        # Convert to numpy arrays after validation
        query = np.array(needle, dtype=np.float32)
        matrix = np.array(valid_vectors, dtype=np.float32)
        
        # Rest of similarity computation
        dot_product = matrix @ query
        matrix_norm = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        
        eps = np.finfo(float).eps
        matrix_norm = np.maximum(matrix_norm, eps)
        query_norm = max(query_norm, eps)
        
        cos_sim = dot_product / (matrix_norm * query_norm)
        
        k = min(top_k, len(cos_sim))
        indices = np.argsort(cos_sim)[-k:][::-1]
        scores = cos_sim[indices]
        
        return list(zip(scores, indices))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Needle shape: {len(needle)}")
        print(f"First haystack vector shape: {len(haystack[0]) if haystack else 'empty'}")
        return []

def main():
    SYSTEM_PROMPT = """You are a food allergen detection expert. Analyze ingredient lists to identify allergens and format them according to specifications:
    1.ALLERGENS TO IDENTIFY (case-insensitive):gluten, wheat, oats, barley, rye, spelt, kamut, milk, eggs, fish, crustaceans, molluscs, peanut, nuts, sesame, mustard, celery, soya, sulphites, lupin, phenylalanine
    2.PROCESSING STEPS:
    - Scan input text for allergens
    - Wrap each found allergen in <b> tags
    3.RULES:
    - Match allergens case-insensitively
    - Maintain original text formatting
    4.OUTPUT: return only original text with each allergen word wrapped in <b> tags separately """

    file_path = "data.json"
    quids = parse_file(file_path)
    embeddings = get_embeddings("embeddings.json", "embeddinggemma", quids[:5000])
    prompt = input("Enter the ingredient list -> ")
    prompt_embedding = ollama.embeddings(model="embeddinggemma", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    # for item in most_similar_chunks:
    #     print(item[0], quids[item[1]])

    similar_texts = [quids[item[1]] for item in most_similar_chunks]
    print("Generating response:")
    with alive_bar(spinner='dots') as bar:
        response = ollama.chat(
            model="gemma3",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n".join(similar_texts),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        bar()  # Complete the progress

    print(f"\n{response['message']['content']}")

if __name__ == "__main__":
    main()