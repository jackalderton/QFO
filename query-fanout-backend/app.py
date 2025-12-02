import os
import io
import time
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trafilatura import fetch_url, extract
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Configuration & Global Variables ---

# These environment variables are set in the Dockerfile
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
FANOUT_MODEL_NAME = os.getenv("FANOUT_MODEL_NAME", "google/flan-t5-small")

# Caching for models to ensure they are only loaded once
_embed_model = None
_fanout_tokenizer = None
_fanout_model = None

app = FastAPI(title="Reliable Search Fanout & Document Analysis API", version="1.0.0")

# --- Helper Functions for Model Loading ---

def get_embed_model():
    """Loads and caches the Sentence Transformer embedding model."""
    global _embed_model
    if _embed_model is None:
        try:
            print(f"Loading embedding model from: {EMBED_MODEL_NAME}")
            # Load from local path defined in Dockerfile
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")
    return _embed_model

def get_fanout_model():
    """Loads and caches the Flan-T5 model and tokenizer."""
    global _fanout_tokenizer, _fanout_model
    if _fanout_model is None:
        try:
            print(f"Loading fanout model from: {FANOUT_MODEL_NAME}")
            # Load from local path defined in Dockerfile
            _fanout_tokenizer = AutoTokenizer.from_pretrained(FANOUT_MODEL_NAME)
            _fanout_model = AutoModelForSeq2SeqLM.from_pretrained(FANOUT_MODEL_NAME)
        except Exception as e:
            print(f"Error loading fanout model: {e}")
            raise RuntimeError(f"Could not load fanout model: {e}")
    return _fanout_tokenizer, _fanout_model

# --- Core Business Logic ---

def fetch_main_text(url: str) -> str:
    """Fetches and extracts the main content from a URL using trafilatura."""
    try:
        downloaded = fetch_url(url, no_ssl_check=True)
        if downloaded:
            extracted_text = extract(
                downloaded, 
                output_format='text', 
                with_metadata=False,
                include_comments=False,
                include_images=False
            )
            # Simple cleanup and length check
            return extracted_text.strip() if extracted_text else ""
        return ""
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def generate_fanouts(base_query: str, k: int = 5) -> list[str]:
    """
    Generates k unique, related query variants using the FLAN-T5 model.

    Incorporates:
    - Clear instruction prompting.
    - Optimized decoding parameters for diversity (temperature, repetition_penalty).
    - Robust post-processing for cleanup and deduplication.
    """
    tokenizer, model = get_fanout_model()

    # 1. Improved Single-Variant Prompt Structure
    prompt = (
        "Task: Rewrite the original search query into a single, closely related variant query.\n"
        "Constraints:\n"
        "- DO NOT use the original query exactly.\n"
        "- Keep the variant query under 12 words.\n"
        f"Original Query: \"{base_query}\"\n"
        "Variant Query:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    # 2. Optimized Decoding Parameters (Crucial for diversity/reliability)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,          # Short limit to match query length constraint
            do_sample=True,             # Enable sampling for creative diversity
            temperature=0.85,           # Higher temp for diverse language
            num_return_sequences=k,     # Generate k sequences
            repetition_penalty=1.2,     # Discourage repetition within each sequence
            num_beams=1,                # Must be 1 when using do_sample=True
            pad_token_id=tokenizer.eos_token_id,
        )

    # 3. Robust Post-Processing and Deduplication
    raw_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    variants = []
    seen = set()
    base_lower = base_query.lower().strip()

    for raw_text in raw_texts:
        cleaned = raw_text.strip()

        # Remove the prompt prefix if the model repeats it
        if cleaned.lower().startswith(("variant query:", "variant query")):
            cleaned = cleaned.split(":", 1)[-1].strip()

        # Basic filter criteria
        lower = cleaned.lower()
        if not cleaned:
            continue
        if lower == base_lower: # Reject if it's identical to the original
            continue

        # Simple filter for length (if too long, it's likely a bad generation)
        if len(cleaned.split()) > 15:
            continue

        if lower not in seen:
            seen.add(lower)
            variants.append(cleaned)

    # Ensure we only return up to k unique, valid variants
    return variants[:k]

def analyze_documents(search_queries: list[str], urls_and_content: dict[str, str], threshold: float = 0.5) -> dict:
    """
    Generates embeddings for queries and documents, calculates similarity,
    and returns relevant content based on a similarity threshold.
    """
    embed_model = get_embed_model()

    # 1. Prepare Data
    query_embeddings = embed_model.encode(search_queries, convert_to_tensor=True)

    # Filter out empty documents (which would fail embedding or bias similarity)
    valid_urls = [url for url, content in urls_and_content.items() if content]
    valid_contents = [content for content in urls_and_content.values() if content]

    if not valid_contents:
        return {"analysis": "No valid document content was retrieved for analysis."}

    # 2. Generate Document Embeddings
    print(f"Embedding {len(valid_contents)} documents...")
    doc_embeddings = embed_model.encode(valid_contents, convert_to_tensor=True)

    # 3. Calculate Cosine Similarity (using dot product for normalized vectors)
    similarities = torch.nn.functional.cosine_similarity(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)
    # Convert tensor to numpy array
    similarities_np = similarities.cpu().numpy()

    # 4. Process Results
    results = {}
    for i, query in enumerate(search_queries):
        query_results = []
        for j, url in enumerate(valid_urls):
            score = similarities_np[i, j]
            if score >= threshold:
                query_results.append({
                    "url": url,
                    "similarity_score": round(float(score), 4),
                    # Exclude content here to keep the API response small,
                    # but you can include a snippet if needed.
                })

        # Sort results by score (highest first)
        query_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        results[query] = {
            "query": query,
            "match_count": len(query_results),
            "relevant_documents": query_results
        }

    return results

# --- FastAPI Endpoints ---

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyzes a CSV file containing a single base query and a list of URLs.

    The CSV file must have at least two columns: 'BaseQuery' and 'URLs'.
    The BaseQuery cell must contain the query string. The URLs cell must contain
    a semicolon-separated list of URLs.
    """
    start_time = time.time()

    # 1. Validate and Read CSV Input
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_file)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file. Ensure it is correctly formatted.")

    # 2. Extract Base Query and URLs
    try:
        # Assuming the first row is the main configuration row
        base_query = df['BaseQuery'].iloc[0].strip()
        raw_urls = df['URLs'].iloc[0].strip().split(';')
        urls = [u.strip() for u in raw_urls if u.strip()]
    except KeyError:
        raise HTTPException(status_code=400, detail="CSV must contain 'BaseQuery' and 'URLs' columns.")
    except IndexError:
        raise HTTPException(status_code=400, detail="CSV must contain at least one row of data.")

    # 3. Generate Fanout Queries
    print(f"Generating fanout queries for: {base_query}")
    fanout_queries = generate_fanouts(base_query, k=5)
    all_queries = [base_query] + fanout_queries
    print(f"Generated {len(fanout_queries)} variants.")

    # 4. Fetch Document Content
    print(f"Fetching content for {len(urls)} URLs...")
    urls_and_content = {}
    for url in urls:
        urls_and_content[url] = fetch_main_text(url)

    # 5. Run Semantic Analysis
    print("Running semantic analysis...")
    analysis_results = analyze_documents(
        search_queries=all_queries,
        urls_and_content=urls_and_content,
        threshold=0.55 # Slightly higher threshold for better relevance
    )

    # 6. Compile Final Response
    total_time = round(time.time() - start_time, 2)
    
    response = {
        "base_query": base_query,
        "fanout_queries": fanout_queries,
        "document_count": len(urls),
        "analysis": analysis_results,
        "processing_time_seconds": total_time,
    }

    print(f"Request complete in {total_time} seconds.")
    return JSONResponse(content=response)

# Health check endpoint
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "API is operational"}

# Initialize models on startup (FastAPI's lifespan events)
@app.on_event("startup")
async def startup_event():
    print("Application startup: Pre-loading models...")
    try:
        get_embed_model()
        get_fanout_model()
        print("Models successfully pre-loaded.")
    except Exception as e:
        print(f"FATAL: Model loading failed during startup: {e}")
        # In a production environment, you might stop the process here
        pass

if __name__ == "__main__":
    import uvicorn
    # Use the default ports set in the Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8080)
