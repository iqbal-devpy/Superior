import pickle
import os
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize local embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Store embeddings in a dictionary
vector_store = {}

# Function to split text into chunks
def chunk_text(text, chunk_size=400, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# Load PDFs, Extract Text, and Chunk
def process_pdfs(pdf_folder="data"):
    all_chunks = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                chunks = chunk_text(doc.page_content)  # Split long text into chunks
                all_chunks.extend(chunks)

    return all_chunks

# # Convert Extracted Text to Embeddings and Store in Dictionary
# def setup_memory_store():
#     texts = process_pdfs()  # Now returns smaller text chunks

#     if not texts:
#         print("No text extracted from PDFs.")
#         return

#     print(f"Generating embeddings for {len(texts)} chunks...")
    
#     embeddings = embedding_model.encode(texts).tolist()

#     # Store chunk embeddings in dictionary
#     for i, text in enumerate(texts):
#         vector_store[i] = {"embedding": embeddings[i], "text": text}

EMBEDDING_FILE = "embeddings.pkl"

def save_embeddings():
    """Saves vector_store to a file so embeddings don’t regenerate every time."""
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(vector_store, f)

def load_embeddings():
    """Loads embeddings from the file if it exists."""
    global vector_store
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            vector_store = pickle.load(f)
        print(f"Loaded {len(vector_store)} stored embeddings from file.")
    else:
        print("No saved embeddings found. Generating new ones.")

def setup_memory_store():
    """Loads embeddings if they exist; otherwise, processes PDFs and generates new embeddings."""
    load_embeddings()
    
    if vector_store:  # If embeddings are already loaded, skip regeneration
        return
    
    texts = process_pdfs()  # Extract text from PDFs

    if not texts:
        print("No text extracted from PDFs.")
        return

    print(f"Generating embeddings for {len(texts)} chunks...")
    
    embeddings = embedding_model.encode(texts).tolist()

    # Store chunk embeddings in dictionary
    for i, text in enumerate(texts):
        vector_store[i] = {"embedding": embeddings[i], "text": text}

    save_embeddings()  # Save the generated embeddings for future use


setup_memory_store()

# def retrieve_answer(query):
#     if not vector_store:
#         return "No documents available to retrieve an answer."

#     query_embedding = embedding_model.encode([query])[0]
#     query_vector = np.array([query_embedding])

#     stored_vectors = np.array([v["embedding"] for v in vector_store.values()])
#     similarities = cosine_similarity(query_vector, stored_vectors)[0]

#     # Retrieve top 3 matches
#     top_indices = np.argsort(similarities)[-3:][::-1]
#     results = [vector_store[i]["text"] for i in top_indices]

#     # Extract only sentences containing query keywords
#     query_keywords = query.lower().split()
#     best_sentence = None
#     max_keyword_match = 0

#     for result in results:
#         sentences = re.split(r'(?<=[.!?])\s+', result)
#         for sentence in sentences:
#             match_count = sum(keyword in sentence.lower() for keyword in query_keywords)
#             if match_count > max_keyword_match:
#                 max_keyword_match = match_count
#                 best_sentence = sentence

#     return best_sentence if best_sentence else "No relevant info found."


# def retrieve_answer(query):
#     if not vector_store:
#         return "No documents available to retrieve an answer."

#     query_embedding = embedding_model.encode([query])[0]
#     query_vector = np.array([query_embedding])

#     stored_vectors = np.array([v["embedding"] for v in vector_store.values()])
#     similarities = cosine_similarity(query_vector, stored_vectors)[0]

#     # Retrieve top 3 most relevant chunks
#     top_indices = np.argsort(similarities)[-3:][::-1]
#     retrieved_texts = [vector_store[i]["text"] for i in top_indices]

#     # Extract only the most relevant part (first 2-3 sentences)
#     query_keywords = query.lower().split()
#     best_sentences = []
    
#     for text in retrieved_texts:
#         sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
#         for sentence in sentences:
#             if any(keyword in sentence.lower() for keyword in query_keywords):
#                 best_sentences.append(sentence)
#                 if len(best_sentences) >= 2:  # Limit to 2 most relevant sentences
#                     return " ".join(best_sentences)

#     return best_sentences[0] if best_sentences else "No relevant info found."

import re
import random


def make_conversational(response, query):
    """Adds a conversational tone but avoids redundant phrases."""
    
    response = clean_response(response)  # Clean text first

    if len(response.split()) < 10:  # If response is short, return it directly
        return response  

    if "return" in query.lower():
        intro = random.choice(["Sure! Here’s our return policy:", "Of course! Here's how returns work:", "You can return your product under these conditions:"])
    elif "exchange" in query.lower():
        intro = random.choice(["No problem! Here's our exchange policy:", "You can exchange your product under these conditions:", "Exchanges are allowed under these terms:"])
    elif "contact" in query.lower():
        intro = ""  # No need for "You can contact us here:"
    elif "shipping" in query.lower() or "order" in query.lower():
        intro = random.choice(["Your order details:", "Shipping details are as follows:"])
    else:
        intro = ""

    return f"{intro} {response}".strip()  # Avoid extra spaces if no intro is used


def clean_response(text):
    """Cleans response by removing unnecessary headers, bullet points, and special characters."""
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)  # Remove leading non-alphanumeric characters (like "•")
    text = text.replace("Contact Us", "").strip()  # Remove redundant headers like "Contact Us"
    text = text.replace("Return & Exchange Policy", "").strip()  # Remove unnecessary section headers
    text = text.replace("Shipping & Delivery Policy", "").strip()
    
    return text[0].upper() + text[1:] if text else "No relevant info found."

def retrieve_answer(query):
    if not vector_store:
        return "No documents available to retrieve an answer."

    # Handle greetings
    greetings = ["hello", "hi", "hey", "good morning", "good evening"]
    if query.lower() in greetings:
        return random.choice(["Hey there! How can I assist you today?", "Hello! What would you like to know?", "Hi! Feel free to ask me anything."])

    query_embedding = embedding_model.encode([query])[0]
    query_vector = np.array([query_embedding])

    stored_vectors = np.array([v["embedding"] for v in vector_store.values()])
    similarities = cosine_similarity(query_vector, stored_vectors)[0]

    # Retrieve top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]
    retrieved_texts = [vector_store[i]["text"] for i in top_indices]

    # Extract only the most relevant sentence
    query_keywords = query.lower().split()
    
    for text in retrieved_texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in query_keywords):
                cleaned_sentence = clean_response(sentence)
                return make_conversational(cleaned_sentence, query)  # Return only the most relevant sentence

    return "No relevant info found."
