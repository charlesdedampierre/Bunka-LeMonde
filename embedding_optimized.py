import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from tqdm import tqdm
import json


def save_embeddings_with_ids(batch_embeddings, batch_ids, batch_id):
    # Define file path
    file_path = os.path.join(output_dir, f"embeddings_with_ids_batch_{batch_id}.npz")
    # Save IDs and embeddings in one .npz file
    np.savez(file_path, ids=batch_ids, embeddings=batch_embeddings)


df_docs = pd.read_csv("raw_data/lemonde_bunka.csv")
sentences = list(df_docs['titles'])
ids = list(df_docs['0'])

model = SentenceTransformer('OrdalieTech/Solon-embeddings-large-0.1', device="cuda")

# Settings
batch_size = 1000  # Adjust according to your machine's memory; smaller if you encounter memory issues
num_articles = len(sentences)
output_dir = "./embeddings"
os.makedirs(output_dir, exist_ok=True)

for batch_start in tqdm(range(0, num_articles, batch_size), desc="Processing articles"):
    # Process in batches
    batch_articles = sentences[batch_start:batch_start+batch_size]
    batch_ids = ids[batch_start:batch_start+batch_size]
    
    # Generate embeddings
    batch_embeddings = model.encode(batch_articles, convert_to_tensor=True, batch_size=batch_size)
    
    # Convert tensors to numpy array for saving
    batch_embeddings_np = batch_embeddings.cpu().numpy()
    
    # Save embeddings and IDs together
    save_embeddings_with_ids(batch_embeddings_np, batch_ids, batch_start // batch_size)
    
    # Clear memory (if necessary)
    del batch_embeddings, batch_embeddings_np
    torch.cuda.empty_cache()  # If using GPU