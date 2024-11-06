# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:26:55 2024

@author: jveraz
"""

import networkx as nx
import json
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv

###########
## GRAFO ##
###########

# leemos el grafo
with open("networkx_graph/archivos/saved_graph.json", "r") as file:
    data = json.load(file)

# de vuelta a networkx
nx_graph = nx.node_link_graph(data)

################
## EMBEDDINGS ##
################

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model for answer generation
#llm = ChatOpenAI(model_name="gpt-4")

# Initialize a dictionary to store embeddings
node_embeddings = {}

# Generate embeddings for each node based on its content
for node_id, node_data in nx_graph.nodes(data=True):
    content = node_data.get("content", "")  # Assuming each node has a 'content' field
    if content:
        # Generate embedding using OpenAI's API
        response = openai.Embedding.create(
            input=content,
            model="text-embedding-ada-002"  # Replace with appropriate embedding model
        )
        embedding = response['data'][0]['embedding']
        node_embeddings[node_id] = np.array(embedding)  # Store as a numpy array

##################
## VECTOR STORE ##
##################   


# Convert embeddings to a numpy array for FAISS
embedding_matrix = np.array(list(node_embeddings.values())).astype("float32")

# Check the dimensionality of embeddings
embedding_dim = embedding_matrix.shape[1] if embedding_matrix.size > 0 else 0
if embedding_dim == 0:
    raise ValueError("Embedding matrix is empty or embeddings have zero dimension.")

# Create a FAISS index with the correct dimensionality
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)  # Add the embeddings to the FAISS index

# Map node IDs to embedding positions for retrieval
node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_embeddings.keys())}
index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}

print("FAISS index created successfully with {} embeddings.".format(index.ntotal))


#########
## RAG ##
#########

def rag_query(query, top_k=5):
    try:
        # Step 1: Generate an embedding for the query using OpenAI's API
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response['data'][0]['embedding']).astype("float32")
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return None

    try:
        # Step 2: Retrieve top-k most similar nodes from FAISS
        distances, indices = index.search(np.array([query_embedding]), top_k)
        retrieved_nodes = [index_to_node_id[idx] for idx in indices[0]]
    except Exception as e:
        print(f"Error retrieving nodes from FAISS index: {e}")
        return None

    try:
        # Step 3: Concatenate content from the retrieved nodes as context
        context = "\n".join(
            nx_graph.nodes[node_id]["content"]
            for node_id in retrieved_nodes
            if "content" in nx_graph.nodes[node_id]
        )
    except KeyError as e:
        print(f"Error accessing content for node {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during context construction: {e}")
        return None

    try:
        # Step 4: Use the LLM to answer the query based on the retrieved context
        messages = [
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
        ]

        # Use OpenAI ChatCompletion API to generate response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150
        )
        answer = response.choices[0].message['content'].strip()

    except Exception as e:
        print(f"Error generating response from LLM: {e}")
        return None

    return answer

# Example usage
response = rag_query("Qué es la danza Chunchada?")
print(response)


