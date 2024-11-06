# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:00:25 2024

@author: jveraz
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
import json
import matplotlib.pyplot as plt

#############################
## Conexiones con las APIs ##
#############################

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
api_key = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

############
## Textos ##
############
    
# Leer el archivo 'video.txt' y procesar el texto
with open("archivos/video_processed.txt", "r", encoding="utf-8") as file:
    lines = file.read()

###########
## Grafo ##
###########

# documentos
documents = [Document(page_content=lines)]

# inicializar LLMGraphTransformer 
graph_transformer = LLMGraphTransformer(llm=llm)

# generar el grafo a partir de los documentos
graph = graph_transformer.convert_to_graph_documents(documents)

# grafo networkx
nx_graph = nx.Graph()

# nodos
for node in graph[0].nodes:
    node_id = dict(node)['id']
    if node_id:  # Ensure the node has a valid ID
        nx_graph.add_node(node_id)
        
# aristas
for edge in graph[0].relationships:
    source = dict(dict(edge)['source'])['id']
    target = dict(dict(edge)['target'])['id']
    #attributes = dict(dict(edge))['type']
    if source and target:
        nx_graph.add_edge(source, target)#, **attributes)


# guardamos en json
data = nx.node_link_data(nx_graph)  # Convert to dictionary format suitable for JSON
with open("archivos/saved_graph.json", "w") as file:
    json.dump(data, file)


# leemos
with open("archivos/saved_graph.json", "r") as file:
    data = json.load(file)

# de vuelta a networkx
nx_graph = nx.node_link_graph(data)

# layout
pos = nx.spring_layout(nx_graph)  # Spring layout positions

# figura
plt.figure(figsize=(10, 10))  # Set the figure size

# aristas
nx.draw_networkx_edges(nx_graph, pos, edge_color="k", width=0.75)

# nodos
nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="gold")

# labels
nx.draw_networkx_labels(nx_graph, pos, font_size=7, font_weight="bold")

# guardamos
plt.savefig("archivos/graph.png", format="PNG")
plt.show()  

