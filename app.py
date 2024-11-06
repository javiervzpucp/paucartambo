# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:09:22 2024

@author: jveraz
"""

## credenciales

import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import GraphQAChain
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
api_key = os.getenv("OPENAI_API_KEY")

# Inicializar la conexión a Neo4j
#graph = Neo4jGraph()

## LLMs

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

llm_transformer = LLMGraphTransformer(llm=llm)

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

#graph.add_graph_documents(graph_documents)

## cruzamos con Networkx
graph_nx = NetworkxEntityGraph()

# Add nodes to the graph
for node in graph_documents[0].nodes:
    graph_nx.add_node(node.id)

# Add edges to the graph
for edge in graph_documents[0].relationships:
    graph_nx._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )

# Interrogamos a nuestro grafo
chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph_nx, 
    verbose=True
)

question = """Who is Marie Curie?"""
chain.run(question)