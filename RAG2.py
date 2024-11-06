# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:10:37 2024

@author: jveraz
"""

import networkx as nx
import json
import openai
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.Embedding.create(
        input="What are the contributions of Marie Curie?",
        model="text-embedding-ada-002"
    )
    print(response)
except Exception as e:
    print(f"Error: {e}")
