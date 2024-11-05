# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:09:22 2024

@author: jveraz
"""

## credenciales

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
api_key = os.getenv("OPENAI_API_KEY")

# Astra DB
client = DataAPIClient("YOUR_TOKEN")
db = client.get_database_by_api_endpoint(
  "https://8d471473-2e08-47ff-8bdb-9be2871addcb-us-east-2.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")

