# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:09:22 2024

@author: jveraz
"""

## credenciales

import os
from dotenv import load_dotenv

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
api_key = os.getenv("OPENAI_API_KEY")

