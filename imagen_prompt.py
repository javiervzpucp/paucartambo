# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:28:23 2024

@author: jveraz
"""

import openai
import os
import pandas as pd
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Función para generar el prompt de análisis de imágenes
def generate_prompt(img_url, title):
    prompt = f"""
Analiza la imagen proporcionada en el enlace y proporciona palabras clave que se relacionen con los siguientes temas:
1. **Significado Cultural**: Identifica símbolos, objetos o figuras que reflejen tradiciones culturales andinas, especialmente aquellas relevantes para la celebración de la Mamacha Carmen en Paucartambo. Busca elementos como máscaras, danzantes, altares y objetos rituales.
2. **Elementos Rituales**: Identifica elementos que puedan representar prácticas religiosas o espirituales, como vestimentas, altares o movimientos específicos de danza. Enfócate en aspectos que resalten el sincretismo o la fusión de tradiciones indígenas y católicas.
3. **Identidad y Transformación**: Observa objetos o posturas que signifiquen identidad o transformación, como máscaras, poses de danza o gestos simbólicos. Describe cómo estos elementos pueden representar la fusión de identidades (por ejemplo, identidad mestiza) o transformación en un contexto ritual.
4. **Contexto Social e Histórico**: Busca indicaciones de roles sociales o símbolos históricos, como vestimentas específicas de diferentes comparsas (por ejemplo, Qhapac Qolla, Qhapac Negro). Nota cualquier referencia que simbolice eventos históricos o conexiones culturales entre diferentes regiones andinas.

Proporciona una lista detallada de palabras clave basadas en la imagen, con énfasis en referencias culturales e históricas, símbolos de devoción y elementos de transformación.
URL de la imagen: {img_url}
Título o descripción de la imagen: {title}
"""
    return prompt

# Función principal para analizar cada imagen utilizando la API de OpenAI
def analyze_image(img_url, title):
    prompt = generate_prompt(img_url, title)
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500
        )
        
        keywords = response.choices[0].text.strip()
        print(f"Palabras clave para '{title}':\n{keywords}\n")
    
    except openai.error.OpenAIError as e:
        print(f"Error al analizar '{title}':", e)

# Función para leer el CSV y procesar cada imagen
def process_images_from_csv(csv_path):
    # Leer el archivo CSV
    data = pd.read_csv(csv_path,delimiter=';')
    
    # Iterar sobre cada fila del CSV
    for index, row in data.iterrows():
        img_url = row['imagen']
        title = row['descripción']
        analyze_image(img_url, title)

csv_path = "imagenes.csv"  # Asegúrate de tener el archivo CSV en la misma carpeta
process_images_from_csv(csv_path)
