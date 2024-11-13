# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:34:42 2024

@author: jveraz
"""

import os
from dotenv import load_dotenv
from IPython.display import Image, display
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI()

# Loading dataset
dataset_path =  "imagenes.csv"
df = pd.read_csv(dataset_path,delimiter=';')
print(df.head())

system_prompt = '''
    Eres un agente especializado en etiquetar imágenes de escenas culturales y rituales andinas con palabras clave relevantes basadas en festividades tradicionales andinas, especialmente la celebración de la Mamacha Carmen en Paucartambo.

Se te proporcionará una imagen y un título que describe la escena. Tu objetivo es extraer palabras clave concisas y en minúsculas que reflejen temas culturales y rituales andinos.

Las palabras clave deben describir aspectos como:

    Símbolos u objetos culturales, por ejemplo: 'máscara', 'altar', 'danza', 'procesión'
    Elementos rituales, por ejemplo: 'ofrenda', 'sincretismo', 'devoción'
    Elementos de identidad y transformación, por ejemplo: 'identidad mestiza', 'ancestral', 'guardián'
    Contexto social e histórico, por ejemplo: 'qhapac qolla', 'qhapac negro', 'virgen del carmen'

Incluye solo palabras clave que sean directamente relevantes y claramente representadas en la imagen. Evita términos genéricos o demasiado amplios, a menos que contribuyan claramente a comprender la identidad cultural y ritual andina.

Devuelve las palabras clave en el formato de un arreglo de cadenas, como este: ['máscara', 'qhapac qolla', 'devoción', 'virgen del carmen']
'''

def analyze_image(img_url, title):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": title
        }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content

examples = df.iloc[:2]
print(examples)

for index, ex in examples.iterrows():
    url = ex['imagen']
    img = Image(url=url)
    display(img)
    result = analyze_image(url, ex['descripción'])
    print(result)
    print("\n\n")
    
describe_system_prompt = '''
    Eres un sistema especializado en generar descripciones para escenas culturales y rituales andinas, especialmente aquellas relacionadas con la festividad de la Mamacha Carmen en Paucartambo.

Se te proporcionará una imagen y un título que describe la escena. Tu tarea es describir el tema principal de la imagen, brindando detalles pero manteniéndote conciso.

Puedes describir claramente el tipo de escena, los símbolos culturales, elementos rituales, y el contexto histórico o social si son identificables.

Si en la imagen se muestran varios elementos, usa el título para entender qué aspecto de la escena debes describir.
    '''

def describe_image(img_url, title):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": describe_system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": title
        }
    ],
    max_tokens=300,
    )

    return response.choices[0].message.content

for index, row in examples.iterrows():
    print(f"{row['descripción'][:50]}{'...' if len(row['descripción']) > 50 else ''} - {row['imagen']} :\n")
    img_description = describe_image(row['imagen'], row['descripción'])
    print(f"{img_description}\n--------------------------\n")