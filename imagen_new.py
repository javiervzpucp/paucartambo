import os
from dotenv import load_dotenv
from IPython.display import Image, display
import pandas as pd
from openai import OpenAI

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configurar la clave de la API de OpenAI desde el archivo .env
api_key = os.getenv("OPENAI_API_KEY")

# Inicializar cliente de OpenAI
client = OpenAI()

# Cargar el conjunto de datos
dataset_path = "imagenes/imagenes.csv"
df = pd.read_csv(dataset_path, delimiter=';')

# Prompt para generar descripciones para escenas culturales y rituales
describe_system_prompt = '''
Eres un sistema especializado en generar descripciones para escenas culturales y rituales andinas, especialmente aquellas relacionadas con la festividad de la Mamacha Carmen en Paucartambo.

Se te proporcionará una imagen y un título que describe la escena. Tu tarea es describir el tema principal de la imagen, brindando detalles pero manteniéndote conciso.

Puedes describir claramente el tipo de escena, los símbolos culturales, elementos rituales, y el contexto histórico o social si son identificables.

Si en la imagen se muestran varios elementos, usa el título para entender qué aspecto de la escena debes describir.
'''

def generate_combined_examples(df):
    # Agregar una columna 'generated_description' al DataFrame si no existe
    if 'generated_description' not in df.columns:
        df['generated_description'] = None
    
    # Generar descripciones para las imágenes que no tengan una descripción generada aún
    for index, row in df.iterrows():
        if pd.isna(row['generated_description']):
            img_url = row['imagen']
            title = row['descripción']
            description = describe_image_with_prompt(img_url, title, "")
            df.at[index, 'generated_description'] = description  # Guardar en el DataFrame

    # Guardar el DataFrame actualizado en el archivo CSV
    df.to_csv(dataset_path, sep=';', index=False)
    print("Descripciones generadas y guardadas en el archivo CSV.")

    # Crear texto con ejemplos de descripciones generadas para usar como contexto
    combined_examples = "Aquí tienes ejemplos de descripciones generadas previamente:\n\n"
    for index, row in df.iterrows():
        combined_examples += f"Título: {row['descripción']}\nDescripción: {row['generated_description']}\n\n"
    return combined_examples

def describe_image_with_prompt(img_url, title, example_descriptions):
    # Crear el prompt incluyendo el sistema prompt y ejemplos previos como contexto
    prompt = f"{describe_system_prompt}\n\n{example_descriptions}\n\nBasado en estos ejemplos, genera una descripción para la siguiente imagen:\nURL: {img_url}\nTítulo: {title}"
    
    # Generar la descripción con OpenAI
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5
    )

    return response.choices[0].text.strip()

# Generar descripciones para nuevas imágenes usando el contexto de las descripciones del CSV
def describe_user_image():
    # Obtener ejemplos combinados de descripciones previas del CSV
    example_descriptions = generate_combined_examples(df)
    
    # Pedir al usuario que ingrese la URL y el título de la imagen
    img_url = input("Ingresa la URL de la imagen: ")
    title = input("Ingresa un título o descripción breve de la imagen: ")

    # Mostrar la imagen al usuario
    display(Image(url=img_url))

    # Generar la descripción utilizando el contexto de las descripciones previas
    img_description = describe_image_with_prompt(img_url, title, example_descriptions)
    print(f"Descripción generada para la imagen:\n{img_description}\n")

# Procesar imágenes en el CSV y agregar descripciones generadas
generate_combined_examples(df)

# Permitir que el usuario cargue una imagen nueva
describe_user_image()
