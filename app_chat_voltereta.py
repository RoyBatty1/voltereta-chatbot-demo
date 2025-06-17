import streamlit as st
from openai import OpenAI
import urllib.request
import os

index_file = "index.faiss"
file_id = "1NPW3J-coWBt_7Nts-jZkxt19F92JpdAP"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(index_file):
    print("Descargando el archivo FAISS desde Google Drive...")
    urllib.request.urlretrieve(download_url, index_file)
    print("Descarga completada.")

import faiss
index = faiss.read_index(index_file)


import pickle
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import urllib.request

if not os.path.exists("index.faiss"):
    url = "https://drive.google.com/file/d/1NPW3J-coWBt_7Nts-jZkxt19F92JpdAP/view?usp=drive_link"
    urllib.request.urlretrieve(url, "index.faiss")




# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Voltereta Chatbot", page_icon="😮")
client = OpenAI(api_key="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")

# --- FAISS SETUP ---
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")

# --- GOOGLE SHEETS SETUP ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client_gs = gspread.authorize(creds)
sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/13Xl5wqYv1zPWJhAGeVfW32el-_Bm3N25bO9scjQTKas/edit#gid=0")
data = sheet.get_worksheet(0).get_all_records()

# --- FUNCIÓN DE BÚSQUEDA EN SHEET ---
def check_sheet_override(question):
    q_lower = question.lower()
    for row in data:
        if row["Pregunta"].strip().lower() in q_lower:
            return row["Respuesta"].strip()
    return None

# --- UI ---
st.markdown("""
    <h1 style='text-align: center; color: #BDA892;'>💭 Chatbot Voltereta</h1>
    <p style='text-align: center;'>Tu compañero de viaje por el mundo ✨</p>
""", unsafe_allow_html=True)

query = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Pregúntame sobre reservas, destinos o alergias...")

if query:
    with st.spinner("Explorando el mundo para darte la mejor respuesta..."):
        # Paso 1: Verificar override en Sheet
        respuesta_fija = check_sheet_override(query)

        if respuesta_fija:
            st.success(respuesta_fija)
        else:
            # Paso 2: Recuperar contexto con FAISS
            embedding = model.encode([query])
            D, I = index.search(embedding, k=3)
            contexto = "\n".join([f"{i+1}. Resultado relacionado {idx}" for i, idx in enumerate(I[0])])

            # Paso 3: Llamar a OpenAI
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Actúa como un asistente de Voltereta: cercano, amable, experto en viajes y experiencias del restaurante."},
                    {"role": "user", "content": f"Pregunta: {query}\nContexto: {contexto}"}
                ]
            )
            respuesta = completion.choices[0].message.content.strip()
            st.success(respuesta)
