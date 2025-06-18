import streamlit as st
from openai import OpenAI
import os
import requests
import faiss
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="Voltereta Chatbot", page_icon="🧳")
client = OpenAI(api_key="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")

# --- DESCARGA DEL ÍNDICE FAISS COMPATIBLE DESDE GCS CON REQUESTS ---
index_file = "index_streamlit_compatible.faiss"
download_url = "https://storage.googleapis.com/voltereta-chatbot-assets/index_streamlit_compatible.faiss"

if not os.path.exists(index_file):
    st.write("📥 Descargando el archivo FAISS desde GCS con requests...")
    r = requests.get(download_url)
    if r.status_code == 200:
        with open(index_file, "wb") as f:
            f.write(r.content)
        st.write("✅ Descarga completada.")
    else:
        st.error(f"❌ Error al descargar el archivo FAISS: {r.status_code}")

if os.path.exists(index_file):
    st.write("📦 Tamaño del índice FAISS descargado:", os.path.getsize(index_file), "bytes")
    index = faiss.read_index(index_file)
    model = SentenceTransformer("all-MiniLM-L6-v2")
else:
    st.error("No se encontró el archivo de índice FAISS.")

# --- CARGA DE GOOGLE SHEETS ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
with open("credentials.json") as f:
    creds_json = json.load(f)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)

client_gs = gspread.authorize(creds)
sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/1xyijzryEyTp4vBzuDg4CPDMsvHX-E9PUvipXJiG-gPU/edit?gid=0")
data = sheet.get_worksheet(0).get_all_records()

# --- FUNCIÓN DE OVERRIDE ---
def check_sheet_override(question):
    q_lower = question.lower()
    for row in data:
        if row["Pregunta"].strip().lower() in q_lower:
            return row["Respuesta"].strip()
    return None

# --- UI ---
st.markdown("""
    <h1 style='text-align: center; color: #BDA892;'>💬 Chatbot Voltereta</h1>
    <p style='text-align: center;'>Tu compañero de viaje ✨</p>
""", unsafe_allow_html=True)

query = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Pregúntame sobre reservas, destinos o alergias...")

if query:
    with st.spinner("Explorando el mundo para darte la mejor respuesta..."):
        respuesta_fija = check_sheet_override(query)

        if respuesta_fija:
            st.success(respuesta_fija)
        else:
            embedding = model.encode([query])
            D, I = index.search(embedding, k=3)
            contexto = "\n".join([f"{i+1}. Resultado relacionado {idx}" for i, idx in enumerate(I[0])])

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Actúa como un asistente de Voltereta: cercano, amable, experto en viajes y experiencias del restaurante."},
                    {"role": "user", "content": f"Pregunta: {query}\nContexto: {contexto}"}
                ]
            )
            respuesta = completion.choices[0].message.content.strip()
            st.success(respuesta)
