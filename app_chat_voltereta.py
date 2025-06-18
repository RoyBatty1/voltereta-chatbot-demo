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

# --- DESCARGA DEL ÍNDICE FAISS DESDE GCS ---
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
        st.stop()

# --- CARGA DEL ÍNDICE FAISS ---
index = faiss.read_index(index_file)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- CARGA DE GOOGLE SHEETS ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
with open("credentials.json") as f:
    creds_json = json.load(f)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)

client_gs = gspread.authorize(creds)
sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/1xyijzryEyTp4vBzuDg4CPDMsvHX-E9PUvipXJiG-gPU/edit#gid=0")
data = sheet.get_worksheet(0).get_all_records()

# --- FUNCIÓN DE OVERRIDE ---
def check_sheet_override(question):
    q_lower = question.lower()
    for row in data:
        if row.get("Pregunta", "").strip().lower() in q_lower:
            return row.get("Respuesta", "").strip()
    return None

# --- INTERFAZ DE USUARIO ---
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
