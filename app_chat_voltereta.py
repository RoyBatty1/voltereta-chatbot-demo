import streamlit as st
import openai, faiss, json, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from bs4 import BeautifulSoup
import os
import urllib.request

# === CONFIGURACIÓN ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === DESCARGA FAISS DESDE CLOUD STORAGE ===
INDEX_URL = "https://storage.googleapis.com/voltereta-chatbot-assets/index_streamlit_compatible.faiss"
INDEX_PATH = "voltereta_index.faiss"
METADATA_PATH = "voltereta_metadata.json"

if not os.path.exists(INDEX_PATH):
    urllib.request.urlretrieve(INDEX_URL, INDEX_PATH)

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# === GOOGLE SHEET (Overrides) ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials_dict = st.secrets["gcp_service_account"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
gc = gspread.authorize(credentials)
sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1xyijzryEyTp4vBzuDg4CPDMsvHX-E9PUvipXJiG-gPU").sheet1

LOG_PATH = "logs.txt"

# === FUNCIONES AUXILIARES ===

def log(pregunta, fuente):
    now = datetime.now().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{now}\t{fuente}\t{pregunta}\n")

def buscar_override(p):
    for row in sheet.get_all_records():
        if row.get("pregunta", "").strip().lower() == p.strip().lower():
            return row.get("respuesta", "")
    return None

def filtrar_por_fecha_y_relevancia(I, D, metadata, max_meses=12, top_k=5):
    ahora = datetime.now()
    limite = ahora - timedelta(days=30 * max_meses)
    recientes, antiguos = [], []
    for idx, dist in zip(I[0], D[0]):
        meta = metadata[idx]
        try:
            fecha = datetime.fromisoformat(meta.get("date", "")[:10])
        except:
            fecha = datetime.min
        entrada = {"idx": idx, "dist": dist, "metadata": meta}
        (recientes if fecha >= limite else antiguos).append(entrada)
    seleccionados = recientes if recientes else antiguos
    return sorted(seleccionados, key=lambda r: r["dist"])[:top_k]

def buscar_faiss(p):
    emb = model.encode(p).astype("float32")
    D, I = index.search(np.array([emb]), 20)
    seleccionados = filtrar_por_fecha_y_relevancia(I, D, metadata)
    return "\n".join([
        f"{r['metadata'].get('subject', '')} — {r['metadata'].get('date', '')}"
        for r in seleccionados
    ])

def scrap_voltereta():
    urls = [
        "https://www.volteretarestaurante.com/es/",
        "https://www.volteretarestaurante.com/en/FAQ/"
    ]
    texto = ""
    for u in urls:
        try:
            resp = requests.get(u, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.select("p")
            texto += "\n".join(
                p.get_text().strip()
                for p in paragraphs if p.get_text().strip()
            ) + "\n"
        except Exception:
            pass
    return texto.strip()

SCRAP_CONTEXT = scrap_voltereta()

def responder_con_gpt(p, contexto=""):
    prompt = f"""
Eres el asistente de Voltereta. Usa este contexto sacado de su web y FAQs para ayudar al usuario:

{SCRAP_CONTEXT}

También considera esta información relevante desde FAISS (si aplica):
{contexto}

Pregunta del usuario: {p}
"""
    respuesta = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente profesional de Voltereta."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return respuesta.choices[0].message.content.strip()

# === INTERFAZ STREAMLIT ===
st.set_page_config(page_title="Voltereta Bot", page_icon="🌮")
st.title("🤖 Chatbot Voltereta")

user_input = st.text_input("Haz tu pregunta:")

if user_input:
    respuesta_override = buscar_override(user_input)
    if respuesta_override:
        log(user_input, "OVERRIDE")
        st.success(respuesta_override)
    else:
        contexto = buscar_faiss(user_input)
        respuesta = responder_con_gpt(user_input, contexto)
        log(user_input, "FAISS+GPT" if contexto else "GPT puro")
        st.info(respuesta)
