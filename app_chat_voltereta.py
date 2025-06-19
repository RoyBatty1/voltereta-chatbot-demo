import streamlit as st
import openai
import faiss
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === CONFIGURACIÓN ===
openai.api_key = "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"
index_path = "voltereta_index.faiss"
metadata_path = "voltereta_metadata.json"
model = SentenceTransformer("all-MiniLM-L6-v2")

# === CARGA FAISS & METADATA ===
index = faiss.read_index(index_path)
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === GOOGLE SHEET (Overrides) ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(credentials)
sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1xyijzryEyTp4vBzuDg4CPDMsvHX-E9PUvipXJiG-gPU").sheet1

def buscar_override(pregunta):
    registros = sheet.get_all_records()
    for row in registros:
        if row.get("pregunta", "").strip().lower() == pregunta.strip().lower():
            return row.get("respuesta", "")
    return None

# === FILTRADO FAISS: Prioridad a contenido reciente ===
def filtrar_por_fecha_y_relevancia(I, D, metadata, max_meses=12, top_k=5):
    ahora = datetime.now()
    limite = ahora - timedelta(days=max_meses * 30)

    recientes = []
    antiguos = []

    for idx, dist in zip(I[0], D[0]):
        item = metadata[idx]
        fecha_raw = item.get("date", "")
        try:
            fecha = datetime.fromisoformat(fecha_raw[:10])
        except:
            fecha = datetime.min

        resultado = {
            "idx": idx,
            "dist": dist,
            "metadata": item
        }

        if fecha >= limite:
            recientes.append(resultado)
        else:
            antiguos.append(resultado)

    seleccionados = recientes if recientes else antiguos
    seleccionados_ordenados = sorted(seleccionados, key=lambda r: r["dist"])
    return seleccionados_ordenados[:top_k]

# === CONSULTA FAISS ===
def buscar_faiss(pregunta, top_k=5):
    emb = model.encode(pregunta).astype("float32")
    D, I = index.search(np.array([emb]), 20)
    resultados = filtrar_por_fecha_y_relevancia(I, D, metadata, max_meses=12, top_k=top_k)

    contexto = []
    for r in resultados:
        meta = r["metadata"]
        contexto.append(f"{meta.get('subject', '')}\n{meta.get('from', '')}\n{meta.get('date', '')}\n")

    return "\n".join(contexto).strip()

# === GPT Fallback ===
def responder_con_gpt(pregunta, contexto=""):
    prompt = f"""
Eres el asistente del restaurante Voltereta. Usa la siguiente información para ayudar al usuario si es relevante:

{contexto}

Pregunta del usuario: {pregunta}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Eres un asistente profesional de Voltereta."},
                  {"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# === INTERFAZ STREAMLIT ===
st.set_page_config(page_title="Voltereta Bot", page_icon="🌮")
st.title("🤖 Chatbot Voltereta")

user_input = st.text_input("Haz tu pregunta:")

if user_input:
    # Paso 1: Override
    respuesta_override = buscar_override(user_input)
    if respuesta_override:
        st.success(f"📌 {respuesta_override}")
    else:
        # Paso 2: Buscar contexto en FAISS
        contexto = buscar_faiss(user_input)
        # Paso 3: Preguntar a GPT con contexto
        respuesta = responder_con_gpt(user_input, contexto)
        st.info(respuesta)
