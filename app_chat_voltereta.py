import streamlit as st
import openai, faiss, json, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from bs4 import BeautifulSoup

# === CONFIG ===
openai.api_key = "TU_API_KEY"
index = faiss.read_index("voltereta_index.faiss")
with open("voltereta_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
gc = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope))
sheet = gc.open_by_url("https://docs.google.com/spreadsheets/...").sheet1

LOG_PATH = "logs.txt"

def log(pregunta, fuente):
    now = datetime.now().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{now}\t{fuente}\t{pregunta}\n")

def buscar_override(p):
    for row in sheet.get_all_records():
        if row.get("pregunta","").strip().lower() == p.strip().lower():
            return row.get("respuesta","")
    return None

def filtrar_por_fecha_y_relevancia(I, D, metadata, max_meses=12, top_k=5):
    ahora = datetime.now()
    limite = ahora - timedelta(days=30*max_meses)
    recientes, antiguos = [], []
    for idx, dist in zip(I[0], D[0]):
        meta = metadata[idx]
        fecha = None
        try:
            fecha = datetime.fromisoformat(meta.get("date","")[:10])
        except:
            fecha = datetime.min
        rec = {"idx":idx,"dist":dist,"metadata":meta}
        (recientes if fecha>=limite else antiguos).append(rec)
    sel = recientes if recientes else antiguos
    sel = sorted(sel, key=lambda r:r["dist"])[:top_k]
    return sel

def buscar_faiss(p):
    emb = model.encode(p).astype("float32")
    D,I = index.search(np.array([emb]),20)
    sel = filtrar_por_fecha_y_relevancia(I,D,metadata)
    return "\n".join([f"{r['metadata'].get('subject','')} — {r['metadata'].get('date','')}" for r in sel])

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
            texto += "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip()) + "\n"
        except Exception as e:
            pass
    return texto.strip()

SCRAP_CONTEXT = scrap_voltereta()

def responder_con_gpt(p, contexto=""):
    prompt = f"""
Eres el asistente de Voltereta. Usa este contexto de su web y FAQs para ayudar al usuario:

{SCRAP_CONTEXT}

Información relevante desde FAISS:
{contexto}

Usuario: {p}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"Eres un asistente profesional de Voltereta."},
            {"role":"user","content":prompt}
        ],
        temperature=0.4
    )
    return resp.choices[0].message.content.strip()

# === UI Streamlit ===
st.set_page_config(page_title="Voltereta Bot", page_icon="🌮")
st.title("🤖 Chatbot Voltereta")

user_input = st.text_input("Tu pregunta:")

if user_input:
    # Step 1: override
    ov = buscar_override(user_input)
    if ov:
        log(user_input, "OVERRIDE")
        st.success(ov)
    else:
        ctx = buscar_faiss(user_input)
        resp = responder_con_gpt(user_input, ctx)
        fuente = "FAISS+GPT" if ctx else "GPT puro"
        log(user_input, fuente)
        st.info(resp)
