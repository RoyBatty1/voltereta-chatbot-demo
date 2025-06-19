import streamlit as st
import openai, faiss, json, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from bs4 import BeautifulSoup
import urllib.request

# === CONFIGURACIÓN ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === CARGA FORZADA DEL ÍNDICE FAISS DESDE GCS ===
INDEX_URL = "https://storage.googleapis.com/voltereta-chatbot-assets/index_streamlit_compatible.faiss"
INDEX_PATH = "voltereta_index.faiss"
METADATA_PATH = "voltereta_metadata.json"

# Descargar siempre desde GCS (sin comprobar si ya existe)
try:
    st.write("📥 Descargando índice FAISS desde GCS...")
    urllib.request.urlretrieve(INDEX_URL, INDEX_PATH)
    st.write("✅ Descargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al descargar el índice FAISS: {e}")
    st.stop()

# Cargar índice
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    st.error(f"❌ Error al cargar el índice FAISS: {e}")
    st.
