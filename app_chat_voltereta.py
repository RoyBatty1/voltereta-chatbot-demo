import streamlit as st
from openai import OpenAI
import os, requests, faiss, json, io
from sentence_transformers import SentenceTransformer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# --- CONFIG GENERAL ---
st.set_page_config(page_title="Voltereta Chatbot", page_icon="🧳")

def verificar_secrets_expuestos():
    texto = st.session_state.get("_streamlit_messages", [])
    for s in st.secrets:
        if isinstance(st.secrets[s], str) and st.secrets[s][:4] in texto:
            st.error(f"❌ Se detectó un intento de mostrar parte del secreto '{s}' en pantalla.")
            st.stop()
verificar_secrets_expuestos()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- FAISS ---
index_file = "index_streamlit_compatible.faiss"
download_url = "https://storage.googleapis.com/voltereta-chatbot-assets/index_streamlit_compatible.faiss"

if not os.path.exists(index_file):
    st.write("📥 Descargando FAISS desde GCS...")
    r = requests.get(download_url)
    if r.status_code == 200:
        with open(index_file, "wb") as f:
            f.write(r.content)
        st.write("✅ Descarga completada.")
    else:
        st.error(f"❌ Error al descargar FAISS: {r.status_code}")
        st.stop()

index = faiss.read_index(index_file)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- GOOGLE SHEETS desde st.secrets ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["google_service_account"], scope
)
client_gs = gspread.authorize(creds)
sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/1xyijzryEyTp4vBzuDg4CPDMsvHX-E9PUvipXJiG-gPU/edit#gid=0")
data = sheet.get_worksheet(0).get_all_records()

# --- OVERRIDE ---
def check_sheet_override(question):
    q_lower = question.lower()
    for row in data:
        if row.get("Pregunta", "").strip().lower() in q_lower:
            return row.get("Respuesta", "").strip()
    return None

# --- SCRAPING DE VOLTERETA ---
@st.cache_data(ttl=3600)
def get_voltereta_context():
    urls = {
        "restaurantes": "https://www.volteretarestaurante.com/es/",
        "faq": "https://www.volteretarestaurante.com/en/FAQ/",
        "experiencia": "https://www.volteretarestaurante.com/en/the-experience/"
    }

    secciones = []

    for nombre, url in urls.items():
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.content, "html.parser")
            textos = [p.get_text().strip() for p in soup.find_all("p")]
            relevantes = [t for t in textos if 40 < len(t) < 500]
            bloque = f"📌 Sección: {nombre}\n" + "\n".join(relevantes[:10])
            secciones.append(bloque)
        except Exception as e:
            secciones.append(f"[Error al cargar {nombre}: {e}]")

    # PDFs de cartas
    try:
        r = requests.get(urls["experiencia"], timeout=5)
        soup = BeautifulSoup(r.content, "html.parser")
        pdf_urls = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(".pdf")]

        for i, pdf_url in enumerate(pdf_urls):
            try:
                if not pdf_url.startswith("http"):
                    pdf_url = "https://www.volteretarestaurante.com" + pdf_url
                pdf_resp = requests.get(pdf_url, timeout=5)
                pdf_data = fitz.open(stream=pdf_resp.content, filetype="pdf")
                texto = "\n".join([page.get_text() for page in pdf_data])
                texto_limpio = "\n".join([l.strip() for l in texto.split("\n") if len(l.strip()) > 30])
                secciones.append(f"📌 Sección: carta PDF {i+1}\n{texto_limpio[:1000]}...")
            except Exception as e:
                secciones.append(f"[Error al procesar PDF {i+1}: {e}]")
    except Exception as e:
        secciones.append(f"[Error al detectar PDFs: {e}]")

    return "\n\n".join(secciones)

# --- UI ---
st.markdown("""
    <h1 style='text-align: center; color: #BDA892;'>💬 Chatbot Voltereta</h1>
    <p style='text-align: center;'>Tu compañero de viaje ✨</p>
""", unsafe_allow_html=True)

query = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Pregúntame sobre reservas, menús, horarios...")

if query:
    with st.spinner("Explorando el mundo para darte la mejor respuesta..."):
        respuesta_fija = check_sheet_override(query)

        if respuesta_fija:
            st.success(respuesta_fija)
        else:
            embedding = model.encode([query])
            D, I = index.search(embedding, k=3)

            faiss_contexto = "\n".join([f"{i+1}. Resultado relacionado {idx}" for i, idx in enumerate(I[0])])
            web_contexto = get_voltereta_context()

            contexto = f"""
[Información extraída del índice FAISS]
{faiss_contexto}

[Contenido desde la web y cartas de Voltereta]
{web_contexto}
"""

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Actúa como un asistente de Voltereta: cercano, amable, experto en viajes y experiencias del restaurante."},
                    {"role": "user", "content": f"Pregunta: {query}\nContexto: {contexto}"}
                ]
            )
            respuesta = completion.choices[0].message.content.strip()
            st.success(respuesta)
