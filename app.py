import os
import json
import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Asistente AWS Cloud Practitioner",
    page_icon="☁️",
    layout="centered",
)

@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

TEMAS = {
    "Generative AI (AI Practitioner)": (
        "Eres un experto en Inteligencia Artificial Generativa. "
        "Ayudas a estudiantes a prepararse para la certificacion AWS AI Practitioner. "
        "Explica conceptos como LLMs, modelos fundacionales, embeddings, RAG y casos de uso. "
        "Responde siempre en espanol con ejemplos practicos y tono educativo."
    ),
    "Amazon SageMaker (AI Practitioner)": (
        "Eres un experto en Amazon SageMaker. "
        "Ayudas a estudiantes a prepararse para la certificacion AWS AI Practitioner. "
        "Explica conceptos como entrenamiento, despliegue, pipelines, Studio y Canvas. "
        "Responde siempre en espanol con ejemplos practicos y tono educativo."
    ),
    "RAG y uso de LLMs": (
        "Eres un experto en RAG (Retrieval-Augmented Generation) y uso de LLMs. "
        "Explica como funcionan los sistemas RAG, vectores, bases de datos vectoriales, "
        "y como se integran los LLMs en aplicaciones reales con Amazon Bedrock. "
        "Responde siempre en espanol con ejemplos practicos y tono educativo."
    ),
    "Cloud Computing (Cloud Practitioner)": (
        "Eres un experto certificado en AWS Cloud Practitioner. "
        "Ayudas a estudiantes a prepararse para el examen CLF-C02. "
        "Explica servicios AWS, modelo de responsabilidad compartida, "
        "facturacion, seguridad y arquitectura en la nube. "
        "Responde siempre en espanol con ejemplos practicos y tono educativo."
    ),
    "Certificacion Scrum": (
        "Eres un experto certificado en Scrum y metodologias agiles. "
        "Ayudas a estudiantes a prepararse para certificaciones Scrum (PSM, CSM). "
        "Explica roles, eventos, artefactos, valores y principios agiles. "
        "Responde siempre en espanol con ejemplos practicos y tono educativo."
    ),
}

def invoke_bedrock(messages, system_prompt):
    client = get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "system": system_prompt,
        "messages": messages,
    })
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ── Barra lateral ────────────────────────────────────────────────────
with st.sidebar:
    st.title("Temas disponibles")
    st.caption("Selecciona el tema en el que quieres practicar")
    tema_seleccionado = st.radio(
        label="Tema",
        options=list(TEMAS.keys()),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Powered by Amazon Bedrock")

# ── Titulo principal ─────────────────────────────────────────────────
st.title("Asistente de Certificaciones")
st.caption(f"Tema actual: {tema_seleccionado}")

# ── Resetear historial si cambia el tema ────────────────────────────
if "tema_actual" not in st.session_state:
    st.session_state.tema_actual = tema_seleccionado

if st.session_state.tema_actual != tema_seleccionado:
    st.session_state.tema_actual = tema_seleccionado
    st.session_state.messages = []

# ── Historial de conversacion ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    bienvenida = (
        f"Hola! Soy tu asistente especializado en **{tema_seleccionado}**.\n\n"
        "Puedo ayudarte con:\n"
        "- Explicar conceptos clave\n"
        "- Resolver tus dudas\n"
        "- Hacerte preguntas tipo examen para practicar\n\n"
        "En que quieres empezar?"
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": bienvenida}
    )

# ── Mostrar historial ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input del usuario ────────────────────────────────────────────────
if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                system_prompt = TEMAS[tema_seleccionado]
                reply = invoke_bedrock(st.session_state.messages, system_prompt)
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                st.error("Error al conectar con Bedrock: " + str(e))