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

SYSTEM_PROMPT = (
    "Eres un experto certificado en AWS Cloud Practitioner. "
    "Tu rol es ayudar a estudiantes a prepararse para el examen "
    "AWS Certified Cloud Practitioner (CLF-C02). "
    "Responde siempre en espanol. "
    "Da explicaciones claras con ejemplos practicos. "
    "Si el usuario hace una pregunta tipo examen, evalua su respuesta "
    "y explica por que es correcta o incorrecta. "
    "Manten un tono motivador y educativo. "
    "Temas: servicios AWS, modelo de responsabilidad compartida, "
    "facturacion, seguridad, arquitectura en la nube."
)

def invoke_bedrock(messages):
    client = get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
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

st.title("Asistente AWS Cloud Practitioner")
st.caption("Preparacion CLF-C02 · Powered by Amazon Bedrock")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hola! Soy tu asistente para la certificacion "
                "AWS Cloud Practitioner (CLF-C02).\n\n"
                "Puedo ayudarte con:\n"
                "- Explicar servicios de AWS\n"
                "- Resolver dudas sobre el examen\n"
                "- Hacerte preguntas tipo examen para practicar\n\n"
                "En que tema quieres empezar hoy?"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre AWS..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                reply = invoke_bedrock(st.session_state.messages)
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                st.error("Error al conectar con Bedrock: " + str(e))