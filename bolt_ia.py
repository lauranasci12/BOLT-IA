import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# --- ESTILO MIGUEL GENEROSO ---
st.set_page_config(page_title="BOLT IA - Elite", page_icon="💰")
st.markdown("""
    <style>
    .stApp { background-color: #001233; color: #FFFFFF; }
    .stTextInput>div>div>input { background-color: #001f4d; color: #ffd700; border: 2px solid #d4af37; border-radius: 15px; }
    .stButton>button { background-color: #d4af37; color: #001233; border-radius: 20px; font-weight: bold; box-shadow: 0px 0px 15px #d4af37; width: 100%; }
    .user-msg { background-color: #003366; padding: 15px; border-radius: 15px; border-left: 5px solid #d4af37; margin-bottom: 15px; }
    .bolt-msg { background-color: #001f4d; padding: 15px; border-radius: 15px; border-right: 5px solid #ffd700; margin-bottom: 15px; color: #FFFFFF; }
    .categoria-tag { color: #ffd700; font-weight: bold; text-transform: uppercase; display: block; margin-bottom: 10px; border-bottom: 1px solid #d4af37; width: fit-content; }
    .footer { position: fixed; left: 0; bottom: 10px; width: 100%; text-align: center; color: #d4af37; font-family: 'Georgia', serif; font-style: italic; font-size: 14px; }
    [data-testid="stSidebar"] { background-color: #000814; border-right: 2px solid #d4af37; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES DE CÉREBRO ---
def carregar_dados():
    if not os.path.exists('conhecimento.txt'): return None, None, None
    df = pd.read_csv('conhecimento.txt', sep=';', names=['pergunta', 'resposta', 'categoria'], encoding='utf-8').dropna()
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['pergunta'])
    return df, vectorizer, matrix

df_bolt, vectorizer, tfidf_matrix = carregar_dados()

# --- BARRA LATERAL (ENSINO E CURIOSIDADE) ---
st.sidebar.title("💎 Central de Elite")

# Botão de Curiosidade Aleatória
if st.sidebar.button("🎲 Curiosidade do Bolt"):
    if df_bolt is not None:
        linha_aleatoria = df_bolt.sample(n=1).iloc[0]
        st.sidebar.info(f"**Você sabia?** ({linha_aleatoria['categoria']})\n\n{linha_aleatoria['resposta']}")
    else:
        st.sidebar.warning("Ainda não tenho fatos na memória.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Ensinar o Bolt")
novo_p = st.sidebar.text_input("Pergunta:")
novo_r = st.sidebar.text_area("Resposta:")
novo_c = st.sidebar.selectbox("Categoria:", ["Empreendedorismo", "Tecnologia", "Investimentos"])

if st.sidebar.button("Gravar Conhecimento"):
    if novo_p and novo_r:
        with open('conhecimento.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{novo_p};{novo_r};{novo_c}")
        st.sidebar.success("Aprendido! O site vai atualizar.")
        st.rerun()

# --- CHAT PRINCIPAL ---
st.markdown("<h1 style='text-align: center; color: #d4af37;'>⚡ BOLT IA</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    tag = "user-msg" if m["role"] == "user" else "bolt-msg"
    st.markdown(f"<div class='{tag}'><b>{m['role'].upper()}:</b><br>{m['content']}</div>", unsafe_allow_html=True)

pergunta = st.text_input("Sua dúvida de hoje:", key="input_user")

if st.button("Consultar 💸") and pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    if df_bolt is not None:
        v = vectorizer.transform([pergunta])
        sim = cosine_similarity(v, tfidf_matrix)
        if sim.max() > 0.2:
            res = df_bolt.iloc[sim.argmax()]
            txt = f"<span class='categoria-tag'>Sua pergunta está na categoria de {res['categoria']}</span>{res['resposta']}"
        else:
            txt = "🤖 Não encontrei isso. Me ensine na barra lateral!"
    else:
        txt = "⚠️ Erro nos dados."
    st.session_state.messages.append({"role": "bolt", "content": txt})
    st.rerun()

st.markdown("<div class='footer'>By Miguel Generoso</div>", unsafe_allow_html=True)
