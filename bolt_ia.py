import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- CONFIGURAÇÃO E ESTILO ---
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
    /* Estilo da Sidebar de Ensino */
    [data-testid="stSidebar"] { background-color: #000814; border-right: 2px solid #d4af37; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO PARA SALVAR NOVO CONHECIMENTO ---
def salvar_novo_conhecimento(p, r, c):
    nova_linha = f"\n{p};{r};{c}"
    with open('conhecimento.txt', 'a', encoding='utf-8') as f:
        f.write(nova_linha)
    st.sidebar.success("Conhecimento gravado com sucesso! Recarregando...")

# --- CÉREBRO DA IA ---
def carregar_inteligencia():
    if not os.path.exists('conhecimento.txt'):
        return None, None, None
    df = pd.read_csv('conhecimento.txt', sep=';', names=['pergunta', 'resposta', 'categoria'])
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['pergunta'])
    return df, vectorizer, matrix

df_bolt, vectorizer, tfidf_matrix = carregar_inteligencia()

# --- SIDEBAR: MÓDULO DE ENSINO ---
st.sidebar.title("💎 Central de Inteligência")
st.sidebar.markdown("Ensine novos conceitos ao Bolt:")
novo_p = st.sidebar.text_input("Pergunta/Termo:")
novo_r = st.sidebar.text_area("Resposta detalhada:")
novo_c = st.sidebar.selectbox("Categoria:", ["Empreendedorismo", "Tecnologia", "Investimentos"])

if st.sidebar.button("Ensinar ao Bolt 🧠"):
    if novo_p and novo_r:
        salvar_novo_conhecimento(novo_p, novo_r, novo_c)
        st.rerun()

# --- INTERFACE PRINCIPAL ---
st.markdown("<h1 style='text-align: center; color: #d4af37;'>⚡ BOLT IA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ffd700;'>By Miguel Generoso</p>", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    div_class = "user-msg" if msg["role"] == "user" else "bolt-msg"
    icon = "👤 Você" if msg["role"] == "user" else "⚡ Bolt"
    st.markdown(f"<div class='{div_class}'><b>{icon}:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

pergunta_user = st.text_input("O que deseja consultar?", placeholder="Ex: O que é inflação?")

if st.button("Consultar Bolt 💸") and pergunta_user:
    st.session_state.messages.append({"role": "user", "content": pergunta_user})

    if df_bolt is not None:
        query_vec = vectorizer.transform([pergunta_user])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        idx = similarity.argmax()
        score = similarity.max()

        if score > 0.2:
            categoria = df_bolt.iloc[idx]['categoria']
            resposta = df_bolt.iloc[idx]['resposta']
            texto_final = f"<span class='categoria-tag'>Sua pergunta está na categoria de {categoria}</span>{resposta}"
        else:
            texto_final = "🤖 Meus bancos de dados ainda não possuem isso. Use a **Central de Inteligência** ao lado para me ensinar!"
    
    st.session_state.messages.append({"role": "bolt", "content": texto_final})
    st.rerun()

st.markdown("<div class='footer'>By Miguel Generoso</div>", unsafe_allow_html=True)