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
    
    /* Estilo do Card de Instruções */
    .instrucoes {
        background-color: #001f4d;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #d4af37;
        margin-bottom: 25px;
    }
    .instrucoes h4 { color: #ffd700; margin-top: 0; }
    .instrucoes li { color: #FFFFFF; font-size: 14px; margin-bottom: 5px; }
    .exemplo-box { 
        background: #003366; 
        padding: 10px; 
        border-radius: 10px; 
        color: #ffd700; 
        font-family: monospace;
        margin-top: 10px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES DE CÉREBRO ---
def carregar_dados():
    if not os.path.exists('conhecimento.txt'): return None, None, None
    try:
        df = pd.read_csv('conhecimento.txt', sep=';', names=['pergunta', 'resposta', 'categoria'], encoding='utf-8').dropna()
        if df.empty: return None, None, None
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(df['pergunta'])
        return df, vectorizer, matrix
    except:
        return None, None, None

df_bolt, vectorizer, tfidf_matrix = carregar_dados()

# --- BARRA LATERAL (ENSINO E CURIOSIDADE) ---
st.sidebar.title("💎 Central de Elite")

if st.sidebar.button("🎲 Curiosidade do Bolt"):
    if df_bolt is not None:
        linha_aleatoria = df_bolt.sample(n=1).iloc[0]
        st.sidebar.info(f"**Você sabia?** ({linha_aleatoria['categoria']})\n\n{linha_aleatoria['resposta']}")
    else:
        st.sidebar.warning("Ainda não tenho fatos na memória.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Ensinar o Bolt")
novo_p = st.sidebar.text_input("Pergunta (ex: O que é IA?):")
novo_r = st.sidebar.text_area("Resposta explicativa:")
novo_c = st.sidebar.selectbox("Categoria:", ["Empreendedorismo", "Tecnologia", "Investimentos"])

if st.sidebar.button("Gravar Conhecimento"):
    if novo_p and novo_r:
        with open('conhecimento.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{novo_p.strip()};{novo_r.strip()};{novo_c}")
        st.sidebar.success("Aprendido! O site vai atualizar.")
        st.rerun()

# --- CONTEÚDO PRINCIPAL ---
st.markdown("<h1 style='text-align: center; color: #d4af37;'>⚡ BOLT IA</h1>", unsafe_allow_html=True)

# Card de instruções para o usuário
st.markdown("""
    <div class="instrucoes">
        <h4>Como obter a melhor resposta? ⚡</h4>
        <ul>
            <li>✅ Comece com <b>Letra Maiúscula</b> (Ex: O que é...).</li>
            <li>✅ Escreva termos técnicos corretamente (Ex: <b>LinkedIn</b>, <b>Selic</b>).</li>
            <li>✅ Sempre termine com o ponto de interrogação <b>?</b>.</li>
        </ul>
        <div class="exemplo-box">Exemplo: O que é um Investidor Anjo?</div>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    tag = "user-msg" if m["role"] == "user" else "bolt-msg"
    st.markdown(f"<div class='{tag}'><b>{m['role'].upper()}:</b><br>{m['content']}</div>", unsafe_allow_html=True)

pergunta = st.text_input("Sua dúvida de hoje:", key="input_user")

if st.button("Consultar 💸") and pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    
    pergunta_processada = pergunta.strip() # Remove espaços extras
    
    if df_bolt is not None:
        v = vectorizer.transform([pergunta_processada])
        sim = cosine_similarity(v, tfidf_matrix)
        
        # Se a similaridade for baixa (menor que 0.3), ele admite que não sabe
        if sim.max() > 0.3:
            res = df_bolt.iloc[sim.argmax()]
            txt = f"<span class='categoria-tag'>Sua pergunta está na categoria de {res['categoria']}</span>{res['resposta']}"
        else:
            txt = "🤖 <b>Ainda não tenho esse conhecimento gravado.</b><br><br>Que tal me ensinar agora? Use o formulário 'Ensinar o Bolt' na barra lateral à esquerda!"
    else:
        txt = "⚠️ Minha base de dados está vazia ou inacessível no momento."
        
    st.session_state.messages.append({"role": "bolt", "content": txt})
    st.rerun()

st.markdown("<div class='footer'>By Miguel Generoso</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>By Miguel Generoso</div>", unsafe_allow_html=True)
