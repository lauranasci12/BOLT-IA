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
    .categoria-tag { color: #ffd700; font-weight: bold; text-transform: uppercase; display: block; margin-bottom: 10px; border-bottom: 1px solid #d4af37; width: fit-content; font-size: 0.8em; }
    
    .footer { text-align: center; color: #d4af37; font-family: 'Georgia', serif; font-style: italic; font-size: 13px; padding: 20px; opacity: 0.8; }
    
    .aviso-ia {
        background-color: rgba(212, 175, 55, 0.1);
        color: #d4af37;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        text-align: center;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .instrucoes {
        background-color: #001f4d;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #d4af37;
        margin-bottom: 25px;
    }
    .instrucoes h4 { color: #ffd700; margin-top: 0; }
    [data-testid="stSidebar"] { background-color: #000814; border-right: 2px solid #d4af37; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO DE CARREGAMENTO (O CÉREBRO) ---
@st.cache_data(show_spinner=False)
def carregar_inteligencia():
    if not os.path.exists('conhecimento.txt'):
        return None, None, None
    try:
        df = pd.read_csv('conhecimento.txt', sep=';', names=['pergunta', 'resposta', 'categoria'], encoding='utf-8')
        df = df.dropna().drop_duplicates(subset=['pergunta'], keep='last')
        if df.empty:
            return None, None, None
        
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(df['pergunta'])
        return df, vectorizer, matrix
    except:
        return None, None, None

# Carrega os dados
df_bolt, vectorizer, tfidf_matrix = carregar_inteligencia()

# --- BARRA LATERAL (ENSINAR) ---
st.sidebar.title("💎 Central de Elite")
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Ensinar o Bolt")

n_p = st.sidebar.text_input("Pergunta exata:", placeholder="Ex: O que é...")
n_r = st.sidebar.text_area("Resposta detalhada:")
n_c = st.sidebar.selectbox("Categoria:", ["Empreendedorismo", "Tecnologia", "Investimentos", "Finanças"])

if st.sidebar.button("Gravar Conhecimento 💾"):
    if n_p and n_r:
        with open('conhecimento.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{n_p.strip()};{n_r.strip()};{n_c}")
        st.cache_data.clear() # Limpa o cache para ele aprender na hora
        st.sidebar.success("Aprendido com sucesso!")
        st.rerun()
    else:
        st.sidebar.error("Preencha todos os campos!")

# --- CONTEÚDO PRINCIPAL ---
st.markdown("<h1 style='text-align: center; color: #d4af37;'>⚡ BOLT IA</h1>", unsafe_allow_html=True)

st.markdown("""
    <div class="aviso-ia">
        ⚠️ <b>Aviso:</b> O Bolt é uma Inteligência Artificial em constante aprendizado. 
        Embora o sistema se esforce para ser preciso, erros podem ocorrer. Sempre verifique informações importantes.
    </div>
    """, unsafe_allow_html=True)

# --- HISTÓRICO DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    tag = "user-msg" if m["role"] == "user" else "bolt-msg"
    st.markdown(f"<div class='{tag}'><b>{m['role'].upper()}:</b><br>{m['content']}</div>", unsafe_allow_html=True)

# --- ENTRADA DE PERGUNTA ---
pergunta_usuario = st.text_input("Sua dúvida de hoje:", key="input_user")

if st.button("Consultar 💸") and pergunta_usuario:
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    
    # Lógica de Busca (A Curiosidade/Similaridade)
    if df_bolt is not None:
        pergunta_limpa = pergunta_usuario.strip()
        vetor_pergunta = vectorizer.transform([pergunta_limpa])
        similaridade = cosine_similarity(vetor_pergunta, tfidf_matrix)
        
        # Se ele achar algo com mais de 30% de certeza
        if similaridade.max() > 0.3:
            indice_melhor_resposta = similaridade.argmax()
            resultado = df_bolt.iloc[indice_melhor_resposta]
            resposta_final = f"<span class='categoria-tag'>SUA PERGUNTA ESTÁ NA CATEGORIA DE {resultado['categoria']}</span>{resultado['resposta']}"
        else:
            resposta_final = "🤖 O sistema ainda não possui esse conhecimento gravado. Me ensine na barra lateral!"
    else:
        resposta_final = "⚠️ A base de conhecimento está vazia ou inacessível."

    # Adiciona resposta do Bolt ao histórico
    st.session_state.messages.append({"role": "bolt", "content": resposta_final})
    st.rerun()

st.markdown("<div class='footer'>By Miguel Generoso | O Bolt pode fornecer informações imprecisas.</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>By Miguel Generoso | O Bolt pode fornecer informações imprecisas.</div>", unsafe_allow_html=True)
