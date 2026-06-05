import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURAÇÃO DE TELA E DEPENDÊNCIAS
# ==========================================
st.set_page_config(
    page_title="WaterDash | Inteligência Preditiva",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa o estado da previsão no session_state para não perder o clique
if "predicao_realizada" not in st.session_state:
    st.session_state.predicao_realizada = False

# Injeção de Ícones Profissionais e CSS Premium (UI/UX Custom)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
    }
    
    .norma-ref-container {
        background: rgba(30, 41, 59, 0.7);
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #06b6d4;
        margin-top: 14px;
        margin-bottom: 2px;
    }
    .norma-ref-text {
        color: #38bdf8 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .glossario-sidebar-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .glossario-sidebar-title {
        color: #f8fafc !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        margin: 0 0 4px 0 !important;
    }
    .glossario-sidebar-desc {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }
    
    .slider-title-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 4px;
    }
    .slider-title-text {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        font-weight: 500;
    }
    
    /* CARDS DE MÉTRICAS OPERACIONAIS */
    .metric-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.03);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .metric-title {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    
    /* BADGES DE STATUS */
    .badge-success {
        background-color: #f0fdf4;
        color: #16a34a;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #bbf7d0;
        display: inline-block;
    }
    .badge-alert {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #fecaca;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINIÇÃO OPERACIONAL E REFERÊNCIAS BRASIL
# ==========================================
# Fontes das Imagens mantidas 100% ONLINE com ícones de alta definição técnica
PARAM_INFO = {
    "ph": {
        "nome": "Potencial Hidrogeniônico", "sigla": "pH", "unidade": "", 
        "ref_fixa": "Recomendado: 6,00 a 9,50", "limite_vmp": 9.5,
        "img": "https://cdn-icons-png.flaticon.com/512/3655/3655580.png", # Escala de pH/Béquer
        "descricao": "Mede o índice de acidez ou alcalinidade da água para evitar a corrosão de tubulações ou rejeição ao consumo."
    },
    "Hardness": {
        "nome": "Dureza Total", "sigla": "Dureza", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 300,00 mg/L", "limite_vmp": 300.0,
        "img": "https://cdn-icons-png.flaticon.com/512/2942/2942911.png", # Estrutura Mineral/Cálcio
        "descricao": "Indica a concentração de sais de cálcio e magnésio, cujo excesso pode incrustar tubulações e reduzir a eficiência de sabões."
    },
    "Solids": {
        "nome": "Sólidos Totais Dissolvidos", "sigla": "STD", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 1000,00 mg/L", "limite_vmp": 1000.0,
        "img": "https://cdn-icons-png.flaticon.com/512/8643/8643834.png", # Filtro de partículas dissolvidas
        "descricao": "Representa o total de minerais, sais e metais dissolvidos na água, afetando diretamente o sabor e a salubridade."
    },
    "Chloramines": {
        "nome": "Cloraminas", "sigla": "Cloraminas", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 4,00 mg/L", "limite_vmp": 4.0,
        "img": "https://cdn-icons-png.flaticon.com/512/2800/2800114.png", # Molécula purificadora de Cloro
        "descricao": "Derivados de cloro utilizados para desinfetar a água, garantindo a eliminação de patógenos ao longo da rede de distribuição."
    },
    "Sulfate": {
        "nome": "Sulfato", "sigla": "Sulfato", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 250,00 mg/L", "limite_vmp": 250.0,
        "img": "https://cdn-icons-png.flaticon.com/512/4115/4115456.png", # Íon/Composto químico
        "descricao": "Sais naturais dissolvidos que, em concentrações muito elevadas, podem causar efeitos laxativos e gosto amargo."
    },
    "Conductivity": {
        "nome": "Condutividade Elétrica", "sigla": "Condutividade", "unidade": " μS/cm", 
        "ref_fixa": "Ref. Operacional: < 400,00 μS/cm", "limite_vmp": 400.0,
        "img": "https://cdn-icons-png.flaticon.com/512/11502/11502542.png", # Sonda com sinal elétrico
        "descricao": "Mede a capacidade da água de conduzir corrente elétrica, sinalizando de forma indireta a quantidade de íons dissolvidos."
    },
    "Organic_carbon": {
        "nome": "Carbono Orgânico Total", "sigla": "COT", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 3,00 mg/L", "limite_vmp": 3.0,
        "img": "https://cdn-icons-png.flaticon.com/512/1546/1546200.png", # Molécula orgânica de Carbono
        "descricao": "Quantifica a presença de matéria orgânica na água, servindo como indicador básico de contaminação biológica."
    },
    "Trihalomethanes": {
        "nome": "Trihalometanos Totais", "sigla": "THM", "unidade": " mg/L", 
        "ref_fixa": "Máximo Permitido: 0,10 mg/L", "limite_vmp": 0.1,
        "img": "https://cdn-icons-png.flaticon.com/512/9746/9746197.png", # Vidraria analítica de precisão
        "descricao": "Subprodutos químicos formados pela reação do cloro com a matéria orgânica, monitorados por seu potencial carcinogênico."
    },
    "Turbidity": {
        "nome": "Turbidez", "sigla": "Turbidez", "unidade": " uT", 
        "ref_fixa": "Limite de Saída ETA: 5,00 uT", "limite_vmp": 5.0,
        "img": "https://cdn-icons-png.flaticon.com/512/3127/3127357.png", # Fluido disperso / Tubo óptico
        "descricao": "Mede o grau de atenuação da luz devido a partículas sólidas suspensas, indicando o aspecto visual de clareza da água."
    }
}

# ==========================================
# 3. ENGINE DE DADOS
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Potabilidade_da_agua.csv")
    df = df.fillna(df.mean())
    for col in df.columns:
        if col.lower() != "potability":
            max_value = df[col].max()
            if max_value > 100000:
                if col.lower() == "ph": df[col] = (df[col] / max_value) * 14
                elif col in ["Hardness", "Sulfate"]: df[col] = (df[col] / max_value) * 350
                elif col == "Solids": df[col] = (df[col] / max_value) * 1200
                elif col in ["Chloramines", "Organic_carbon"]: df[col] = (df[col] / max_value) * 5
                elif col == "Conductivity": df[col] = (df[col] / max_value) * 500
                elif col == "Trihalomethanes": df[col] = (df[col] / max_value) * 0.15
                elif col == "Turbidity": df[col] = (df[col] / max_value) * 6
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except: return None, None

df_clean = load_data()
cols_to_drop = [c for c in df_clean.columns if c.lower() == "potability"]
X = df_clean.drop(columns=cols_to_drop, errors='ignore')
model, scaler = load_model()

# ==========================================
# 4. SIDEBAR - CONTROLES
# ==========================================
st.sidebar.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 20px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3105/3105803.png' style='width: 42px; height: 42px;'>
        <h2 style='color: #ffffff; font-size: 1.3rem; margin: 0; font-weight:700;'>WaterPredictor</h2>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("Referências baseadas na Portaria GM/MS nº 888.")
st.sidebar.markdown("<hr style='border-color: #1e293b; margin: 10px 0;'>", unsafe_allow_html=True)

input_data = {}

# Callback para re-executar dinamicamente quando mexer nos sliders
def reset_click():
    st.session_state.predicao_realizada = True

# pH - Componente isolado com legenda intacta
ph_csv_col = next((c for c in df_clean.columns if c.lower() == "ph"), None)
ph_mean = float(df_clean[ph_csv_col].mean()) if ph_csv_col else 7.0
ph_min = float(df_clean[ph_csv_col].min()) if ph_csv_col else 0.0
ph_max = float(df_clean[ph_csv_col].max()) if ph_csv_col else 14.0

st.sidebar.markdown(f'<div class="norma-ref-container"><p class="norma-ref-text">🛡️ {PARAM_INFO["ph"]["ref_fixa"]}</p></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="slider-title-container"><span class="slider-title-text"><b>pH</b> <small style="color:#94a3b8;">(Potencial Hidrogeniônico)</small></span></div>', unsafe_allow_html=True)
input_data["ph"] = st.sidebar.slider(label="InputSliderDefinitivo_ph", min_value=ph_min, max_value=ph_max, value=ph_mean, format="%.4f", label_visibility="collapsed", key="id_real_slider_ph", on_change=reset_click)
st.sidebar.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# Loop dos demais parâmetros na Sidebar
for col, info in PARAM_INFO.items():
    if col == "ph": continue
    col_alvo = next((c for c in df_clean.columns if c.lower() == col.lower()), None)
    if col_alvo is not None:
        mean_val = float(df_clean[col_alvo].mean())
        min_val = float(df_clean[col_alvo].min())
        max_val = float(df_clean[col_alvo].max())
        
        if info["ref_fixa"] != "":
            st.sidebar.markdown(f'<div class="norma-ref-container"><p class="norma-ref-text">🛡️ {info["ref_fixa"]}</p></div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        st.sidebar.markdown(f'<div class="slider-title-container"><span class="slider-title-text"><b>{info["sigla"]}</b> <small style="color:#94a3b8;">({info["nome"]})</small> <small style="color:#38bdf8;">{info["unidade"]}</small></span></div>', unsafe_allow_html=True)
        input_data[col] = st.sidebar.slider(label=f"InputSliderDefinitivo_{col}", min_value=min_val, max_value=max_val, value=mean_val, format="%.4f", label_visibility="collapsed", key=f"id_real_slider_{col}", on_change=reset_click)
        st.sidebar.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)

# Botão atualiza o session state de forma estrita
if st.sidebar.button("🔬 Rodar Análise Preditiva (IA)", type="primary", use_container_width=True):
    st.session_state.predicao_realizada = True

# Dicionário de Variáveis
st.sidebar.markdown("<br><h4 style='color: #ffffff; font-size: 1rem; font-weight:700; margin-bottom:12px;'>📖 Dicionário de Variáveis</h4>", unsafe_allow_html=True)
for col, info in PARAM_INFO.items():
    st.sidebar.markdown(f"""
        <div class="glossario-sidebar-box">
            <p class="glossario-sidebar-title">🔹 {info['sigla']} ({info['nome']})</p>
            <p class="glossario-sidebar-desc">{info['descricao']}</p>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. PAINEL PRINCIPAL (DASHBOARD)
# ==========================================
st.markdown("<h1 style='color: #0f172a; font-size: 2.1rem; font-weight: 700; margin-bottom: 4px;'>💧 Monitoramento Preditivo sobre a Potabilidade da Água</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.05rem; margin-bottom: 24px;'>Painel de monitoramento em tempo real integrado com os padrões de potabilidade do Ministério da Saúde.</p>", unsafe_allow_html=True)

st.markdown("<h3 style='color: #0f172a; font-weight: 700; margin-bottom: 16px;'>📋 Visão Global dos Parâmetros Operacionais</h3>", unsafe_allow_html=True)

# --- MATRIZ GLOBAL DE METRIC CARDS (3x3) ---
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2, row3_col3 = st.columns(3)

# --- LINHA 1 ---
with row1_col1:
    v_ph = input_data.get('ph', 7.0)
    is_ok = 6.0 <= v_ph <= 9.5
    badge = '<span class="badge-success">✓ Conforme</span>' if is_ok else '<span class="badge-alert">⚠️ Fora da Faixa (6.0-9.5)</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['ph']['nome']} ({PARAM_INFO['ph']['sigla']})</span><img src="{PARAM_INFO['ph']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_ph:.2f}</div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row1_col2:
    v_hard = input_data.get('Hardness', 150.0)
    is_ok = v_hard <= PARAM_INFO['Hardness']['limite_vmp']
    badge = '<span class="badge-success">✓ Conforme</span>' if is_ok else '<span class="badge-alert">⚠️ Acima do VMP</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Hardness']['nome']}</span><img src="{PARAM_INFO['Hardness']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_hard:.2f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row1_col3:
    v_solid = input_data.get('Solids', 500.0)
    is_ok = v_solid <= PARAM_INFO['Solids']['limite_vmp']
    badge = '<span class="badge-success">✓ Conforme</span>' if is_ok else '<span class="badge-alert">⚠️ Acima do VMP</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Solids']['nome']} ({PARAM_INFO['Solids']['sigla']})</span><img src="{PARAM_INFO['Solids']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_solid:.2f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

# --- LINHA 2 ---
with row2_col1:
    v_chlor = input_data.get('Chloramines', 2.0)
    is_ok = v_chlor <= PARAM_INFO['Chloramines']['limite_vmp']
    badge = '<span class="badge-success">✓ Conforme</span>' if is_ok else '<span class="badge-alert">⚠️ Acima do VMP</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Chloramines']['nome']}</span><img src="{PARAM_INFO['Chloramines']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_chlor:.2f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row2_col2:
    v_sulf = input_data.get('Sulfate', 100.0)
    is_ok = v_sulf <= PARAM_INFO['Sulfate']['limite_vmp']
    badge = '<span class="badge-success">✓ Conforme</span>' if is_ok else '<span class="badge-alert">⚠️ Acima do VMP</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Sulfate']['nome']}</span><img src="{PARAM_INFO['Sulfate']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_sulf:.2f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row2_col3:
    v_cond = input_data.get('Conductivity', 250.0)
    is_ok = v_cond <= PARAM_INFO['Conductivity']['limite_vmp']
    badge = '<span class="badge-success">✓ Estável</span>' if is_ok else '<span class="badge-alert">⚠️ Elevada</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Conductivity']['nome']}</span><img src="{PARAM_INFO['Conductivity']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_cond:.2f}<span style="font-size:0.9rem; color:#64748b;"> μS/cm</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

# --- LINHA 3 ---
with row3_col1:
    v_org = input_data.get('Organic_carbon', 1.5)
    is_ok = v_org <= PARAM_INFO['Organic_carbon']['limite_vmp']
    badge = '<span class="badge-success">✓ Seguro</span>' if is_ok else '<span class="badge-alert">⚠️ Risco Biológico</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Organic_carbon']['nome']} ({PARAM_INFO['Organic_carbon']['sigla']})</span><img src="{PARAM_INFO['Organic_carbon']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_org:.2f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row3_col2:
    v_thm = input_data.get('Trihalomethanes', 0.04)
    is_ok = v_thm <= PARAM_INFO['Trihalomethanes']['limite_vmp']
    badge = '<span class="badge-success">✓ Seguro</span>' if is_ok else '<span class="badge-alert">⚠️ Risco Químico</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Trihalomethanes']['nome']} ({PARAM_INFO['Trihalomethanes']['sigla']})</span><img src="{PARAM_INFO['Trihalomethanes']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_thm:.4f}<span style="font-size:0.9rem; color:#64748b;"> mg/L</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)

with row3_col3:
    v_turb = input_data.get('Turbidity', 2.0)
    is_ok = v_turb <= PARAM_INFO['Turbidity']['limite_vmp']
    badge = '<span class="badge-success">✓ Cristalina</span>' if is_ok else '<span class="badge-alert">⚠️ Turbidez Elevada</span>'
    st.markdown(f"""<div class="metric-card"><div class="metric-header"><span class="metric-title">{PARAM_INFO['Turbidity']['nome']}</span><img src="{PARAM_INFO['Turbidity']['img']}" style="width:38px; height:38px;"></div><div class="metric-value">{v_turb:.2f}<span style="font-size:0.9rem; color:#64748b;"> uT</span></div><div>{badge}</div></div>""", unsafe_allow_html=True)


# --- GRÁFICO INTERATIVO COMPLEMENTAR ---
st.markdown("<br><h3 style='color: #0f172a; font-weight: 700; margin-bottom: 12px;'>📊 Análise Comparativa de Conformidade Legal</h3>", unsafe_allow_html=True)

def plot_premium_bars(input_dict, param_info):
    nomes_exibicao = [f"{info['sigla']} ({info['unidade'].strip()})" if info['unidade'] else info['sigla'] for info in param_info.values()]
    valores_reais = [input_dict.get(col, 0) for col in param_info.keys()]
    limites = [info['limite_vmp'] for info in param_info.values()]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=nomes_exibicao, x=valores_reais, name='Amostra Condomínio', orientation='h', marker=dict(color='#06b6d4', line=dict(color='#0891b2', width=1)), text=[f"{v:.3f}" for v in valores_reais], textposition='auto'))
    fig.add_trace(go.Bar(y=nomes_exibicao, x=limites, name='Teto Legal (VMP - Portaria 888)', orientation='h', marker=dict(color='rgba(244, 63, 94, 0.25)', line=dict(color='#f43f5e', width=1.5)), text=[f"{l:.2f}" for l in limites], textposition='outside'))
    
    fig.update_layout(barmode='group', height=520, margin=dict(l=20, r=40, t=10, b=10), plot_bgcolor='rgba(255, 255, 255, 0.4)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1), xaxis=dict(tickformat=".2f", gridcolor='#e2e8f0', zeroline=False), yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#0f172a', family='Plus Jakarta Sans')))
    return fig

st.plotly_chart(plot_premium_bars(input_data, PARAM_INFO), use_container_width=True)

# Alertas Dinâmicos de Violação
violacoes = [PARAM_INFO[col]['nome'] for col, val in input_data.items() if val > PARAM_INFO[col]['limite_vmp'] or (col == 'ph' and not (6.0 <= val <= 9.5))]
if violacoes:
    st.markdown(f"""<div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 16px; border-radius: 8px; margin-top: 15px;"><p style="color: #991b1b; margin: 0; font-weight: 600; font-size: 0.95rem;">🚨 VMP ULTRAPASSADO: Os seguintes parâmetros encontram-se fora dos padrões legais: <span style="font-weight: 700; text-decoration: underline;">{', '.join(violacoes)}</span>.</p></div>""", unsafe_allow_html=True)

# --- MOTOR ANALÍTICO PREDITIVE (IA) ---
st.markdown("<br><hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #0f172a; font-weight: 700; margin-bottom: 4px;'>🤖 Inteligência Artificial Preditiva</h3>", unsafe_allow_html=True)

if st.session_state.predicao_realizada:
    if model is not None and scaler is not None:
        input_df = pd.DataFrame([input_data])[X.columns]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if pred[0] == 1:
                st.markdown('<div style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); padding: 24px; border-radius: 16px; color: white; box-shadow: 0 4px 15px rgba(34,197,94,0.3);"><h4 style="margin:0; font-weight:700; text-align:center;">LAUDO: ÁGUA POTÁVEL</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 24px; border-radius: 16px; color: white; box-shadow: 0 4px 15px rgba(239,68,68,0.3);"><h4 style="margin:0; font-weight:700; text-align:center;">LAUDO: ÁGUA REJEITADA</h4></div>', unsafe_allow_html=True)
        with res_col2:
            st.info("A inteligência analítica opera de forma integrada calculando correlações cruzadas não-lineares.")
    else:
        st.warning("⚠️ Arquivos do modelo de IA não encontrados.")
else:
    st.markdown('<div style="background-color: #ffffff; border: 1px dashed #cbd5e1; padding: 16px; border-radius: 8px; text-align: center;"><p style="color: #64748b; margin: 0;">💡 Clique em <b>Rodar Análise Preditiva</b> para ativar a inteligência artificial.</p></div>', unsafe_allow_html=True)
