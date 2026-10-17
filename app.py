import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="AI Precision Medicine Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-Powered Precision Medicine Dashboard")
st.markdown("""
    Ce prototype de **médecine de précision** utilise des algorithmes de Machine Learning pour analyser
    le profil biomédical d'un patient et estimer la réponse personnalisée à un protocole de soins.
""")

# --- SIDEBAR : Paramètres du patient ---
st.sidebar.header("📋 Profil Biomédical du Patient")

def user_input_features():
    age = st.sidebar.slider("Âge", 20, 90, 50)
    bmi = st.sidebar.slider("Indice de Masse Corporelle (IMC)", 15.0, 40.0, 24.5)
    glucose = st.sidebar.slider("Taux de Glucose (mg/dL)", 70, 200, 100)
    cholesterol = st.sidebar.slider("Taux de Cholestérol (mg/dL)", 150, 300, 200)
    genetic_risk_score = st.sidebar.slider("Score de Risque Génétique (PRS)", 0.0, 1.0, 0.3)

    data = {
        'age': age,
        'bmi': bmi,
        'glucose': glucose,
        'cholesterol': cholesterol,
        'genetic_risk_score': genetic_risk_score
    }
    return pd.DataFrame(data, index=[0])

df_patient = user_input_features()

# --- CORPS DE LA PAGE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Données entrées du patient")
    st.dataframe(df_patient)

    # Simulation d'un modèle d'IA prédictif (Dans un vrai projet, on chargerait un modèle entraîné .pkl avec joblib)
    risk_score_calculated = (df_patient['age'].values[0] * 0.3 +
                             df_patient['glucose'].values[0] * 0.4 +
                             df_patient['genetic_risk_score'].values[0] * 30) / 100

    st.markdown("### 📊 Analyse Prédictive de l'IA")
    if risk_score_calculated > 0.6:
        st.error(f"⚠️ **Risque Élevé / Réponse Adaptée Restreinte** (Score estimé : {risk_score_calculated:.2f})")
        st.info("💡 **Recommandation de précision :** Opter pour une thérapie ciblée de 2nde génération et un suivi rapproché.")
    else:
        st.success(f"✅ **Risque Modéré / Favorable** (Score estimé : {risk_score_calculated:.2f})")
        st.info("💡 **Recommandation de précision :** Le protocole standard de première intention est conseillé.")

with col2:
    st.subheader("📈 Visualisation comparative des biomarqueurs")
    # Graphique comparatif par rapport à des valeurs seuils de référence
    fig, ax = plt.subplots()
    metrics = ['Glucose', 'Cholestérol']
    patient_values = [df_patient['glucose'].values[0], df_patient['cholesterol'].values[0]]
    thresholds = [125, 240] # Seuils critiques indicatifs

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, patient_values, width, label='Patient', color='#2b5c8f')
    ax.bar(x + width/2, thresholds, width, label='Seuil Critique', color='#e74c3c', alpha=0.6)

    ax.set_ylabel('Valeurs (mg/dL)')
    ax.set_title('Comparaison Patient vs Seuils Cliniques')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Projet Portfolio - Intelligence Artificielle appliquée à la Santé & Médecine de Précision</p>", unsafe_allow_html=True)