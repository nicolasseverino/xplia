"""
Dashboard de conformité XPLIA (Streamlit)
=========================================

Visualisation interactive de l’audit trail RGPD, du log AI Act, génération de rapports et export multi-formats.
"""

import streamlit as st
import xplia
import pandas as pd

st.set_page_config(page_title="XPLIA Compliance Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🛡️ Dashboard de conformité XPLIA")

st.sidebar.header("Filtres et actions")
user_id = st.sidebar.text_input("Filtrer par user_id")
format_rapport = st.sidebar.selectbox("Format du rapport", ["pdf", "html", "markdown", "json"])

# Chargement des logs
with st.spinner("Chargement de l’audit trail RGPD et du log AI Act..."):
    audit_trail = xplia.export_audit_trail()
    decision_log = xplia.export_decision_log()

if user_id:
    audit_trail = [entry for entry in audit_trail if entry.get("user_id") == user_id]
    decision_log = [entry for entry in decision_log if entry.get("user_id") == user_id]

st.subheader("Journal RGPD (audit trail)")
if audit_trail:
    st.dataframe(pd.DataFrame(audit_trail))
else:
    st.info("Aucune demande d’explication enregistrée.")

st.subheader("Log AI Act (décisions)")
if decision_log:
    st.dataframe(pd.DataFrame(decision_log))
else:
    st.info("Aucune décision enregistrée.")

st.sidebar.markdown("---")
if st.sidebar.button("Générer rapport de conformité"):
    with st.spinner("Génération du rapport..."):
        if format_rapport == "pdf":
            path = "rapport_conformite.pdf"
            xplia.generate_report(format="pdf", output_path=path)
            with open(path, "rb") as f:
                st.sidebar.download_button("Télécharger PDF", f, file_name=path)
        elif format_rapport == "html":
            html = xplia.generate_report(format="html", output_path="rapport_conformite.html")
            st.sidebar.download_button("Télécharger HTML", html, file_name="rapport_conformite.html")
        elif format_rapport == "markdown":
            md = xplia.generate_report(format="markdown", output_path="rapport_conformite.md")
            st.sidebar.download_button("Télécharger Markdown", md, file_name="rapport_conformite.md")
        elif format_rapport == "json":
            js = xplia.generate_report(format="json", output_path="rapport_conformite.json")
            st.sidebar.download_button("Télécharger JSON", js, file_name="rapport_conformite.json")
        st.sidebar.success("Rapport généré avec succès !")

st.sidebar.markdown("---")
st.sidebar.info("XPLIA 2025 — Conformité RGPD | AI Act | Extensions sectorielles à venir.")
