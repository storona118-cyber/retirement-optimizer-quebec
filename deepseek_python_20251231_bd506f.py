import streamlit as st
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile
import os

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Optimiseur Retraite Québec",
    layout="wide",
    page_icon="🇨🇦"
)

# =========================================================
# FISCALITÉ QC / FED - CORRIGÉ
# =========================================================
class FiscaliteQC:
    def __init__(self, inflation):
        self.inflation = inflation
        self.abattement_qc = 0.165

        # Tranches 2024 (à indexer avec inflation)
        self.tranches_fed = [
            (0, 55867, 0.15),
            (55867, 111733, 0.205),
            (111733, 173205, 0.26),
            (173205, 246752, 0.29),
            (246752, float("inf"), 0.33),
        ]

        self.tranches_qc = [
            (0, 51460, 0.14),
            (51460, 103030, 0.19),
            (103030, 123915, 0.24),
            (123915, float("inf"), 0.2575),
        ]

    def _impot_progressif(self, revenu, tranches, i):
        """Calcule l'impôt progressif avec indexation"""
        f = (1 + self.inflation) ** i
        impot = 0
        for a, b, t in tranches:
            a_adj = a * f
            b_adj = b * f if b != float("inf") else float("inf")
            if revenu > a_adj:
                montant_tranche = min(revenu, b_adj) - a_adj
                impot += montant_tranche * t
        return max(0, impot)

    def credit_age(self, revenu, i):
        """Crédit d'âge pour 65+ (indexé)"""
        f = (1 + self.inflation) ** i
        seuil = 42135 * f  # Seuil 2024
        max_credit = 8334 * 0.15  # Montant maximum 2024
        if revenu <= seuil:
            return max_credit
        reduction = (revenu - seuil) * 0.15
        return max(0, max_credit - reduction)

    def credit_pension(self, revenu_pension):
        """Crédit pour revenu de pension"""
        return min(2000, revenu_pension) * 0.15

    def recuperation_psv(self, revenu_net, psv, i):
        """Récupération du PSV (SRG)"""
        f = (1 + self.inflation) ** i
        seuil = 95864 * f  # Seuil 2024 (revenu net)
        excedent = max(0, revenu_net - seuil)
        taux_recup = 0.15  # 15% de récupération
        return min(psv, excedent * taux_recup)

    def calcul_impot(self, revenu, revenu_pension, age, i):
        """Calcule l'impôt total fédéral + provincial"""
        # Impôt brut
        fed_brut = self._impot_progressif(revenu, self.tranches_fed, i)
        qc_brut = self._impot_progressif(revenu, self.tranches_qc, i)
        
        # Crédits applicables
        credits = 0
        if age >= 65:
            credits += self.credit_age(revenu, i)
            credits += self.credit_pension(revenu_pension)
        
        # Appliquer crédits et abattement Québec
        fed_net = max(0, fed_brut - credits)
        fed_net *= (1 - self.abattement_qc)
        
        return fed_net + qc_brut

# =========================================================
# SIMULATEUR + OPTIMISEUR - AMÉLIORÉ
# =========================================================
class SimulateurRetraite:
    def __init__(self, p):
        self.p = p
        self.fisc = FiscaliteQC(p.inflation)

    def retrait_min_ferr(self, age, solde):
        """Retrait minimal FERR selon l'âge"""
        taux = {
            65: 0.04, 66: 0.041, 67: 0.042, 68: 0.043, 69: 0.044,
            70: 0.05, 71: 0.051, 72: 0.052, 73: 0.053, 74: 0.054,
            75: 0.0582, 76: 0.0593, 77: 0.0605, 78: 0.0617, 79: 0.0629,
            80: 0.0682, 81: 0.0696, 82: 0.0710, 83: 0.0724, 84: 0.0738,
            85: 0.085, 86: 0.0867, 87: 0.0884, 88: 0.0902, 89: 0.0920,
            90: 0.0938, 91: 0.0957, 92: 0.0976, 93: 0.0995, 94: 0.1015
        }
        
        # Trouver le taux le plus proche
        ages_disponibles = sorted(taux.keys())
        for a in reversed(ages_disponibles):
            if age >= a:
                return solde * taux[a]
        
        # Si moins de 65 ans (préretraite)
        return solde * 0.04 if age >= 55 else 0

    def optimiser_retraits(self, age, cible, reer, celi, pension, rrq, psv, i):
        """Optimise les retraits pour minimiser la fiscalité"""
        meilleur = None
        score_min = float("inf")
        
        # Retrait minimal FERR
        min_ferr = self.retrait_min_ferr(age, reer)
        max_ferr = min(reer * 0.12, reer)  # Maximum 12% ou solde restant
        
        # Points d'échantillonnage pour l'optimisation
        if max_ferr > min_ferr:
            points = np.linspace(min_ferr, max_ferr, 50)
        else:
            points = [min_ferr]
        
        for ret_ferr in points:
            # Calcul du revenu
            revenu = pension + rrq + psv + ret_ferr
            
            # Impôt
            impot = self.fisc.calcul_impot(
                revenu, pension + ret_ferr, age, i
            )
            
            # Récupération PSV
            recup = self.fisc.recuperation_psv(revenu - impot, psv, i)
            
            # Besoin CELI pour atteindre la cible
            revenu_apres_impot = revenu - impot - recup
            besoin_celi = max(0, cible - revenu_apres_impot)
            
            # Vérifier si CELI suffisant
            if besoin_celi > celi:
                continue
            
            # Calcul final
            net = revenu + besoin_celi - impot - recup
            
            # Score: minimise impôt + récupération + écart à la cible
            score = impot + recup + abs(net - cible) * 2
            
            if score < score_min:
                score_min = score
                meilleur = (ret_ferr, besoin_celi, net, impot, recup)
        
        return meilleur

    def simuler(self):
        """Exécute la simulation complète"""
        res = []
        reer, celi = self.p.reer, self.p.celi
        
        for i, age in enumerate(range(self.p.age_debut, self.p.age_fin + 1)):
            # Indexation avec inflation
            f = (1 + self.p.inflation) ** i
            cible = self.p.cible * f
            
            # Revenus indexés
            pension = self.p.pension * f
            rrq = self.p.rrq * f if age >= self.p.age_rrq else 0
            psv = self.p.psv * f if age >= self.p.age_psv else 0
            
            # Optimisation des retraits
            opt = self.optimiser_retraits(
                age, cible, reer, celi, pension, rrq, psv, i
            )
            
            if opt is None:
                st.warning(f"Optimisation impossible à {age} ans (CELI insuffisant)")
                break
            
            ret_ferr, ret_celi, net, impot, recup = opt
            
            # Mettre à jour les soldes
            reer = max(0, (reer - ret_ferr) * (1 + self.p.rendement))
            celi = max(0, (celi - ret_celi) * (1 + self.p.rendement))
            
            # Ajouter aux résultats
            res.append({
                "Âge": age,
                "REER/FERR": int(reer),
                "CELI": int(celi),
                "Capital Total": int(reer + celi),
                "Retrait REER": int(ret_ferr),
                "Retrait CELI": int(ret_celi),
                "RRQ": int(rrq),
                "PSV": int(psv),
                "Pension": int(pension),
                "Impôt": int(impot),
                "Récup PSV": int(recup),
                "Revenu Net": int(net),
                "Écart Cible": int(net - cible)
            })
        
        return pd.DataFrame(res)

# =========================================================
# GÉNÉRATION PDF - AMÉLIORÉE
# =========================================================
def generer_pdf(df, params):
    """Génère un rapport PDF détaillé"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    story = []
    
    # Titre
    story.append(Paragraph("RAPPORT D'OPTIMISATION DE DÉCAISSEMENT RETRAITE", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Québec - Canada", styles["Heading2"]))
    story.append(Spacer(1, 20))
    
    # Paramètres de simulation
    story.append(Paragraph("Paramètres initiaux:", styles["Heading3"]))
    params_data = [
        ["REER/FERR initial:", f"{params.reer:,.0f} $"],
        ["CELI initial:", f"{params.celi:,.0f} $"],
        ["Pension employeur:", f"{params.pension:,.0f} $"],
        ["RRQ à 65 ans:", f"{params.rrq:,.0f} $"],
        ["PSV à 65 ans:", f"{params.psv:,.0f} $"],
        ["Revenu net cible:", f"{params.cible:,.0f} $"],
        ["Période:", f"{params.age_debut} à {params.age_fin} ans"],
        ["Rendement:", f"{params.rendement*100:.1f}%"],
        ["Inflation:", f"{params.inflation*100:.1f}%"]
    ]
    params_table = Table(params_data, colWidths=[200, 100])
    params_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(params_table)
    story.append(Spacer(1, 20))
    
    # Résumé
    story.append(Paragraph("Résumé de la simulation:", styles["Heading3"]))
    resume_data = [
        ["Impôt total payé:", f"{df['Impôt'].sum():,.0f} $"],
        ["PSV récupéré total:", f"{df['Récup PSV'].sum():,.0f} $"],
        ["Capital final à {params.age_fin} ans:", f"{df['Capital Total'].iloc[-1]:,.0f} $"],
        ["Revenu net moyen:", f"{df['Revenu Net'].mean():,.0f} $"],
        ["Revenu net médian:", f"{df['Revenu Net'].median():,.0f} $"],
        ["Âge épuisement CELI:", f"{df[df['CELI'] <= 0]['Âge'].iloc[0] if (df['CELI'] <= 0).any() else 'Non épuisé'}"],
    ]
    resume_table = Table(resume_data, colWidths=[200, 100])
    resume_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(resume_table)
    story.append(Spacer(1, 20))
    
    # Tableau détaillé (premières lignes)
    story.append(Paragraph("Projection annuelle (extrait):", styles["Heading3"]))
    
    # Préparer les données pour le tableau
    df_display = df.head(10).copy()
    table_data = [list(df_display.columns)]
    for _, row in df_display.iterrows():
        table_data.append([
            str(int(row["Âge"])),
            f"{row['REER/FERR']:,.0f}",
            f"{row['CELI']:,.0f}",
            f"{row['Capital Total']:,.0f}",
            f"{row['Retrait REER']:,.0f}",
            f"{row['Retrait CELI']:,.0f}",
            f"{row['Revenu Net']:,.0f}",
            f"{row['Impôt']:,.0f}",
        ])
    
    detailed_table = Table(table_data, repeatRows=1)
    detailed_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(detailed_table)
    
    # Notes
    story.append(Spacer(1, 20))
    notes = """
    <b>Notes:</b><br/>
    • Les montants sont en dollars courants (indexés à l'inflation)<br/>
    • Les taux d'impôt sont ceux de 2024 indexés à l'inflation<br/>
    • Les retraits REER/FERR suivent les minimums selon l'âge<br/>
    • Optimisation pour minimiser l'impôt et la récupération PSV<br/>
    • Simulation à des fins éducatives seulement
    """
    story.append(Paragraph(notes, styles["Normal"]))
    
    # Générer le PDF
    doc.build(story)
    return tmp.name

# =========================================================
# INTERFACE STREAMLIT - AMÉLIORÉE
# =========================================================
st.title("🇨🇦 Optimiseur de Décaissement Retraite Québec")
st.markdown("""
Ce simulateur optimise vos retraits de REER/FERR et CELI pour minimiser l'impôt et la récupération du PSV 
tout en maintenant votre revenu net cible.
""")

# Initialisation des paramètres
class Parametres:
    pass

p = Parametres()

# Sidebar pour les paramètres
with st.sidebar:
    st.header("💰 Paramètres financiers")
    
    col1, col2 = st.columns(2)
    with col1:
        p.reer = st.number_input(
            "REER/FERR ($)",
            min_value=0,
            max_value=5000000,
            value=1250000,
            step=10000,
            help="Solde total de vos régimes enregistrés"
        )
    with col2:
        p.celi = st.number_input(
            "CELI ($)",
            min_value=0,
            max_value=1000000,
            value=135000,
            step=5000,
            help="Solde total de votre CELI"
        )
    
    st.divider()
    st.header("📅 Revenus de retraite")
    
    p.pension = st.number_input(
        "Pension employeur ($/an)",
        min_value=0,
        max_value=200000,
        value=23000,
        step=1000
    )
    
    col1, col2 = st.columns(2)
    with col1:
        p.rrq = st.number_input(
            "RRQ à 65 ans ($)",
            min_value=0,
            max_value=35000,
            value=16336,
            step=100,
            help="Montant annuel maximal en 2024: 16,375$"
        )
    with col2:
        p.psv = st.number_input(
            "PSV à 65 ans ($)",
            min_value=0,
            max_value=20000,
            value=8732,
            step=100,
            help="Montant maximal en 2024: 10,843$"
        )
    
    st.divider()
    st.header("🎯 Objectifs")
    
    p.cible = st.slider(
        "Revenu net cible ($/an)",
        min_value=30000,
        max_value=200000,
        value=85000,
        step=5000
    )
    
    st.divider()
    st.header("📈 Hypothèses")
    
    col1, col2 = st.columns(2)
    with col1:
        p.age_rrq = st.slider("Âge début RRQ", 60, 70, 70)
        p.age_psv = st.slider("Âge début PSV", 65, 70, 70)
    with col2:
        p.rendement = st.slider("Rendement net (%)", -2.0, 8.0, 4.5) / 100
        p.inflation = st.slider("Inflation (%)", 0.0, 5.0, 2.0) / 100
    
    st.divider()
    st.header("⏱️ Période")
    
    col1, col2 = st.columns(2)
    with col1:
        p.age_debut = st.slider("Âge début", 55, 75, 65)
    with col2:
        p.age_fin = st.slider("Âge fin", 75, 105, 95)

# Bouton de simulation
if st.button("🚀 Lancer la simulation", type="primary", use_container_width=True):
    with st.spinner("Optimisation en cours..."):
        # Création du simulateur
        sim = SimulateurRetraite(p)
        
        # Exécution de la simulation
        df = sim.simuler()
        
        if df.empty:
            st.error("Impossible de générer une simulation avec les paramètres actuels.")
        else:
            # Affichage des résultats
            st.success(f"Simulation terminée! Période: {p.age_debut} à {p.age_fin} ans")
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Capital final",
                    f"{df['Capital Total'].iloc[-1]:,.0f} $",
                    f"{((df['Capital Total'].iloc[-1]/(p.reer+p.celi))-1)*100:.1f}%"
                )
            with col2:
                st.metric(
                    "Impôt total payé",
                    f"{df['Impôt'].sum():,.0f} $",
                    delta_color="inverse"
                )
            with col3:
                st.metric(
                    "Revenu net moyen",
                    f"{df['Revenu Net'].mean():,.0f} $",
                    f"{df['Revenu Net'].std():,.0f} $"
                )
            with col4:
                dernier_celi = df['CELI'].iloc[-1]
                st.metric(
                    "CELI final",
                    f"{dernier_celi:,.0f} $",
                    "Épuisé" if dernier_celi <= 0 else "Restant"
                )
            
            # Graphiques
            tab1, tab2, tab3 = st.tabs(["📊 Données détaillées", "📈 Évolution du capital", "💰 Revenus et impôts"])
            
            with tab1:
                # Formater l'affichage
                display_df = df.copy()
                display_df = display_df.set_index("Âge")
                display_df = display_df.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Options d'export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger CSV",
                    csv,
                    "simulation_retraite.csv",
                    "text/csv"
                )
            
            with tab2:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Âge'], y=df['REER/FERR'],
                    name='REER/FERR', line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Âge'], y=df['CELI'],
                    name='CELI', line=dict(color='green')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Âge'], y=df['Capital Total'],
                    name='Capital Total', line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title="Évolution du capital",
                    xaxis_title="Âge",
                    yaxis_title="Montant ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=df['Âge'], y=df['Revenu Net'],
                    name='Revenu Net', marker_color='green'
                ))
                fig2.add_trace(go.Bar(
                    x=df['Âge'], y=df['Impôt'],
                    name='Impôt', marker_color='red'
                ))
                fig2.add_trace(go.Scatter(
                    x=df['Âge'], y=[p.cible * (1 + p.inflation)**i for i in range(len(df))],
                    name='Cible indexée', line=dict(color='orange', dash='dash')
                ))
                
                fig2.update_layout(
                    title="Revenus et impôts annuels",
                    xaxis_title="Âge",
                    yaxis_title="Montant ($)",
                    barmode='group'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Génération du PDF
            with st.spinner("Génération du rapport PDF..."):
                try:
                    pdf_path = generer_pdf(df, p)
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📄 Télécharger rapport PDF complet",
                            f,
                            "rapport_optimisation_retraite.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                    
                    # Nettoyage
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération du PDF: {str(e)}")

# Section d'aide
with st.expander("ℹ️ Comment utiliser cet outil"):
    st.markdown("""
    ### Guide d'utilisation
    
    1. **Paramètres financiers**: Entrez vos soldes actuels de REER/FERR et CELI
    2. **Revenus de retraite**: Indiquez vos revenus garantis (pension, RRQ, PSV)
    3. **Objectifs**: Définissez votre revenu net annuel souhaité
    4. **Hypothèses**: Ajustez les hypothèses de rendement et d'inflation
    5. **Période**: Choisissez l'âge de début et de fin de simulation
    
    ### Points clés
    
    - **REER/FERR**: Retraits optimisés pour minimiser l'impôt
    - **CELI**: Utilisé pour combler l'écart avec la cible de revenu
    - **RRQ**: Peut être pris entre 60 et 70 ans (réduction/augmentation)
    - **PSV**: Récupération à partir de 95,364$ de revenu net (2024)
    - **Impôt**: Calculé selon les tables 2024 indexées à l'inflation
    
    ### Limitations
    
    - Simulation à des fins éducatives seulement
    - Ne remplace pas un conseiller financier
    - Hypothèses fiscales basées sur les règles 2024
    - Ne tient pas compte de tous les crédits d'impôt
    """)

# Pied de page
st.divider()
st.caption("""
*Cet outil est fourni à titre informatif seulement. Les résultats sont basés sur des hypothèses et 
ne constituent pas des conseils financiers. Consultez un conseiller financier qualifié pour votre planification personnelle.*
""")