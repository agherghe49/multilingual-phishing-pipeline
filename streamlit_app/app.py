"""
Streamlit dashboard — Dizertație: Generarea și Detecția Atacurilor Phishing via E-mail

Rulare:
    pip install streamlit plotly
    streamlit run streamlit_app/app.py
"""

import json
import csv
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(
    page_title="Phishing AI Dashboard",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_scaling_csv():
    p = OUTPUTS / "scaling_laws" / "scaling_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data
def load_dataset_stats():
    path = OUTPUTS / "dataset.jsonl"
    if not path.exists():
        return {}
    from collections import Counter
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {
        "total":    len(data),
        "phishing": sum(1 for d in data if d["label"] == 1),
        "ham":      sum(1 for d in data if d["label"] == 0),
        "by_locale": dict(Counter(d.get("locale","?") for d in data)),
        "by_label_locale": {
            loc: {
                "phishing": sum(1 for d in data if d["label"]==1 and d.get("locale")==loc),
                "ham":      sum(1 for d in data if d["label"]==0 and d.get("locale")==loc),
            }
            for loc in ["ro-RO","en-US","de-DE","fr-FR","it-IT"]
        },
        "sources": dict(Counter(d.get("source","?") for d in data)),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎣 Phishing AI")
st.sidebar.markdown("**Generarea și Detecția Atacurilor Phishing via E-mail**")
st.sidebar.markdown("*Dizertație — Universitatea Politehnica București*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigare", [
    "📊 Dataset",
    "🔬 Scaling Laws",
    "⚙️ Pipeline Generare",
    "🤖 Antrenare GRPO",
    "⚔️ Evaluare Adversarială",
    "🌍 Analiză per Limbă",
    "🌐 Transfer Cross-Locale",
    "🔍 Explainability",
    "📝 Analiză Lingvistică",
])

st.sidebar.markdown("---")
st.sidebar.caption("RTX 4090 · Qwen2.5-7B · XLM-RoBERTa")

LOCALES = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]
COLORS  = {"ro-RO": "#1976D2", "en-US": "#E53935", "de-DE": "#388E3C",
           "fr-FR": "#F57C00", "it-IT": "#7B1FA2"}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dataset
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dataset":
    st.title("📊 Dataset Phishing Multilingual")
    st.markdown("Dataset sintetic de emailuri phishing și legitime (ham) în 5 limbi.")

    stats = load_dataset_stats()
    if not stats:
        st.warning("dataset.jsonl nu a fost găsit.")
        st.stop()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total emailuri", f"{stats['total']:,}")
    c2.metric("Phishing", f"{stats['phishing']:,}", f"{stats['phishing']/stats['total']*100:.1f}%")
    c3.metric("Ham legitim", f"{stats['ham']:,}", f"{stats['ham']/stats['total']*100:.1f}%")
    c4.metric("Limbi", "5")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuție per limbă")
        rows = []
        for loc in LOCALES:
            d = stats["by_label_locale"].get(loc, {})
            rows.append({"Limbă": loc, "Phishing": d.get("phishing",0), "Ham": d.get("ham",0)})
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(x=df["Limbă"], y=df["Phishing"], name="Phishing", marker_color="#E53935")
        fig.add_bar(x=df["Limbă"], y=df["Ham"],      name="Ham legitim", marker_color="#1976D2")
        fig.update_layout(barmode="group", height=350,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuție Phishing vs. Ham")
        fig = go.Figure(go.Pie(
            labels=["Phishing", "Ham legitim"],
            values=[stats["phishing"], stats["ham"]],
            hole=0.4,
            marker_colors=["#E53935", "#1976D2"],
        ))
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Surse de date")
    src_df = pd.DataFrame([
        {"Sursă": k.replace("_"," ").title(), "N emailuri": v}
        for k, v in stats["sources"].items()
    ]).sort_values("N emailuri", ascending=False)
    st.dataframe(src_df, use_container_width=True, hide_index=True)

    st.subheader("Tabel per limbă")
    tbl = pd.DataFrame([
        {"Limbă": loc,
         "Phishing": stats["by_label_locale"].get(loc, {}).get("phishing", 0),
         "Ham": stats["by_label_locale"].get(loc, {}).get("ham", 0),
         "Total": stats["by_locale"].get(loc, 0)}
        for loc in LOCALES
    ])
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Scaling Laws
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Scaling Laws":
    st.title("🔬 Scaling Laws — Detecție Phishing")
    st.markdown("Performanța modelelor de detecție în funcție de dimensiunea setului de antrenare.")

    df = load_scaling_csv()
    if df.empty:
        st.warning("scaling_results.csv nu a fost găsit.")
        st.stop()

    models = sorted(df["model"].unique())
    sel_models = st.multiselect("Modele", models, default=models)
    metric = st.selectbox("Metrică", ["f1", "fnr", "auc_roc", "precision", "recall"],
                          format_func=lambda x: {"f1":"F1-phishing","fnr":"FNR","auc_roc":"AUC-ROC",
                                                 "precision":"Precision","recall":"Recall"}[x])

    df_f = df[df["model"].isin(sel_models)]

    pal = px.colors.qualitative.Bold
    color_map = {m: pal[i % len(pal)] for i, m in enumerate(models)}

    fig = go.Figure()
    for model in sel_models:
        sub = df_f[df_f["model"] == model].sort_values("n")
        fig.add_scatter(x=sub["n"], y=sub[metric], mode="lines+markers",
                        name=model, line=dict(color=color_map[model], width=2),
                        marker=dict(size=8))

    fig.update_layout(
        xaxis_title="Exemple de antrenare (N)",
        yaxis_title=metric.upper(),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=20),
        xaxis=dict(type="log"),
    )
    if metric == "fnr":
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="FNR target 5%")
    if metric == "f1":
        fig.add_hline(y=0.99, line_dash="dash", line_color="green", annotation_text="F1=0.99")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabel complet")
    show_cols = ["n", "model", "f1", "fnr", "auc_roc", "precision", "recall"]
    st.dataframe(df_f[show_cols].sort_values(["model","n"]).round(4),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Pipeline Generare
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Pipeline Generare":
    st.title("⚙️ Pipeline de Generare — Comparație Etape")
    st.markdown("Contribuția fiecărei componente (RAG, Self-Correction, GRPO) la calitatea finală.")

    data = load_json(OUTPUTS / "pipeline_comparison" / "pipeline_results.json")
    if not data:
        st.warning("pipeline_results.json nu a fost găsit.")
        st.stop()

    results = data["results"]
    stages  = list(results.keys())

    metrics = ["reward", "quality", "diversity", "format"]
    labels  = {"reward":"Reward total", "quality":"Calitate",
                "diversity":"Diversitate", "format":"Format"}
    colors_m = {"reward":"#1f77b4","quality":"#ff7f0e","diversity":"#2ca02c","format":"#d62728"}

    col1, col2 = st.columns(2)
    for i, metric in enumerate(metrics):
        ax_col = col1 if i % 2 == 0 else col2
        with ax_col:
            vals = [results[s][metric] for s in stages]
            fig = go.Figure(go.Bar(
                x=stages, y=vals,
                marker_color=colors_m[metric], opacity=0.85,
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(
                title=labels[metric], height=320,
                yaxis=dict(range=[0, 1.1]),
                margin=dict(t=40, b=60, l=20, r=20),
                xaxis=dict(tickangle=-20),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lungime medie și emailuri degenerate")
    wc  = [results[s]["avg_words"]      for s in stages]
    deg = [results[s]["pct_degenerate"] for s in stages]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Cuvinte medii / email", "% Emailuri degenerate (<80 cuv.)"])
    fig.add_bar(x=stages, y=wc,  marker_color="#607D8B",
                text=[f"{v:.0f}" for v in wc], textposition="outside", row=1, col=1)
    fig.add_bar(x=stages, y=deg, marker_color="#F44336",
                text=[f"{v:.1f}%" for v in deg], textposition="outside", row=1, col=2)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=50, b=60))
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: GRPO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Antrenare GRPO":
    st.title("🤖 Antrenare GRPO — Convergență și Impact")

    # Convergence
    conv = load_json(OUTPUTS / "grpo_convergence.json")
    grpo = load_json(OUTPUTS / "grpo_eval.json")

    if conv:
        st.subheader("Curbă de convergență")
        results = conv["results"]
        steps = sorted(int(k) for k in results.keys())
        metrics = ["reward", "quality", "diversity", "format"]
        labels  = {"reward":"Reward total","quality":"Calitate",
                   "diversity":"Diversitate","format":"Format"}
        colors_c = {"reward":"#1f77b4","quality":"#ff7f0e","diversity":"#2ca02c","format":"#d62728"}

        fig = go.Figure()
        for m in metrics:
            vals = [results[str(s)][m] for s in steps]
            x_labels = ["Base" if s == 0 else str(s) for s in steps]
            fig.add_scatter(
                x=x_labels, y=vals, mode="lines+markers+text",
                name=labels[m], text=[f"{v:.3f}" for v in vals],
                textposition="top center",
                line=dict(color=colors_c[m], width=2), marker=dict(size=10),
            )
        fig.update_layout(
            xaxis_title="Pași GRPO", yaxis_title="Scor mediu",
            height=400, yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        tbl_rows = []
        for s in steps:
            r = results[str(s)]
            tbl_rows.append({"Pași": "Base" if s == 0 else s,
                              **{labels[m]: round(r[m], 4) for m in metrics}})
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    if grpo:
        st.subheader("Evaluare finală (step 600) — Base vs. GRPO")
        summary = grpo.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Reward base",  f"{summary.get('avg_base_reward', 0):.4f}")
        c2.metric("Reward GRPO",  f"{summary.get('avg_grpo_reward', 0):.4f}",
                  f"Δ {summary.get('avg_grpo_reward',0)-summary.get('avg_base_reward',0):+.4f}")
        c3.metric("N prompturi",  summary.get("n_samples", 20))

        # Per-locale impact
        per_locale = load_json(OUTPUTS / "per_locale_analysis" / "per_locale_results.json")
        if per_locale and per_locale.get("grpo_impact"):
            st.subheader("Impact GRPO per limbă")
            gi = per_locale["grpo_impact"]
            rows = [{"Limbă": loc, "Base": v["base"], "GRPO": v["grpo"],
                     "Δ": v["delta"]} for loc, v in gi.items()]
            df_gi = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_bar(x=df_gi["Limbă"], y=df_gi["Base"],  name="Base model",
                        marker_color="#607D8B", opacity=0.85)
            fig.add_bar(x=df_gi["Limbă"], y=df_gi["GRPO"],  name="GRPO fine-tuned",
                        marker_color="#F44336", opacity=0.85)
            for _, row in df_gi.iterrows():
                fig.add_annotation(
                    x=row["Limbă"], y=max(row["Base"], row["GRPO"]) + 0.02,
                    text=f"Δ{row['Δ']:+.4f}",
                    showarrow=False, font=dict(size=11, color="green" if row["Δ"] > 0 else "red"),
                )
            fig.update_layout(barmode="group", height=380,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              yaxis=dict(range=[0, 0.8]), margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        # Per fraud_stage
        stage_data = load_json(OUTPUTS / "fraud_stage_analysis" / "fraud_stage_results.json")
        if stage_data and stage_data.get("grpo_impact"):
            st.subheader("Impact GRPO per fraud stage")
            gi = stage_data["grpo_impact"]
            rows = [{"Stage": s, "Base": v["base"], "GRPO": v["grpo"],
                     "Δ": v["delta"], "N": v["n"]} for s, v in gi.items()]
            df_s = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_bar(x=df_s["Stage"], y=df_s["Base"],  name="Base",
                        marker_color="#607D8B", opacity=0.85)
            fig.add_bar(x=df_s["Stage"], y=df_s["GRPO"],  name="GRPO",
                        marker_color="#F44336", opacity=0.85)
            for _, row in df_s.iterrows():
                fig.add_annotation(
                    x=row["Stage"], y=max(row["Base"], row["GRPO"]) + 0.02,
                    text=f"Δ{row['Δ']:+.4f}",
                    showarrow=False, font=dict(size=12, color="green" if row["Δ"] > 0 else "red"),
                )
            fig.update_layout(barmode="group", height=350,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              yaxis=dict(range=[0, 0.7]), margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_s.round(4), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Adversarial Eval
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚔️ Evaluare Adversarială":
    st.title("⚔️ Evaluare Adversarială")
    st.markdown("""
    **Întrebarea cheie**: Emailurile phishing generate de modelul GRPO fine-tunat
    sunt mai greu de detectat de un clasificator antrenat pe phishing standard?
    """)

    adv = load_json(OUTPUTS / "adversarial_eval" / "adversarial_results.json")
    if not adv:
        st.warning("adversarial_results.json nu a fost găsit.")
        st.stop()

    bl = adv["baseline"]
    ad = adv["adversarial"]
    dl = adv["delta"]

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FNR Baseline",    f"{bl['fnr']:.4f}", "Phishing standard")
    c2.metric("FNR Adversarial", f"{ad['fnr']:.4f}",
              f"+{dl['fnr']:.4f} față de baseline",
              delta_color="inverse")
    c3.metric("F1 Baseline",     f"{bl['f1_phishing']:.4f}")
    c4.metric("F1 Adversarial",  f"{ad['f1_phishing']:.4f}",
              f"{dl['f1_phishing']:+.4f}", delta_color="inverse")

    st.markdown("---")

    metrics = ["f1_macro","f1_phishing","precision","recall","fnr","auc_roc"]
    labels  = {"f1_macro":"F1-macro","f1_phishing":"F1-phishing",
                "precision":"Precision","recall":"Recall","fnr":"FNR","auc_roc":"AUC-ROC"}

    fig = go.Figure()
    fig.add_bar(x=[labels[m] for m in metrics],
                y=[bl[m] for m in metrics],
                name="Test baseline (phishing standard)",
                marker_color="#2196F3", opacity=0.85,
                text=[f"{bl[m]:.3f}" for m in metrics], textposition="outside")
    fig.add_bar(x=[labels[m] for m in metrics],
                y=[ad[m] for m in metrics],
                name="Test adversarial (phishing GRPO)",
                marker_color="#F44336", opacity=0.85,
                text=[f"{ad[m]:.3f}" for m in metrics], textposition="outside")
    fig.update_layout(
        barmode="group", height=420,
        yaxis=dict(range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=20),
        title="Clasificator XLM-RoBERTa: Baseline vs. Adversarial",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretare")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **FNR crește cu {dl['fnr']:+.4f}** (de la {bl['fnr']:.0%} la {ad['fnr']:.0%})

        Clasificatorul antrenat pe phishing standard **ratează {ad['fnr']:.0%}**
        din emailurile phishing generate de GRPO — față de **0% pentru phishing clasic**.
        """)
    with col2:
        st.warning(f"""
        **Recall scade cu {dl['recall']:+.4f}** ({bl['recall']:.4f} → {ad['recall']:.4f})

        Precision rămâne 1.0 — clasificatorul nu produce fals pozitive,
        dar **omite mai mult de jumătate** din phishing-ul GRPO.
        """)

    # Per-stage FNR (dacă există)
    stage_data = load_json(OUTPUTS / "fraud_stage_analysis" / "fraud_stage_results.json")
    if stage_data and stage_data.get("fnr_baseline"):
        st.subheader("FNR per fraud stage")
        stages = list(stage_data["fnr_baseline"].keys())
        fnr_b = [stage_data["fnr_baseline"].get(s,{}).get("fnr",0) for s in stages]
        fnr_a = [stage_data["fnr_adversarial"].get(s,{}).get("fnr",0) for s in stages]

        fig2 = go.Figure()
        fig2.add_bar(x=stages, y=fnr_b, name="Baseline",    marker_color="#2196F3", opacity=0.85,
                     text=[f"{v:.3f}" for v in fnr_b], textposition="outside")
        fig2.add_bar(x=stages, y=fnr_a, name="Adversarial", marker_color="#FF5722", opacity=0.85,
                     text=[f"{v:.3f}" for v in fnr_a], textposition="outside")
        fig2.update_layout(
            barmode="group", height=380,
            yaxis=dict(range=[0, 1.15]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Adversarial loop results (dacă există)
    loop_data = load_json(OUTPUTS / "adversarial_loop" / "adversarial_loop_results.json")
    if loop_data:
        st.subheader("🔄 Joc Adversarial Iterativ — 3 Runde")
        st.markdown("""
        Clasificatorul XLM-RoBERTa este reantrenat cu emailuri GRPO acumulate per rundă.
        FNR-ul măsoară dacă clasificatorul se adaptează sau rămâne vulnerabil.
        """)

        rounds = loop_data.get("rounds", [])
        if rounds:
            round_labels = [f"Runda {r['round']}" for r in rounds]
            fnr_r = [r["fnr"]           for r in rounds]
            f1_r  = [r["f1_phishing"]   for r in rounds]
            rec_r = [r["recall"]        for r in rounds]
            n_grpo = [r.get("n_grpo_in_train", 0) for r in rounds]

            kc = st.columns(len(rounds))
            for i, (col, r) in enumerate(zip(kc, rounds)):
                delta_str = ""
                if i > 0:
                    d = r["fnr"] - rounds[i-1]["fnr"]
                    delta_str = f"{'↓' if d < 0 else '↑'}{abs(d):.4f}"
                col.metric(f"Runda {r['round']} FNR", f"{r['fnr']:.4f}",
                           delta_str if delta_str else None,
                           delta_color="normal" if (i > 0 and rounds[i]["fnr"] < rounds[i-1]["fnr"]) else "inverse")

            fig_l = go.Figure()
            fig_l.add_scatter(x=round_labels, y=fnr_r, mode="lines+markers+text",
                              name="FNR", line=dict(color="#F44336", width=2.5),
                              marker=dict(size=12), text=[f"{v:.4f}" for v in fnr_r],
                              textposition="top center")
            fig_l.add_scatter(x=round_labels, y=f1_r, mode="lines+markers+text",
                              name="F1-phishing", line=dict(color="#4CAF50", width=2.5),
                              marker=dict(size=12), text=[f"{v:.4f}" for v in f1_r],
                              textposition="bottom center")
            fig_l.add_scatter(x=round_labels, y=rec_r, mode="lines+markers",
                              name="Recall", line=dict(color="#2196F3", width=2, dash="dot"),
                              marker=dict(size=9))
            fig_l.update_layout(
                title="Adaptarea clasificatorului pe 3 runde adversariale",
                height=400, yaxis=dict(range=[0, 1.1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60),
            )
            st.plotly_chart(fig_l, use_container_width=True)

            # Escalation: FNR vs N GRPO in training
            fig_e = make_subplots(specs=[[{"secondary_y": True}]])
            fig_e.add_bar(x=round_labels, y=n_grpo, name="GRPO în antrenare",
                          marker_color="#9C27B0", opacity=0.25, secondary_y=True)
            fig_e.add_scatter(x=round_labels, y=fnr_r, mode="lines+markers",
                              name="FNR", line=dict(color="#F44336", width=2.5),
                              marker=dict(size=12), secondary_y=False)
            fig_e.update_yaxes(title_text="FNR", secondary_y=False, range=[0, 1.1])
            fig_e.update_yaxes(title_text="N GRPO în antrenare", secondary_y=True)
            fig_e.update_layout(
                title="Escalation curve: FNR vs. experiența clasificatorului",
                height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60),
            )
            st.plotly_chart(fig_e, use_container_width=True)

            tbl_r = pd.DataFrame([
                {"Runda": r["round"], "GRPO în train": r.get("n_grpo_in_train", 0),
                 "FNR": r["fnr"], "Recall": r["recall"], "F1-phishing": r["f1_phishing"],
                 "AUC-ROC": r.get("auc_roc", 0), "N test phishing": r["n_phishing"]}
                for r in rounds
            ])
            st.dataframe(tbl_r.round(4), use_container_width=True, hide_index=True)
    else:
        st.info("adversarial_loop_results.json nu a fost găsit. Rulează experiments/adversarial_loop.py.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Per Limbă
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌍 Analiză per Limbă":
    st.title("🌍 Analiză per Limbă")

    data = load_json(OUTPUTS / "per_locale_analysis" / "per_locale_results.json")
    if not data:
        st.warning("per_locale_results.json nu a fost găsit.")
        st.stop()

    gen = data.get("generation", {})
    rew = data.get("reward", {})
    gri = data.get("grpo_impact", {})

    tab1, tab2, tab3 = st.tabs(["Self-Correction", "Reward Heuristic", "Impact GRPO"])

    with tab1:
        st.subheader("Calitate self-correction per limbă")
        rows = [{"Limbă": loc,
                 "N": gen.get(loc,{}).get("n",0),
                 "Scor mediu": gen.get(loc,{}).get("avg_score",0),
                 "Std. Dev.": gen.get(loc,{}).get("std_score",0),
                 "Rata acceptare %": gen.get(loc,{}).get("acceptance_rate",0),
                 "Iterații medii": gen.get(loc,{}).get("avg_iters",0),
                 "% Multi-iter": gen.get(loc,{}).get("pct_multi_iter",0)}
                for loc in LOCALES if loc in gen]
        df = pd.DataFrame(rows)

        fig = go.Figure()
        scores = [gen.get(l,{}).get("avg_score",0) for l in LOCALES]
        stds   = [gen.get(l,{}).get("std_score",0) for l in LOCALES]
        fig.add_bar(x=LOCALES, y=scores, error_y=dict(array=stds),
                    marker_color=[COLORS[l] for l in LOCALES], opacity=0.85,
                    text=[f"{v:.3f}" for v in scores], textposition="outside")
        fig.add_hline(y=6.0, line_dash="dash", line_color="red",
                      annotation_text="Prag acceptare (6.0)")
        fig.update_layout(height=380, yaxis=dict(range=[0,10]),
                          title="Scor self-correction per limbă", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.round(3), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Reward heuristic per limbă (phishing)")
        r_vals = [rew.get(l,{}).get("avg_reward",0) for l in LOCALES]
        q_vals = [rew.get(l,{}).get("avg_quality",0) for l in LOCALES]
        x = list(range(len(LOCALES)))
        fig = go.Figure()
        fig.add_bar(x=LOCALES, y=r_vals, name="Reward total",
                    marker_color="#9C27B0", opacity=0.85,
                    text=[f"{v:.4f}" for v in r_vals], textposition="outside")
        fig.add_bar(x=LOCALES, y=q_vals, name="Quality score",
                    marker_color="#E91E63", opacity=0.85,
                    text=[f"{v:.4f}" for v in q_vals], textposition="outside")
        fig.update_layout(barmode="group", height=380,
                          yaxis=dict(range=[0, 0.8]),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Impact GRPO per limbă")
        if not gri:
            st.info("grpo_impact nu este disponibil.")
        else:
            base = [gri.get(l,{}).get("base",0) for l in LOCALES]
            grpo = [gri.get(l,{}).get("grpo",0) for l in LOCALES]
            dlts = [gri.get(l,{}).get("delta",0) for l in LOCALES]
            fig = go.Figure()
            fig.add_bar(x=LOCALES, y=base, name="Base model",
                        marker_color="#607D8B", opacity=0.85)
            fig.add_bar(x=LOCALES, y=grpo, name="GRPO fine-tuned",
                        marker_color="#F44336", opacity=0.85)
            for i, (loc, d) in enumerate(zip(LOCALES, dlts)):
                fig.add_annotation(
                    x=loc, y=max(base[i], grpo[i]) + 0.02,
                    text=f"Δ{d:+.4f}",
                    showarrow=False, font=dict(size=11, color="green" if d > 0 else "red"),
                )
            fig.update_layout(barmode="group", height=400,
                              yaxis=dict(range=[0, 0.8]),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Cross-Locale Transfer
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌐 Transfer Cross-Locale":
    st.title("🌐 Transferabilitate Cross-Locale")
    st.markdown("""
    **Întrebarea cheie**: Un clasificator antrenat exclusiv pe date en-US se poate generaliza
    la celelalte 4 limbi (ro-RO, de-DE, fr-FR, it-IT)?
    """)

    data = load_json(OUTPUTS / "cross_locale_transfer" / "cross_locale_results.json")
    if not data:
        st.warning("cross_locale_results.json nu a fost găsit. Rulează experiments/cross_locale_transfer.py.")
        st.stop()

    summary = data.get("summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("F1 mediu (en-only)",      f"{summary.get('en_avg_f1', 0):.4f}")
    c2.metric("F1 mediu (multilingual)",  f"{summary.get('multi_avg_f1', 0):.4f}",
              f"+{summary.get('gap', 0):.4f} față de en-only")
    c3.metric("Gap multilingual − en-only", f"{summary.get('gap', 0):.4f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["en-only vs. Multilingual", "Matrice Cross-Locale", "Tabele detaliate"])

    with tab1:
        st.subheader("Comparație en-only vs. multilingual per limbă")
        en_only = data.get("en_only", {})
        multi   = data.get("multilingual", {})
        locales_avail = [l for l in LOCALES if l in en_only]

        for metric, title, yrange in [
            ("f1_phishing", "F1-phishing per limbă", [0.95, 1.02]),
            ("fnr",         "FNR per limbă",          [0.0, 0.06]),
        ]:
            fig = go.Figure()
            en_vals = [en_only.get(l, {}).get(metric, 0) for l in locales_avail]
            mu_vals = [multi.get(l, {}).get(metric, 0) for l in locales_avail]
            fig.add_bar(x=locales_avail, y=en_vals, name="en-only",
                        marker_color="#1976D2", opacity=0.85,
                        text=[f"{v:.4f}" for v in en_vals], textposition="outside")
            fig.add_bar(x=locales_avail, y=mu_vals, name="multilingual",
                        marker_color="#43A047", opacity=0.85,
                        text=[f"{v:.4f}" for v in mu_vals], textposition="outside")
            fig.update_layout(
                barmode="group", title=title, height=370,
                yaxis=dict(range=yrange),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Matrice cross-locale — F1-phishing")
        st.markdown("Fiecare rând = limbă de antrenare, fiecare coloană = limbă de test.")
        matrix = data.get("per_locale_matrix", {})
        if not matrix:
            st.info("Matricea cross-locale nu a fost calculată (rulează fără --quick).")
        else:
            for metric, title in [("f1_phishing", "F1-phishing"), ("fnr", "FNR")]:
                train_locales = [l for l in LOCALES if l in matrix]
                test_locales  = LOCALES
                z = []
                for tr in train_locales:
                    row = []
                    for te in test_locales:
                        row.append(matrix.get(tr, {}).get(te, {}).get(metric, None))
                    z.append(row)

                colorscale = "Blues" if metric == "f1_phishing" else "Reds"
                zmin, zmax = (0.9, 1.0) if metric == "f1_phishing" else (0.0, 0.15)

                fig = go.Figure(go.Heatmap(
                    z=z, x=test_locales, y=train_locales,
                    colorscale=colorscale, zmin=zmin, zmax=zmax,
                    text=[[f"{v:.3f}" if v is not None else "N/A" for v in row] for row in z],
                    texttemplate="%{text}",
                    textfont=dict(size=13),
                    hoverongaps=False,
                ))
                fig.update_layout(
                    title=f"Matrice cross-locale — {title}",
                    xaxis_title="Limbă test", yaxis_title="Limbă antrenare",
                    height=380, margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Tabel detaliat — en-only")
        en_rows = [
            {
                "Limbă test": loc,
                "F1-phishing": round(en_only.get(loc, {}).get("f1_phishing", 0), 4),
                "Precision":   round(en_only.get(loc, {}).get("precision", 0), 4),
                "Recall":      round(en_only.get(loc, {}).get("recall", 0), 4),
                "FNR":         round(en_only.get(loc, {}).get("fnr", 0), 4),
                "AUC-ROC":     round(en_only.get(loc, {}).get("auc_roc", 0), 4),
                "N test":      en_only.get(loc, {}).get("n", 0),
            }
            for loc in LOCALES if loc in en_only
        ]
        st.dataframe(pd.DataFrame(en_rows), use_container_width=True, hide_index=True)

        st.subheader("Tabel detaliat — multilingual")
        mu_rows = [
            {
                "Limbă test": loc,
                "F1-phishing": round(multi.get(loc, {}).get("f1_phishing", 0), 4),
                "Precision":   round(multi.get(loc, {}).get("precision", 0), 4),
                "Recall":      round(multi.get(loc, {}).get("recall", 0), 4),
                "FNR":         round(multi.get(loc, {}).get("fnr", 0), 4),
                "AUC-ROC":     round(multi.get(loc, {}).get("auc_roc", 0), 4),
                "N test":      multi.get(loc, {}).get("n", 0),
            }
            for loc in LOCALES if loc in multi
        ]
        st.dataframe(pd.DataFrame(mu_rows), use_container_width=True, hide_index=True)

        st.subheader("Interpretare")
        st.success(f"""
        **Concluzie**: XLM-RoBERTa antrenat exclusiv pe en-US atinge F1={summary.get('en_avg_f1',0):.4f}
        pe toate cele 5 limbi. Singurul punct slab este **de-DE** (FNR=2.5%), unde modelul multilingual
        corectează la FNR=0%. Gap total multilingual − en-only = **{summary.get('gap',0):.4f}** F1 mediu.

        Aceasta demonstrează că **pattern-urile semantice de phishing se transferă cross-lingvistic**
        prin embeddings-urile cross-lingual ale XLM-RoBERTa, fără date de antrenare în limba țintă.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8: Explainability
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Explainability":
    st.title("🔍 Explainability: De ce F1 ≈ 1?")
    st.markdown(
        "Analiză Integrated Gradients + LIME pe clasificatorul XLM-RoBERTa "
        "($n_{\\text{SFT}}=1000$). Verificăm dacă clasificatorul detectează "
        "**artefacte triviale** sau **vocabular phishing semantic real**."
    )

    expl = load_json(OUTPUTS / "explainability" / "explainability_results.json")

    if not expl:
        st.warning("Rulează `experiments/explainability_analysis.py` pentru a genera rezultatele.")
    else:
        cfg = expl.get("config", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("n_SFT (antrenare)", cfg.get("n_sft", 1000))
        c2.metric("Emailuri analizate / tip", cfg.get("n_samples", 20))
        c3.metric("Metode", cfg.get("method", "both").upper())

        tab1, tab2, tab3 = st.tabs(["🎯 Integrated Gradients", "🍋 LIME", "🔬 Analiza Artefacte"])

        with tab1:
            st.subheader("Integrated Gradients — Top Tokeni per Tip Email")
            ig = expl.get("ig", {})
            if ig:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Phishing Standard**")
                    std_tokens = ig.get("standard_top_tokens", {})
                    rows = [{"Token": k, "Score mediu": round(v["mean"], 4),
                             "Frecvență": v["freq"]} for k, v in list(std_tokens.items())[:15]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                with col2:
                    st.markdown("**Phishing GRPO**")
                    grpo_tokens = ig.get("grpo_top_tokens", {})
                    rows = [{"Token": k, "Score mediu": round(v["mean"], 4),
                             "Frecvență": v["freq"]} for k, v in list(grpo_tokens.items())[:15]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Bar chart comparativ
                std_top  = list(std_tokens.items())[:10]
                grpo_top = list(grpo_tokens.items())[:10]
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Standard phishing", "GRPO phishing"])
                fig.add_bar(x=[v["mean"] for _, v in std_top],  y=[k for k, _ in std_top],
                            orientation="h", marker_color="#2196F3", row=1, col=1)
                fig.add_bar(x=[v["mean"] for _, v in grpo_top], y=[k for k, _ in grpo_top],
                            orientation="h", marker_color="#F44336", row=1, col=2)
                fig.update_layout(height=400, showlegend=False,
                                  title_text="IG: Importanța tokenilor per tip email")
                st.plotly_chart(fig, use_container_width=True)

                analysis = expl.get("analysis", {})
                if analysis.get("tokens_unique_to_grpo"):
                    st.info(f"**Tokeni unici GRPO** (față de standard): "
                            f"`{'`, `'.join(analysis['tokens_unique_to_grpo'][:10])}`")

        with tab2:
            st.subheader("LIME — Top Cuvinte per Tip Email")
            lime = expl.get("lime", {})
            if lime:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Phishing Standard**")
                    std_w = sorted(lime.get("standard_top_words", {}).items(), key=lambda x: -x[1])
                    rows = [{"Cuvânt": k, "Score LIME": round(v, 4)} for k, v in std_w[:15]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                with col2:
                    st.markdown("**Phishing GRPO**")
                    grpo_w = sorted(lime.get("grpo_top_words", {}).items(), key=lambda x: -x[1])
                    rows = [{"Cuvânt": k, "Score LIME": round(v, 4)} for k, v in grpo_w[:15]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Comparație vizuală
                std15  = std_w[:10]
                grpo15 = grpo_w[:10]
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Standard phishing", "GRPO phishing"])
                fig.add_bar(x=[v for _, v in std15],  y=[k for k, _ in std15],
                            orientation="h", marker_color="#4CAF50", row=1, col=1)
                fig.add_bar(x=[v for _, v in grpo15], y=[k for k, _ in grpo15],
                            orientation="h", marker_color="#FF9800", row=1, col=2)
                fig.update_layout(height=400, showlegend=False,
                                  title_text="LIME: Cuvinte influente per tip email")
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "**Concluzie**: clasificatorul detectează vocabular phishing semantic "
                    "(security, privacy, gdpr, identitatea) — **nu** artefacte de prompt. "
                    "F1≈1 reflectă consistența stilistică a datelor sintetice."
                )

        with tab3:
            st.subheader("Analiza Artefactelor în Emailurile GRPO")
            st.markdown(
                "Verificăm ce proporție din emailurile GRPO conțin text contaminat "
                "cu instrucțiunile de prompt (liste de cuvinte, directive de generare)."
            )
            artifact_img = OUTPUTS.parent / "raport4" / "pics" / "grpo_artifact_analysis.png"
            if artifact_img.exists():
                st.image(str(artifact_img), use_column_width=True)
            else:
                st.info("Graficul grpo_artifact_analysis.png nu a fost găsit.")

            st.markdown("""
**Interpretare**:
- **69%** din emailurile GRPO sunt curate (text email coerent, fără instrucțiuni scurse)
- **31%** conțin artefacte: liste de cuvinte-cheie, directive de generare, markeri de format
- Tokenii detectați de clasificator provin din **vocabular semantic** (gdpr, security, privacy),
  **nu** din artefactele de prompt — validând că F1≈1 nu este un fals-pozitiv metodologic
""")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9: Lingvistică
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📝 Analiză Lingvistică":
    st.title("📝 Analiză Lingvistică")
    st.markdown("Caracteristici lingvistice ale emailurilor phishing vs. ham legitim.")

    data = load_json(OUTPUTS / "linguistic_analysis" / "linguistic_results.json")
    if not data:
        st.warning("linguistic_results.json nu a fost găsit.")
        st.stop()

    glb = data.get("global", {})
    ph  = glb.get("phishing", {})
    hm  = glb.get("ham", {})

    # Global comparison
    st.subheader("Phishing vs. Ham — Global")
    metrics = ["avg_words","avg_ttr","urgency_density","authority_density","threat_density"]
    labels  = {"avg_words":"Cuvinte medii","avg_ttr":"TTR (vocabular)",
                "urgency_density":"Densitate urgency","authority_density":"Densitate authority",
                "threat_density":"Densitate threat"}
    c1, c2 = st.columns(2)
    for i, (m, lbl) in enumerate(labels.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            ph_v = ph.get(m, 0)
            hm_v = hm.get(m, 0)
            fig = go.Figure()
            fig.add_bar(x=["Phishing","Ham"], y=[ph_v, hm_v],
                        marker_color=["#E53935","#1976D2"], opacity=0.85,
                        text=[f"{ph_v:.2f}", f"{hm_v:.2f}"], textposition="outside")
            fig.update_layout(title=lbl, height=280,
                              yaxis=dict(range=[0, max(ph_v, hm_v) * 1.3 + 0.001]),
                              margin=dict(t=50, b=20))
            col.plotly_chart(fig, use_container_width=True)

    # Tabel global
    st.subheader("Tabel comparativ global")
    rows = [{"Metrică": labels.get(m, m),
             "Phishing": round(ph.get(m,0), 4),
             "Ham": round(hm.get(m,0), 4),
             "Δ (Ph-Ham)": round(ph.get(m,0) - hm.get(m,0), 4)}
            for m in metrics]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per locale
    st.subheader("Per limbă: urgency și authority density")
    per_ph  = data.get("per_locale_phishing", {})
    per_ham = data.get("per_locale_ham", {})
    locales = sorted(per_ph.keys())

    for density_metric, title in [("urgency_density","Densitate Urgency"),
                                   ("authority_density","Densitate Authority")]:
        ph_vals  = [per_ph.get(l,{}).get(density_metric,0) for l in locales]
        ham_vals = [per_ham.get(l,{}).get(density_metric,0) for l in locales]
        fig = go.Figure()
        fig.add_bar(x=locales, y=ph_vals,  name="Phishing",   marker_color="#E53935", opacity=0.85,
                    text=[f"{v:.2f}" for v in ph_vals], textposition="outside")
        fig.add_bar(x=locales, y=ham_vals, name="Ham legitim", marker_color="#1976D2", opacity=0.85,
                    text=[f"{v:.2f}" for v in ham_vals], textposition="outside")
        fig.update_layout(barmode="group", title=f"{title} per limbă",
                          height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
