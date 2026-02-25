import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Nov 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f1a; color: #e0e0f0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2d2d5e;
    }
    [data-testid="stSidebar"] * { color: #c0c0e0 !important; }
    
    /* Cards / Metric boxes */
    [data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #818cf8 !important; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e3a, #2a2a52);
        border: 1px solid #3d3d7a;
        border-radius: 12px;
        padding: 16px !important;
    }
    
    /* Headers */
    h1 { color: #a78bfa !important; }
    h2, h3 { color: #818cf8 !important; }
    
    /* Scenario cards */
    .scenario-card {
        background: linear-gradient(135deg, #1e1e3a, #2a2a52);
        border: 1px solid #3d3d7a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .scenario-card h3 { color: #a78bfa !important; margin: 0 0 12px 0; }
    .scenario-card .val { font-size: 1.8rem; font-weight: 700; color: #c4b5fd; }
    .scenario-card .sub { font-size: 1rem; color: #818cf8; margin-top: 6px; }
    
    /* Section separator */
    .section-sep { border: none; border-top: 1px solid #2d2d5e; margin: 24px 0; }
    
    /* Big button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    
    /* Black friday row highlight */
    .bf-row { background-color: rgba(239, 68, 68, 0.15) !important; }
    
    /* Dataframe scroll */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_final_forecasting.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "inferencia_df_transformado.csv")

# ─── LOAD RESOURCES ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

model_ok = os.path.exists(MODEL_PATH)
data_ok  = os.path.exists(DATA_PATH)

if not model_ok or not data_ok:
    st.error("❌ No se encontró el modelo o los datos. Verifica las rutas.")
    if not model_ok: st.error(f"  Modelo esperado en: `{MODEL_PATH}`")
    if not data_ok:  st.error(f"  Datos esperados en: `{DATA_PATH}`")
    st.stop()

model    = load_model()
df_full  = load_data()
FEATURES = model.feature_names_in_.tolist()

# ─── PRODUCT LIST ───────────────────────────────────────────────────────────────
if "nombre" in df_full.columns:
    prods_df = df_full[["producto_id", "nombre"]].drop_duplicates().sort_values("nombre")
    prod_names = prods_df["nombre"].tolist()
    prod_map   = dict(zip(prods_df["nombre"], prods_df["producto_id"]))
else:
    prod_names = sorted(df_full["producto_id"].unique().tolist())
    prod_map   = {p: p for p in prod_names}

DAY_NAMES_ES = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]

# ─── RECURSIVE PREDICTION FUNCTION ──────────────────────────────────────────────
def run_recursive_prediction(df_prod, descuento_adj_pct, comp_adj_pct):
    """
    Runs recursive day-by-day prediction for one product over 30 November days.
    Returns a DataFrame with columns: fecha, dia, dia_semana_nombre, precio_venta,
    precio_competencia, descuento_pct, unidades, ingresos.
    """
    df = df_prod.copy().sort_values("fecha").reset_index(drop=True)

    lag_cols   = [f"unidades_vendidas_lag{i}" for i in range(1, 8)]  # lag1..lag7
    MISSING    = [c for c in lag_cols if c not in df.columns]

    results = []
    predictions_so_far = []

    for i, row in df.iterrows():
        row = row.copy()

        # ── 1. Adjust prices ──────────────────────────────────────────────────
        precio_base      = row["precio_base"]
        precio_venta_new = precio_base * (1 + descuento_adj_pct / 100)

        if "precio_min_competencia" in row.index:
            precio_comp_new = row["precio_min_competencia"] * (1 + comp_adj_pct / 100)
        else:
            precio_comp_new = precio_venta_new  # fallback

        row["precio_venta"]            = precio_venta_new
        row["precio_min_competencia"]  = precio_comp_new
        row["prec_competencia"]        = precio_comp_new
        row["descuento_porcentaje"]    = ((precio_base - precio_venta_new) / precio_base) * 100 if precio_base != 0 else 0
        row["ratio_competencia"]       = (precio_venta_new / precio_comp_new) if precio_comp_new != 0 else 1
        row["ratio_precio"]            = row["ratio_competencia"]
        row["gap_precio"]              = precio_venta_new - precio_comp_new if "gap_precio" in row.index else 0
        row["gap_porcentaje"]          = ((row["gap_precio"] / precio_comp_new) * 100) if precio_comp_new != 0 else 0
        row["promo_rival"]             = bool(row["gap_porcentaje"] > 10) if "promo_rival" in row.index else False
        row["margen_unitario"]         = precio_venta_new - precio_base if "margen_unitario" in row.index else 0

        # ── 2. Update lags recursively (days 2-30) ───────────────────────────
        if i > 0 and len(predictions_so_far) > 0:
            # Shift lags: lag2←lag1, lag3←lag2, ..., lag7←lag6
            for lag_n in range(7, 1, -1):
                col_n    = f"unidades_vendidas_lag{lag_n}"
                col_prev = f"unidades_vendidas_lag{lag_n - 1}"
                if col_n in df.columns and col_prev in df.columns:
                    row[col_n] = df.at[i, col_prev]  # yesterday's lag(n-1) before update

            # lag1 = previous day's prediction
            if "unidades_vendidas_lag1" in row.index:
                row["unidades_vendidas_lag1"] = predictions_so_far[-1]

            # ventas_lag1 / ventas_lag7
            if "ventas_lag1" in row.index:
                row["ventas_lag1"] = predictions_so_far[-1]
            if "ventas_lag7" in row.index and len(predictions_so_far) >= 7:
                row["ventas_lag7"] = predictions_so_far[-7]

            # Update media_movil_7
            if "media_movil_7" in row.index:
                last7 = predictions_so_far[-7:]
                row["media_movil_7"] = np.mean(last7)

        # ── 3. Predict ────────────────────────────────────────────────────────
        X = pd.DataFrame([row])[FEATURES].fillna(0)
        pred = float(model.predict(X)[0])
        pred = max(0, pred)
        predictions_so_far.append(pred)

        # ── 4. Store result ───────────────────────────────────────────────────
        results.append({
            "fecha":             row["fecha"],
            "dia":               int(row["fecha"].day),
            "dia_semana_nombre": DAY_NAMES_ES[int(row["fecha"].dayofweek)],
            "precio_venta":      round(precio_venta_new, 2),
            "precio_competencia":round(precio_comp_new, 2),
            "descuento_pct":     round(row["descuento_porcentaje"], 2),
            "unidades":          round(pred, 1),
            "ingresos":          round(pred * precio_venta_new, 2),
        })

    return pd.DataFrame(results)

# ─── RUN ALL 3 SCENARIOS ─────────────────────────────────────────────────────────
def run_all_scenarios(df_prod, descuento_adj_pct):
    scenarios = {
        "Actual (0%)":       0,
        "Competencia -5%":  -5,
        "Competencia +5%":  +5,
    }
    return {name: run_recursive_prediction(df_prod, descuento_adj_pct, adj)
            for name, adj in scenarios.items()}

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 Controles de Simulación")
    st.markdown("---")

    selected_prod = st.selectbox(
        "🛍️ Producto",
        options=prod_names,
        help="Selecciona el producto a simular"
    )

    st.markdown("---")

    descuento_adj = st.slider(
        "🏷️ Ajuste de Descuento (%)",
        min_value=-50, max_value=50, value=0, step=5,
        help="Ajusta el precio de venta sobre el precio base"
    )

    st.markdown("---")

    comp_scenario = st.radio(
        "🏪 Escenario de Competencia",
        options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Ajusta los precios de la competencia"
    )

    comp_adj_map = {"Actual (0%)": 0, "Competencia -5%": -5, "Competencia +5%": +5}
    comp_adj = comp_adj_map[comp_scenario]

    st.markdown("---")
    simulate_btn = st.button("🚀 Simular Ventas", use_container_width=True)

    # Show current settings
    st.markdown("---")
    st.markdown("**Configuración actual:**")
    prod_id = prod_map.get(selected_prod, selected_prod)
    st.caption(f"📦 {selected_prod}")
    st.caption(f"🏷️ Descuento: {'±0' if descuento_adj == 0 else f'{descuento_adj:+d}'}%")
    st.caption(f"🏪 Competencia: {comp_scenario}")

# ─── MAIN AREA ───────────────────────────────────────────────────────────────────
st.markdown("# 📈 Dashboard de Forecasting — Noviembre 2025")
st.markdown(f"### Simulando: **{selected_prod}** | {comp_scenario} | Descuento: **{descuento_adj:+d}%**")
st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)

# State init
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "all_scenarios" not in st.session_state:
    st.session_state.all_scenarios = None
if "last_config" not in st.session_state:
    st.session_state.last_config = None

current_config = (selected_prod, descuento_adj, comp_scenario)

# Auto-run on first visit or Simulate button
if simulate_btn or st.session_state.results_df is None:
    with st.spinner("⚙️ Ejecutando predicciones recursivas..."):
        try:
            df_prod = df_full[df_full["producto_id"] == prod_map.get(selected_prod, selected_prod)].copy()
            if len(df_prod) == 0:
                st.error("No hay datos para el producto seleccionado.")
                st.stop()

            results_df       = run_recursive_prediction(df_prod, descuento_adj, comp_adj)
            all_scenarios    = run_all_scenarios(df_prod, descuento_adj)

            st.session_state.results_df    = results_df
            st.session_state.all_scenarios = all_scenarios
            st.session_state.last_config   = current_config
        except Exception as e:
            st.error(f"❌ Error durante la predicción: {e}")
            st.exception(e)
            st.stop()

results_df    = st.session_state.results_df
all_scenarios = st.session_state.all_scenarios

if results_df is None or len(results_df) == 0:
    st.info("👈 Pulsa **Simular Ventas** en el sidebar para comenzar.")
    st.stop()

# ─── KPIs ─────────────────────────────────────────────────────────────────────────
total_unidades  = results_df["unidades"].sum()
total_ingresos  = results_df["ingresos"].sum()
avg_precio      = results_df["precio_venta"].mean()
avg_descuento   = results_df["descuento_pct"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Unidades Totales",   f"{total_unidades:,.0f} u")
c2.metric("💶 Ingresos Proyectados", f"€ {total_ingresos:,.2f}")
c3.metric("💰 Precio Promedio",     f"€ {avg_precio:,.2f}")
c4.metric("🏷️ Descuento Promedio",  f"{avg_descuento:,.1f}%")

st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)

# ─── DAILY PREDICTION CHART ─────────────────────────────────────────────────────
st.markdown("### 📊 Predicción Diaria de Ventas")

sns.set_style("dark")
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#2d2d5e",
    "text.color":       "#c0c0e0",
    "xtick.color":      "#8888bb",
    "ytick.color":      "#8888bb",
    "grid.color":       "#2d2d5e",
    "axes.labelcolor":  "#c0c0e0",
})

fig, ax = plt.subplots(figsize=(14, 5))

# Main line
ax.plot(
    results_df["dia"], results_df["unidades"],
    color="#a78bfa", linewidth=2.5, marker="o",
    markersize=5, markerfacecolor="#7c3aed", markeredgewidth=0,
    zorder=3,
)

# Black Friday (day 28)
bf_row = results_df[results_df["dia"] == 28]
if not bf_row.empty:
    bf_val = float(bf_row["unidades"].values[0])
    ax.axvline(x=28, color="#ef4444", linestyle="--", linewidth=2, alpha=0.7, zorder=2, label="Black Friday")
    ax.scatter([28], [bf_val], color="#ef4444", s=150, zorder=5)
    ax.annotate(
        f"🖤 Black Friday\n{bf_val:.1f} u",
        xy=(28, bf_val),
        xytext=(25 if bf_val > results_df["unidades"].mean() else 28, bf_val * 1.15),
        color="#ef4444", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5),
        ha="center",
    )

# Weekend shading
for _, r in results_df.iterrows():
    if r["dia_semana_nombre"] in ["Sáb", "Dom"]:
        ax.axvspan(r["dia"] - 0.4, r["dia"] + 0.4, alpha=0.06, color="#818cf8", zorder=1)

ax.set_xlabel("Día de Noviembre", fontsize=12)
ax.set_ylabel("Unidades Proyectadas", fontsize=12)
ax.set_title(f"Ventas Diarias Proyectadas — {selected_prod}", fontsize=14, fontweight="bold", color="#a78bfa")
ax.set_xticks(results_df["dia"])
ax.set_xticklabels(results_df["dia"], fontsize=8)
ax.grid(axis="y", alpha=0.3)

bf_patch = mpatches.Patch(color="#ef4444", label="Black Friday (día 28)")
wk_patch = mpatches.Patch(color="#818cf8", alpha=0.2, label="Fin de semana")
ax.legend(handles=[bf_patch, wk_patch], facecolor="#1a1a2e", edgecolor="#3d3d7a", labelcolor="#c0c0e0", fontsize=9)

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)

# ─── DETAILED TABLE ─────────────────────────────────────────────────────────────
st.markdown("### 📋 Tabla Detallada de Noviembre")

table_df = results_df.copy()
table_df["fecha"] = table_df["fecha"].dt.strftime("%d %b %Y")

# Black Friday emoji highlight in column
table_df["Día"] = table_df.apply(
    lambda r: f"🖤 {r['dia']} {r['dia_semana_nombre']}" if r["dia"] == 28 else f"{r['dia']} {r['dia_semana_nombre']}",
    axis=1
)

display_df = table_df.rename(columns={
    "fecha":            "Fecha",
    "precio_venta":     "Precio Venta (€)",
    "precio_competencia":"Comp. Mín (€)",
    "descuento_pct":   "Descuento (%)",
    "unidades":         "Unidades",
    "ingresos":         "Ingresos (€)",
})[["Día", "Fecha", "Precio Venta (€)", "Comp. Mín (€)", "Descuento (%)", "Unidades", "Ingresos (€)"]]

st.dataframe(
    display_df.style
    .format({
        "Precio Venta (€)":   "€ {:.2f}",
        "Comp. Mín (€)":      "€ {:.2f}",
        "Descuento (%)":      "{:.1f}%",
        "Unidades":           "{:.1f}",
        "Ingresos (€)":       "€ {:.2f}",
    })
    .apply(lambda row: ["background-color: rgba(239,68,68,0.15)" if "🖤" in str(row["Día"]) else "" for _ in row], axis=1),
    use_container_width=True,
    height=400,
)

st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)

# ─── SCENARIO COMPARISON ────────────────────────────────────────────────────────
st.markdown("### 🔮 Comparativa de Escenarios de Competencia")
st.caption(f"Manteniendo descuento de **{descuento_adj:+d}%** | Variando precios de competencia")

sc_icons = {"Actual (0%)": "⚖️", "Competencia -5%": "📉", "Competencia +5%": "📈"}
sc_cols  = st.columns(3)

for idx, (sc_name, sc_df) in enumerate(all_scenarios.items()):
    tot_u = sc_df["unidades"].sum()
    tot_i = sc_df["ingresos"].sum()
    is_selected = sc_name == comp_scenario
    border_color = "#a78bfa" if is_selected else "#3d3d7a"
    badge = " ✅ Seleccionado" if is_selected else ""
    with sc_cols[idx]:
        st.markdown(f"""
        <div class="scenario-card" style="border-color:{border_color}; border-width: {'2px' if is_selected else '1px'};">
            <h3>{sc_icons[sc_name]} {sc_name}{badge}</h3>
            <div class="val">{tot_u:,.0f} u</div>
            <div class="sub">€ {tot_i:,.2f} ingresos</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)

# ─── SCENARIO COMPARISON CHART ────────────────────────────────────────────────────
st.markdown("##### Evolución diaria por escenario")
fig2, ax2 = plt.subplots(figsize=(14, 4))

palette = {"Actual (0%)": "#818cf8", "Competencia -5%": "#34d399", "Competencia +5%": "#f87171"}
for sc_name, sc_df in all_scenarios.items():
    lw = 3 if sc_name == comp_scenario else 1.5
    ax2.plot(sc_df["dia"], sc_df["unidades"], label=sc_name,
             color=palette[sc_name], linewidth=lw,
             linestyle="-" if sc_name == comp_scenario else "--",
             alpha=1 if sc_name == comp_scenario else 0.65)

ax2.axvline(x=28, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7)
ax2.set_xlabel("Día de Noviembre", fontsize=11)
ax2.set_ylabel("Unidades Proyectadas", fontsize=11)
ax2.set_title("Comparativa de Escenarios de Competencia", fontsize=13, fontweight="bold", color="#a78bfa")
ax2.set_xticks(results_df["dia"])
ax2.set_xticklabels(results_df["dia"], fontsize=8)
ax2.grid(axis="y", alpha=0.3)
ax2.legend(facecolor="#1a1a2e", edgecolor="#3d3d7a", labelcolor="#c0c0e0", fontsize=9)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("<hr class='section-sep'>", unsafe_allow_html=True)
st.caption("🤖 Predicciones generadas con HistGradientBoostingRegressor | Lags actualizados recursivamente día a día")
