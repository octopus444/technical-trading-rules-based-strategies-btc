"""
streamlit_app.py
Bitcoin Technical Trading Strategies — CV Showcase Dashboard
Place at: C:\\Projects\\TAstratBTC_ESE Reconstruction\\streamlit_app.py
Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Strategy Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom css ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #07090d;
    border-right: 1px solid #161c24;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background: #090c12; }

[data-testid="stMetric"] {
    background: #0e1420;
    border: 1px solid #182030;
    border-radius: 4px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a7888 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 26px !important;
    color: #d8eaf8 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

h1 {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: #b8d0e8 !important;
    letter-spacing: -0.03em;
    border-bottom: 1px solid #161c24;
    padding-bottom: 12px;
    margin-bottom: 4px !important;
}
h2 {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5a8aaa !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 28px !important;
}
h3 {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #4a6a80 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

[data-testid="stAlert"] {
    background: #0c1520 !important;
    border: 1px solid #1a2d3d !important;
    border-left: 3px solid #2a6090 !important;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #7a9ab0 !important;
}

[data-testid="stDataFrame"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

hr { border-color: #161c24 !important; margin: 20px 0 !important; }

[data-testid="stCaptionContainer"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #5a7888 !important;
    letter-spacing: 0.05em;
}

label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5a7888 !important;
}

.phase-card {
    background: #0c1420;
    border: 1px solid #16202c;
    border-top: 2px solid #1e3a54;
    border-radius: 4px;
    padding: 18px 20px;
    height: 100%;
}
.phase-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3a6080;
    margin-bottom: 8px;
}
.phase-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #7aaac8;
    margin-bottom: 10px;
    letter-spacing: -0.01em;
}
.phase-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    line-height: 1.7;
    color: #5a7888;
}
.result-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 2px;
    margin-top: 10px;
}
.tag-neg { background: #1a0f0f; color: #c05050; border: 1px solid #3a1818; }
.tag-pos { background: #0f1a12; color: #4a9a60; border: 1px solid #183a20; }
.tag-neu { background: #0f141a; color: #5a8aaa; border: 1px solid #182030; }
</style>
""", unsafe_allow_html=True)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

# ── contrarian availability ───────────────────────────────────────────────────
CONTRARIAN_AVAILABLE = {
    "Moving Average":       True,
    "Support & Resistance": True,
    "Channel Breakout":     True,
    "Bollinger Bands":      False,
    "RSI":                  False,
    "On-Balance Volume":    False,
    "Filter":               False,
}

# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_results():
    df = pd.read_csv(os.path.join(RESULTS, "all_results.csv"))
    df["class"] = df["strategy"].apply(_extract_class)
    return df

@st.cache_data
def load_profitable_after_tc():
    path = os.path.join(RESULTS, "profitable_after_tc.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["class"] = df["strategy"].apply(_extract_class)
    return df

@st.cache_data
def load_wfa():
    return pd.read_csv(os.path.join(RESULTS, "wfa_results.csv"))

@st.cache_data
def load_wrc():
    return pd.read_csv(os.path.join(RESULTS, "wrc_results.csv"))

@st.cache_data
def load_signal_combination():
    path = os.path.join(RESULTS, "signal_combination_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # normalize column names: handle both 'return_pct' and 'oos_return_pct' variants
    rename = {}
    for col in ("return_pct", "sharpe", "n_trades", "accuracy"):
        if col in df.columns and f"oos_{col}" not in df.columns:
            rename[col] = f"oos_{col}"
    if rename:
        df = df.rename(columns=rename)
    # rename 'period' → 'quarter' if needed
    if "period" in df.columns and "quarter" not in df.columns:
        df = df.rename(columns={"period": "quarter"})
    # derive oos_profitable
    if "oos_profitable" not in df.columns and "oos_return_pct" in df.columns:
        df["oos_profitable"] = df["oos_return_pct"] > 0
    return df

@st.cache_data
def load_wfa_equity():
    path = os.path.join(RESULTS, "wfa_equity.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)

def chain_equity(df):
    """Chain per-period normalized equity into a continuous curve."""
    periods = ["Period 1", "Period 2", "Period 3", "Period 4"]
    chunks, eq_mult, bh_mult = [], 1.0, 1.0
    for p in periods:
        sub = df[df["period"] == p].copy().sort_values("date")
        if len(sub) == 0:
            continue
        sub["equity_chain"]    = sub["equity"]    * eq_mult
        sub["bh_equity_chain"] = sub["bh_equity"] * bh_mult
        eq_mult = float(sub["equity"].iloc[-1])    * eq_mult
        bh_mult = float(sub["bh_equity"].iloc[-1]) * bh_mult
        chunks.append(sub)
    return pd.concat(chunks).reset_index(drop=True)

def _extract_class(name: str) -> str:
    if name.startswith(("MAc", "MA(")): return "Moving Average"
    if name.startswith("BB"):           return "Bollinger Bands"
    if name.startswith("SR"):           return "Support & Resistance"
    if name.startswith("CB"):           return "Channel Breakout"
    if name.startswith("RSI"):          return "RSI"
    if name.startswith("OBV"):          return "On-Balance Volume"
    if name.startswith(("F(", "Filt")): return "Filter"
    return "Other"

# ── design tokens ─────────────────────────────────────────────────────────────
C = dict(
    primary  = "#3a8abf",
    positive = "#3a9a60",
    negative = "#c04848",
    neutral  = "#5a7888",
    muted    = "#3a5060",
)
CLASS_COLORS = {
    "Moving Average":       "#3a8abf",
    "Bollinger Bands":      "#d07828",
    "Support & Resistance": "#3a9a60",
    "Channel Breakout":     "#b04040",
    "RSI":                  "#8a5ab8",
    "On-Balance Volume":    "#5a8898",
    "Filter":               "#8a8848",
}
PERIOD_COLORS = [
    "rgba(58,138,191,0.08)",
    "rgba(208,120,40,0.08)",
    "rgba(58,154,96,0.08)",
    "rgba(176,64,64,0.08)",
]

def _base_layout(fig, height=400, xtitle=None, ytitle=None):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="IBM Plex Mono, monospace", size=10, color="#5a7888"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=10), title=xtitle,
                   title_font=dict(size=10, color="#4a6070")),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=10), title=ytitle,
                   title_font=dict(size=10, color="#4a6070")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=24, b=10),
    )
    return fig

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:13px;"
        "color:#5a8aaa;padding:4px 0 12px;letter-spacing:-0.01em;'>₿ btc strategy research</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "nav",
        ["overview", "strategy explorer", "walk-forward analysis",
         "signal combination", "conclusions"],
        format_func=str.title,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#4a6878;
line-height:2.1;letter-spacing:0.04em;'>
DATASET<br>
bitstamp btc/usd · 5-min<br>
2017-01 → 2022-12<br>
630,720 observations<br><br>
UNIVERSE<br>
3,669 strategies<br>
7 strategy classes<br><br>
COSTS<br>
10 bps round-trip<br>
applied before WRC
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.title("Technical Trading Strategy Analysis")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#4a6878;"
        "margin-top:4px;margin-bottom:24px;'>3,669 rules · bitstamp btc/usd · 5-min · 2017–2022</p>",
        unsafe_allow_html=True,
    )

    wrc    = load_wrc().iloc[0]
    wfa_ov = load_wfa()
    sc_ov  = load_signal_combination()

    _p1_tag   = f'p-value = {float(wrc["wrc_p_value"]):.3f} · cannot reject H₀'
    _p2_prof  = int(wfa_ov["oos_profitable"].sum())
    _p2_total = len(wfa_ov)
    _p2_tag   = f'profitable in {_p2_prof}/{_p2_total} oos periods'
    if sc_ov is not None:
        _p3_prof   = int(sc_ov["oos_profitable"].sum())
        _p3_total  = len(sc_ov)
        _p3_sharpe = sc_ov["oos_sharpe"].mean()
        _p3_bw     = int(sc_ov["bandwidth_bps"].iloc[0]) if "bandwidth_bps" in sc_ov.columns else "?"
        _p3_tag    = f'{_p3_prof}/{_p3_total} quarters profitable · avg sharpe {_p3_sharpe:.3f}'
    else:
        _p3_bw  = "?"
        _p3_tag = 'data not available'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("strategies tested", "3,669")
    c2.metric("strategy classes",  "7")
    c3.metric("wrc p-value",       f"{float(wrc['wrc_p_value']):.3f}",
              help=f"White's Reality Check. H₀: no strategy has positive expected net return after TC. p = {float(wrc['wrc_p_value']):.3f} — cannot reject H₀ at 1%, 5%, or 10% significance.")
    c4.metric("bootstrap draws",   f"{int(wrc['wrc_B']):,}")

    st.markdown("---")
    st.subheader("methodology")
    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""
<div class="phase-card">
<div class="phase-num">phase 01</div>
<div class="phase-title">full-sample backtest</div>
<div class="phase-body">
All 3,669 strategies evaluated over 2017–2022. Ranked by gross Sharpe ratio.
White's Reality Check corrects for data snooping bias across the full universe
using 1,000 stationary bootstrap replications. Transaction costs applied before
statistical testing.
</div>
<div class="result-tag tag-neg">{_p1_tag}</div>
</div>""", unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
<div class="phase-card">
<div class="phase-num">phase 02</div>
<div class="phase-title">walk-forward analysis</div>
<div class="phase-body">
MAc(7,10,0.01), best after transaction costs, tested out-of-sample across
{_p2_total} expanding windows (2019–2022) with no parameter re-optimization.
Walk-forward optimization confirms d=0, c=0 as the best variant in all periods.
</div>
<div class="result-tag tag-pos">{_p2_tag}</div>
</div>""", unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
<div class="phase-card">
<div class="phase-num">phase 03</div>
<div class="phase-title">signal combination</div>
<div class="phase-body">
LightGBM aggregates signals from all 3,669 strategies using walk-forward
cross-validation. Bandwidth-adjusted targets ({_p3_bw} bps) to reduce label noise.
Final WRC test determines whether ML aggregation recovers alpha invisible
to individual rules.
</div>
<div class="result-tag tag-neu">{_p3_tag}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("strategy universe · counts by class")

    df = load_all_results()

    # bar chart
    class_order = [
        "Moving Average", "Support & Resistance", "Channel Breakout",
        "Bollinger Bands", "RSI", "On-Balance Volume", "Filter",
    ]
    class_counts = (
        df.groupby("class")["strategy"].count()
        .reindex(class_order).dropna()
        .sort_values(ascending=True).reset_index()
    )
    class_counts.columns = ["class", "n"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=class_counts["class"],
        x=class_counts["n"],
        orientation="h",
        marker_color=[CLASS_COLORS.get(c, C["neutral"]) for c in class_counts["class"]],
        marker_opacity=0.8,
        text=class_counts["n"].astype(str),
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
    ))
    fig = _base_layout(fig, height=260, xtitle="number of strategies")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # table: split standard / contrarian rows
    rows = []
    for cls in class_order:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        has_contrarian = CONTRARIAN_AVAILABLE.get(cls, False)
        std = sub[sub["contrarian"] == False]
        ctr = sub[sub["contrarian"] == True]
        rows.append({
            "strategy class": cls,
            "variant":        "standard",
            "n":              len(std),
            "median gross sharpe": round(float(std["sharpe_ratio"].median()), 3) if len(std) else None,
        })
        if has_contrarian and len(ctr) > 0:
            rows.append({
                "strategy class": cls,
                "variant":        "contrarian",
                "n":              len(ctr),
                "median gross sharpe": round(float(ctr["sharpe_ratio"].median()), 3),
            })

    tbl_df = pd.DataFrame(rows)
    _ov_center = ["variant", "n", "median gross sharpe"]

    st.dataframe(
        tbl_df.style.set_properties(subset=_ov_center, **{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "strategy class":      st.column_config.TextColumn("strategy class"),
            "variant":             st.column_config.TextColumn("variant",            width="small"),
            "n":                   st.column_config.NumberColumn("n",                width="small"),
            "median gross sharpe": st.column_config.NumberColumn("median gross sharpe",
                                                                  format="%.3f",
                                                                  width="medium"),
        },
    )
    st.caption(
        "gross sharpe · before transaction costs · annualization: 365 × 288 (24/7)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "strategy explorer":
    st.title("strategy explorer")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#4a6878;"
        "margin-top:4px;margin-bottom:24px;'>all 3,669 strategies · gross and net metrics · 2017–2022</p>",
        unsafe_allow_html=True,
    )

    df     = load_all_results()
    df_atc = load_profitable_after_tc()

    # ── section 1: gross metrics ───────────────────────────────────────────
    st.subheader("section 1 · gross metrics (before transaction costs)")

    col1, col2, col3 = st.columns(3)
    with col1:
        classes = ["all"] + sorted(df["class"].unique().tolist())
        sel_class = st.selectbox("strategy class", classes, key="gross_class")
    with col2:
        sort_col = st.selectbox("sort table by", [
            "sharpe_ratio", "total_return", "n_trades", "betc", "mean_excess_return_bps"
        ], format_func=lambda x: {
            "sharpe_ratio":           "sharpe ratio (gross)",
            "total_return":           "total return (gross)",
            "n_trades":               "number of trades",
            "betc":                   "breakeven tc (bps)",
            "mean_excess_return_bps": "mean excess return (bps)",
        }.get(x, x), key="gross_sort")
    with col3:
        direction = st.selectbox("variant", ["all", "standard", "contrarian"], key="gross_dir")

    filt = df.copy()
    if sel_class != "all":
        filt = filt[filt["class"] == sel_class]
    if direction == "standard":
        filt = filt[filt["contrarian"] == False]
    elif direction == "contrarian":
        filt = filt[filt["contrarian"] == True]
    filt = filt.sort_values(sort_col, ascending=False)

    has_both = (
        sel_class in ("all", "Moving Average", "Support & Resistance", "Channel Breakout")
        and direction == "all"
    )
    if has_both:
        st.info(
            "MA, SR, and CB each include a contrarian counterpart. "
            "For every standard strategy with gross sharpe X, its contrarian mirror has "
            "approximately −X. Distributions appear symmetric around 0 by construction. "
            "Use the **variant** filter to isolate one direction."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    gc1, gc2 = st.columns(2)
    with gc1:
        st.subheader("sharpe ratio distribution")
        fig = go.Figure()
        for cls in sorted(filt["class"].unique()):
            sub = filt[filt["class"] == cls]
            fig.add_trace(go.Histogram(
                x=sub["sharpe_ratio"], name=cls, opacity=0.7,
                marker_color=CLASS_COLORS.get(cls, C["neutral"]), nbinsx=60,
            ))
        fig = _base_layout(fig, height=300)
        fig.update_layout(barmode="overlay")
        fig.add_vline(x=0, line_color="white", line_width=1, opacity=0.2, line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("gross sharpe · before transaction costs")

    with gc2:
        st.subheader("sharpe vs trade frequency")
        fig2 = go.Figure()
        for cls in sorted(filt["class"].unique()):
            sub = filt[filt["class"] == cls]
            fig2.add_trace(go.Scatter(
                x=sub["n_trades"], y=sub["sharpe_ratio"],
                mode="markers", name=cls,
                marker=dict(color=CLASS_COLORS.get(cls, C["neutral"]), size=3.5, opacity=0.5),
            ))
        fig2 = _base_layout(fig2, height=300, xtitle="number of trades", ytitle="sharpe ratio")
        fig2.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.15, line_dash="dot")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("gross sharpe and trade frequency · before transaction costs")

    st.subheader("breakeven transaction cost")
    betc_filt  = filt[(filt["n_trades"] >= 10) & (filt["betc"] <= 200)].copy()
    n_excluded = len(filt) - len(betc_filt)

    fig3 = go.Figure()
    for cls in sorted(betc_filt["class"].unique()):
        sub = betc_filt[betc_filt["class"] == cls]
        fig3.add_trace(go.Scatter(
            x=sub["betc"], y=sub["total_return"],
            mode="markers", name=cls,
            marker=dict(color=CLASS_COLORS.get(cls, C["neutral"]), size=3.5, opacity=0.5),
            hovertemplate="<b>%{text}</b><br>betc: %{x:.1f} bps<br>gross return: %{y:.2f}x<extra></extra>",
            text=sub["strategy"],
        ))
    fig3 = _base_layout(fig3, height=300,
                        xtitle="breakeven tc (bps)", ytitle="total return (gross)")
    fig3.add_vline(x=10, line_color=C["negative"], line_width=1.5, line_dash="dot", opacity=0.7,
                   annotation_text="actual cost: 10 bps",
                   annotation_font=dict(size=10, color=C["negative"]),
                   annotation_position="top right")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"strategies right of the line survive 10 bps round-trip costs · "
        f"{n_excluded} excluded: fewer than 10 trades or betc > 200 bps"
    )

    st.subheader("data table · top 200")
    show = filt[[
        "strategy", "class", "contrarian", "sharpe_ratio", "total_return",
        "n_trades", "betc", "mean_excess_return_bps",
    ]].head(200).copy()
    show["sharpe_ratio"]           = show["sharpe_ratio"].round(3)
    show["total_return"]           = show["total_return"].round(3)
    show["betc"]                   = show["betc"].round(2)
    show["mean_excess_return_bps"] = show["mean_excess_return_bps"].round(5)
    show.columns = [
        "strategy", "class", "contrarian", "sharpe (gross)", "total return (gross)",
        "n trades", "betc (bps)", "mean excess return (bps)",
    ]
    _exp_center = ["class", "contrarian", "sharpe (gross)", "total return (gross)", "n trades", "betc (bps)", "mean excess return (bps)"]
    st.dataframe(
        show.style.set_properties(subset=_exp_center, **{"text-align": "center"}),
        use_container_width=True, height=360, hide_index=True)

    # ── section 2: after TC ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("section 2 · strategies surviving transaction costs")

    if df_atc is None:
        st.warning("profitable_after_tc.csv not found in results/")
    else:
        st.markdown(
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#4a6878;'>"
            f"{len(df_atc)} strategies with betc > 10 bps · net metrics shown</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            st.subheader("survivors by class")
            surv = (
                df_atc.groupby("class")["strategy"].count()
                .reset_index().sort_values("strategy", ascending=True)
            )
            surv.columns = ["class", "n"]
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                y=surv["class"], x=surv["n"],
                orientation="h",
                marker_color=[CLASS_COLORS.get(c, C["neutral"]) for c in surv["class"]],
                marker_opacity=0.8,
                text=surv["n"].astype(str),
                textposition="outside",
                textfont=dict(size=10, family="IBM Plex Mono"),
            ))
            fig_s = _base_layout(fig_s, height=260, xtitle="strategies with betc > 10 bps")
            fig_s.update_layout(showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

        with ac2:
            st.subheader("net excess return distribution")
            if "net_excess_return_bps" in df_atc.columns:
                fig_n = go.Figure()
                for cls in sorted(df_atc["class"].unique()):
                    sub = df_atc[df_atc["class"] == cls]
                    fig_n.add_trace(go.Histogram(
                        x=sub["net_excess_return_bps"], name=cls, opacity=0.7,
                        marker_color=CLASS_COLORS.get(cls, C["neutral"]), nbinsx=40,
                    ))
                fig_n = _base_layout(fig_n, height=260, xtitle="net excess return (bps)")
                fig_n.update_layout(barmode="overlay")
                fig_n.add_vline(x=0, line_color="white", line_width=1, opacity=0.2, line_dash="dot")
                st.plotly_chart(fig_n, use_container_width=True)
            else:
                st.caption("net_excess_return_bps column not available in profitable_after_tc.csv")

        st.subheader("net excess return vs trade frequency")
        _wrc_row   = load_wrc().iloc[0]
        _wrc_n_obs = int(_wrc_row["n_observations"])
        _tc_dec    = float(_wrc_row["transaction_cost_bps"]) / 10000
        _net_df    = filt.copy()
        _net_df["net_exc_bps"] = (
            _net_df["mean_excess_return_bps"]
            - (_net_df["n_trades"] * 2 * _tc_dec / _wrc_n_obs) * 10000
        )
        fig_net = go.Figure()
        for cls in sorted(_net_df["class"].unique()):
            _sub = _net_df[_net_df["class"] == cls]
            fig_net.add_trace(go.Scatter(
                x=_sub["n_trades"], y=_sub["net_exc_bps"],
                mode="markers", name=cls,
                marker=dict(color=CLASS_COLORS.get(cls, C["neutral"]), size=3.5, opacity=0.5),
            ))
        fig_net = _base_layout(fig_net, height=300,
                               xtitle="number of trades", ytitle="net excess return (bps/bar)")
        fig_net.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.15, line_dash="dot")
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(
            "mean excess return per bar after subtracting amortised round-trip tc · "
            "high-trade strategies are pushed below zero by cumulative tc drag"
        )

        st.subheader("surviving strategies · full table")
        atc_show = df_atc[[
            "strategy", "class", "contrarian",
            "sharpe_ratio", "total_return", "n_trades", "betc",
        ] + (["net_excess_return_bps"] if "net_excess_return_bps" in df_atc.columns else [])
        ].copy()
        atc_show["sharpe_ratio"] = atc_show["sharpe_ratio"].round(3)
        atc_show["total_return"] = atc_show["total_return"].round(3)
        atc_show["betc"]         = atc_show["betc"].round(2)
        if "net_excess_return_bps" in atc_show.columns:
            atc_show["net_excess_return_bps"] = atc_show["net_excess_return_bps"].round(5)
        atc_show = atc_show.sort_values("betc", ascending=True)
        _atc_center = [c for c in atc_show.columns if c != "strategy"]
        st.dataframe(
            atc_show.style.set_properties(subset=_atc_center, **{"text-align": "center"}),
            use_container_width=True, height=400, hide_index=True)
        st.caption(
            "betc = gross mean return per trade in bps — strategies above 10 bps recover the round-trip cost · sorted ascending by betc"
        )





# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "walk-forward analysis":
    st.title("walk-forward analysis")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#4a6878;"
        "margin-top:4px;margin-bottom:24px;'>"
        "MAc(7,10,0.01) · fixed parameters · expanding windows · oos: 2019–2022</p>",
        unsafe_allow_html=True,
    )

    wfa    = load_wfa()
    eq_raw = load_wfa_equity()

    st.info(
        "Walk-forward analysis tests MAc(7,10,0.01), selected on the full 2017–2022 sample, "
        "on successive out-of-sample years with no parameter re-optimization. "
        "The strategy was identified using all available data — this is acknowledged data mining, "
        "but the exercise tests whether the identified edge holds when applied forward in time."
    )

    # ── period cards ───────────────────────────────────────────────────────
    st.subheader("oos period results")
    cols = st.columns(len(wfa))
    for i, (_, row) in enumerate(wfa.iterrows()):
        oos_r  = float(row["oos_total_return_pct"])
        sharpe = float(row["oos_sharpe"])
        with cols[i]:
            st.metric(
                row["period"],
                f"SR {sharpe:.2f}",
                delta=f"{oos_r:+.1f}% return",
                delta_color="normal" if oos_r >= 0 else "inverse",
            )
            st.caption(f"test: {row['test']}\ntrades: {int(row['oos_n_trades'])}")

    st.markdown("---")

    if eq_raw is not None:
        eq       = chain_equity(eq_raw)
        eq_daily = (
            eq.set_index("date")[["equity_chain", "bh_equity_chain"]]
            .resample("D").last().dropna()
        )

        st.subheader("equity curve · oos 2019–2022")
        fig = go.Figure()
        for i, (_, row) in enumerate(wfa.iterrows()):
            parts = [p.strip() for p in row["test"].split(" to ")]
            t0 = pd.to_datetime(parts[0])
            t1 = pd.to_datetime(parts[1]) + pd.offsets.MonthEnd(0)
            fig.add_vrect(x0=t0, x1=t1, fillcolor=PERIOD_COLORS[i % 4],
                          layer="below", line_width=0,
                          annotation_text=row["period"],
                          annotation_font=dict(size=9, color="#5a8aaa"),
                          annotation_position="top left")

        fig.add_trace(go.Scatter(
            x=eq_daily.index, y=eq_daily["equity_chain"],
            name="MAc(7,10,0.01)",
            line=dict(color=C["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=eq_daily.index, y=eq_daily["bh_equity_chain"],
            name="btc buy & hold",
            line=dict(color="#8abacc", width=1.5, dash="dot"),
            opacity=0.8,
        ))
        fig = _base_layout(fig, height=380, ytitle="cumulative return (normalized to 1)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "buy & hold shown for context only · btc appreciated ~350% over 2019–2022 "
            "during an exceptional bull cycle · not a meaningful performance benchmark"
        )

        st.subheader("drawdown from peak")
        cum  = eq_daily["equity_chain"]
        peak = cum.expanding().max()
        dd   = (cum / peak - 1) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy", fillcolor="rgba(192,72,72,0.15)",
            line=dict(color=C["negative"], width=1),
        ))
        fig_dd = _base_layout(fig_dd, height=200, ytitle="drawdown (%)")
        fig_dd.update_layout(showlegend=False)
        fig_dd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)
        st.caption(
            "measured from the strategy's own cumulative all-time high · "
            "a period with positive sharpe can still show large drawdown "
            "if the strategy hasn't recovered a prior peak"
        )

        st.markdown("---")

        st.subheader("quarterly returns")
        eq_q = eq_daily["equity_chain"].resample("QE").agg(["first", "last"])
        eq_q["return_pct"] = (eq_q["last"] / eq_q["first"] - 1) * 100
        eq_q["quarter"]    = eq_q.index.to_period("Q").astype(str)

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=eq_q["quarter"], y=eq_q["return_pct"],
            marker_color=[C["positive"] if r >= 0 else C["negative"]
                          for r in eq_q["return_pct"]],
            marker_opacity=0.8,
            text=[f"{r:.1f}%" for r in eq_q["return_pct"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono"),
        ))
        fig_q = _base_layout(fig_q, height=280, ytitle="return (%)")
        fig_q.update_layout(showlegend=False)
        fig_q.add_hline(y=0, line_color="white", line_width=0.6, opacity=0.15)
        st.plotly_chart(fig_q, use_container_width=True)

    else:
        st.warning(
            "wfa_equity.csv not found in results/ · "
            "run: python walk_forward.py .bitstampUSD.csv --skip-full-wfo"
        )

    st.markdown("---")
    st.subheader("in-sample vs out-of-sample sharpe")
    tbl = wfa[["period", "train", "test", "is_sharpe", "oos_sharpe",
               "oos_total_return_pct", "oos_n_trades", "oos_profitable"]].copy()
    tbl["Δ sharpe %"] = (
        (tbl["oos_sharpe"] - tbl["is_sharpe"]) / tbl["is_sharpe"].abs() * 100
    ).round(1).astype(str) + "%"
    tbl["is_sharpe"]            = tbl["is_sharpe"].round(3)
    tbl["oos_sharpe"]           = tbl["oos_sharpe"].round(3)
    tbl["oos_total_return_pct"] = tbl["oos_total_return_pct"].round(1)
    tbl.columns = ["period", "train", "test", "is sharpe", "oos sharpe",
                   "oos return %", "oos trades", "profitable", "Δ sharpe %"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMBINATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "signal combination":
    st.title("signal combination · lightgbm")

    sc = load_signal_combination()

    if sc is None:
        st.warning("signal_combination_results.csv not found in results/")
        st.stop()

    bw_bps    = int(sc["bandwidth_bps"].iloc[0]) if "bandwidth_bps" in sc.columns else "?"
    tc_val    = int(sc["tc_bps"].iloc[0])         if "tc_bps"        in sc.columns else "?"
    n_periods = len(sc)

    st.markdown(
        f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#4a6878;"
        f"margin-top:4px;margin-bottom:24px;'>"
        f"walk-forward · {n_periods} quarters · bandwidth {bw_bps} bps · tc {tc_val} bps</p>",
        unsafe_allow_html=True,
    )

    n_prof  = int(sc["oos_profitable"].sum())
    avg_acc = sc["oos_accuracy"].mean() * 100 if "oos_accuracy" in sc.columns else None
    tot_ret = ((sc["oos_return_pct"] / 100 + 1).prod() - 1) * 100

    st.info(
        f"LightGBM is trained on raw signals from all 3,669 strategies to predict next-bar "
        f"price direction. Features are sparse position signals (\u22121, 0, +1), where zero "
        f"means the rule is not actively signalling, this represents the distinction between "
        f"a rule firing and a rule simply holding. The target applies {bw_bps} bps bandwidth: "
        f"a bar is labelled long or short only if the subsequent move exceeds {bw_bps} bps, "
        f"otherwise hold. This concentrates the classification problem on economically "
        f"meaningful moves and reduces label noise at 5-minute frequency. Model predictions "
        f"are forward-filled, so the portfolio holds its last directional position until a "
        f"new signal is generated. The model is retrained each quarter on all preceding data."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("profitable quarters", f"{n_prof} / {len(sc)}")
    c2.metric("avg accuracy",        f"{avg_acc:.1f}%" if avg_acc is not None else "n/a",
              help=f"Accuracy is inflated when the hold class dominates — predicting hold "
                   f"correctly generates no trades. Avg: {avg_acc:.1f}%." if avg_acc is not None
                   else "accuracy not available")
    c3.metric("compounded return",   f"{tot_ret:.1f}%")

    st.markdown("---")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("quarterly return")
        fig_sc1 = go.Figure()
        fig_sc1.add_trace(go.Bar(
            x=sc["quarter"], y=sc["oos_return_pct"],
            marker_color=[C["positive"] if r >= 0 else C["negative"]
                          for r in sc["oos_return_pct"]],
            marker_opacity=0.8,
            text=[f"{r:.1f}%" for r in sc["oos_return_pct"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono"),
        ))
        fig_sc1 = _base_layout(fig_sc1, height=320, ytitle="return (%)")
        fig_sc1.update_layout(showlegend=False)
        fig_sc1.add_hline(y=0, line_color="white", line_width=0.6, opacity=0.15)
        st.plotly_chart(fig_sc1, use_container_width=True)

    with cc2:
        st.subheader("trade count per quarter")
        fig_sc2 = go.Figure()
        fig_sc2.add_trace(go.Bar(
            x=sc["quarter"], y=sc["oos_n_trades"],
            marker_color=C["primary"], marker_opacity=0.6,
            text=sc["oos_n_trades"].astype(str),
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono"),
        ))
        fig_sc2 = _base_layout(fig_sc2, height=320, ytitle="trades")
        fig_sc2.update_layout(showlegend=False)
        st.plotly_chart(fig_sc2, use_container_width=True)

    _tc_cost_str = f"{int(tc_val) / 10:.0f}%" if tc_val != "?" else "?"
    _worst = sc.nlargest(2, "oos_n_trades")[["quarter", "oos_n_trades"]]
    _worst_str = " · ".join(
        f"{row['quarter']}: {int(row['oos_n_trades'])} trades" for _, row in _worst.iterrows()
    )
    st.caption(
        f"at {tc_val} bps round-trip, 100 trades = {_tc_cost_str} cost regardless of signal quality · "
        f"highest-trade quarters: {_worst_str}"
    )

    st.markdown("---")
    st.subheader("cumulative equity · oos 2019–2022")

    sc_eq = sc[["quarter", "oos_return_pct"]].copy()
    sc_eq["equity"] = (1 + sc_eq["oos_return_pct"] / 100).cumprod()
    import pandas as _pd_sc
    _start = _pd_sc.DataFrame([{"quarter": "start", "oos_return_pct": 0.0, "equity": 1.0}])
    sc_eq  = _pd_sc.concat([_start, sc_eq], ignore_index=True)

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=sc_eq["quarter"], y=sc_eq["equity"],
        mode="lines+markers",
        line=dict(color=C["primary"], width=2),
        marker=dict(size=5),
        name="LightGBM",
    ))
    fig_eq.add_hline(y=1.0, line_color="white", line_width=0.6, opacity=0.15)
    fig_eq = _base_layout(fig_eq, height=280, ytitle="cumulative return (normalized to 1)")
    fig_eq.update_layout(showlegend=False)
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("drawdown from peak")
    _peak = sc_eq["equity"].expanding().max()
    _dd   = (sc_eq["equity"] / _peak - 1) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=sc_eq["quarter"], y=_dd,
        fill="tozeroy", fillcolor="rgba(192,72,72,0.15)",
        line=dict(color=C["negative"], width=1),
    ))
    fig_dd = _base_layout(fig_dd, height=180, ytitle="drawdown (%)")
    fig_dd.update_layout(showlegend=False)
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True)
    st.caption("quarterly resolution — equity compounded across 16 oos quarters")

    st.markdown("---")
    st.subheader("interpretation")
    _acc_str   = f"{avg_acc:.0f}%" if avg_acc is not None else "high avg accuracy"
    _avg_trades = int(sc["oos_n_trades"].mean()) if "oos_n_trades" in sc.columns else "~100"
    st.markdown(
        "The pattern is consistent: quarters with few predicted directional moves are "
        "profitable; quarters with many trades are not. The root cause is structural. "
        "LightGBM minimises classification log-loss with no awareness of transaction costs. "
        f"At 5-minute frequency the true signal-to-noise ratio is extremely low, WRC already "
        f"established that no individual rule generates statistically significant alpha. "
        f"Aggregating 3,669 weak signals does not manufacture alpha from noise; the model "
        f"learns to classify slightly better than random, but fires long/short whenever any "
        f"directional class probability marginally exceeds the threshold. At the default "
        f"threshold of 0.33, this produces ~{_avg_trades} trades per quarter on average. "
        f"At {tc_val} bps round-trip, high trade frequency is lethal regardless of per-trade "
        f"accuracy. The {_acc_str} average accuracy is largely explained by hold-class "
        f"dominance — predicting hold correctly generates no trades and no costs."
    )
    st.markdown(
        "For this strategy universe and model specification, 5-minute frequency is "
        f"incompatible with {tc_val} bps round-trip costs — the signal quality available "
        "from classical technical rules is insufficient to generate enough per-trade alpha "
        "to overcome transaction costs at this frequency. A natural extension would be to "
        "implement a cost-aware loss function that directly penalises position changes during "
        "training, so the model internalises transaction costs rather than ignoring them."
    )

    st.markdown("---")
    st.subheader("full period table")
    disp_cols = ["quarter", "oos_return_pct", "oos_n_trades", "oos_profitable"]
    rename    = {"quarter": "quarter", "oos_return_pct": "return %",
                 "oos_n_trades": "n trades", "oos_profitable": "profitable"}
    if "oos_accuracy" in sc.columns:
        disp_cols.append("oos_accuracy")
        rename["oos_accuracy"] = "accuracy"
    disp = sc[disp_cols].copy()
    disp["oos_return_pct"] = disp["oos_return_pct"].round(1)
    if "oos_accuracy" in disp.columns:
        disp["oos_accuracy"] = (disp["oos_accuracy"] * 100).round(1).astype(str) + "%"
    disp.rename(columns=rename, inplace=True)
    _sc_center = [c for c in disp.columns if c != "quarter"]
    st.dataframe(
        disp.style.set_properties(subset=_sc_center, **{"text-align": "center"}),
        use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "conclusions":
    st.title("conclusions")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#4a6878;"
        "margin-top:4px;margin-bottom:24px;'>what the analysis found and what it means</p>",
        unsafe_allow_html=True,
    )

    wrc = load_wrc().iloc[0]
    wfa = load_wfa()
    sc  = load_signal_combination()

    st.subheader("statistical result · white's reality check")
    st.info(
        f"p-value = {float(wrc['wrc_p_value']):.3f}  ·  "
        f"{int(wrc['n_strategies']):,} strategies  ·  "
        f"{int(wrc['wrc_B']):,} bootstrap replications  ·  "
        f"tc = {int(wrc['transaction_cost_bps'])} bps applied before testing\n\n"
        "Cannot reject the null hypothesis of no outperformance at 1%, 5%, or 10%. "
        "The same conclusion holds after LightGBM signal combination."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("findings")
        _sc_prof_n = int(sc["oos_profitable"].sum()) if sc is not None else "?"
        _sc_tot_n  = len(sc) if sc is not None else "?"
        st.markdown(
            "No individual technical trading rule based strategy produces statistically "
            "significant alpha on 5-minute BTC/USD after correcting for data snooping bias "
            f"across {int(wrc['n_strategies']):,} rules. Transaction costs are the dominant "
            "factor separating gross and net performance. "
            f"MAc(7,10,0.01) is profitable in all 4 out-of-sample periods with an average "
            f"Sharpe of {wfa['oos_sharpe'].mean():.2f} — an encouraging practical result, "
            "but one that does not constitute statistical evidence of alpha under White's "
            "Reality Check. As a further test, LightGBM was used to aggregate signals from "
            f"all {int(wrc['n_strategies']):,} strategies in a walk-forward framework; "
            f"signal combination achieved {_sc_prof_n}/{_sc_tot_n} profitable quarters, "
            "with the model's tendency to overtrade at 5-minute frequency limiting net "
            f"performance under {int(wrc['transaction_cost_bps'])} bps round-trip costs."
        )

    with c2:
        st.subheader("methodology notes")
        st.markdown(
            "White's Reality Check is necessary when selecting from a large strategy universe, "
            "as standard t-tests do not account for the multiple comparison penalty. "
            "WRC is applied to net returns after transaction costs, which is the economically "
            "relevant performance measure. Walk-forward analysis uses expanding windows rather "
            "than rolling, which produces more stable out-of-sample results — expanding windows "
            "preserve all available history and avoid reweighting the training set arbitrarily "
            "as time progresses."
        )

    st.markdown("---")
    n_profitable   = int(wfa["oos_profitable"].sum())
    avg_oos_sharpe = wfa["oos_sharpe"].mean()
    sc_profitable  = int(sc["oos_profitable"].sum()) if sc is not None else None

    co1, co2, co3, co4 = st.columns(4)
    co1.metric("wrc p-value",            f"{float(wrc['wrc_p_value']):.3f}")
    co2.metric("wfa profitable periods", f"{n_profitable} / {len(wfa)}")
    co3.metric("wfa avg oos sharpe",     f"{avg_oos_sharpe:.3f}")
    co4.metric("ml profitable quarters",
               f"{sc_profitable} / {len(sc)}" if sc_profitable is not None else "n/a")

    st.markdown("---")
    st.caption(
        "data: bitstamp btc/usd · 5-min bars · 2017–2022 · 630,720 observations · "
        "7 strategy classes · 3,669 strategies · tc = 10 bps round-trip"
    )
