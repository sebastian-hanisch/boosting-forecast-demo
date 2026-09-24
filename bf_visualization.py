"""Plotly-Abbildungen der Boosting-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bf_constants as C
import bf_features as F
import bf_gbm as G

COLORS = {"gbm": "#2e7d32", "hw_mult": "#e6550d", "regression": "#00897b", "snaive_k": "#17becf", "oracle": "#54a24b"}
SHORT = {"gbm": "Globales Boosting", "hw_mult": "Holt-Winters", "regression": "Regression", "snaive_k": "Wochenmittel", "oracle": "Orakel"}
ACTUAL = "#14233B"
WARN = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_series(a, dep, origin):
    """Die ganze Reihe eines Depots (drei Jahre) mit Erwartungswert, Testjahr und Ursprung."""
    port = a.port
    t = np.arange(port.y.shape[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=port.y[dep], mode="lines", name="Tagesaufträge", line=dict(color=ACTUAL, width=1)))
    fig.add_trace(go.Scatter(x=t, y=port.mu[dep], mode="lines", name="Erwartungswert (ohne Rauschen)", line=dict(color=COLORS["oracle"], width=1.5, dash="dot")))
    fig.add_vrect(x0=C.FIRST_TEST, x1=port.y.shape[1], fillcolor="rgba(245,133,24,0.08)", line_width=0, annotation_text="Testjahr (Ursprünge)", annotation_position="top left")
    fig.add_vline(x=origin, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 280)


def build_origin(a, dep, origin):
    """42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Erwartungswert und die Prognosen der vier Verfahren."""
    port, h = a.port, a.settings.horizon
    i = int(origin - a.origins[0])
    x_hist, x_fut = np.arange(origin - 42, origin), np.arange(origin, origin + h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=port.y[dep, origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=port.y[dep, origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=2), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=port.mu[dep, origin:origin + h], mode="lines", name="Erwartungswert", line=dict(color=COLORS["oracle"], width=2, dash="dot")))
    for m in ("gbm", "hw_mult", "regression", "snaive_k"):
        fig.add_trace(go.Scatter(x=x_fut, y=a.forecasts[m][dep, i], mode="lines", name=C.METHOD_NAMES[m].split(" (")[0], line=dict(color=COLORS[m], width=3 if m == "gbm" else 1.4, dash="solid" if m == "gbm" else "dash")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.35))


def build_importance(a, top=12):
    imp = G.importances(a.model)
    order = np.argsort(imp)[::-1][:top][::-1]
    fig = go.Figure(go.Bar(x=[100 * imp[k] for k in order], y=[a.names[k] for k in order], orientation="h", marker=dict(color=COLORS["gbm"]), text=[f"{100 * imp[k]:.1f} %".replace(".", ",") for k in order], textposition="outside", showlegend=False))
    fig.update_xaxes(title_text="Anteil am Gain der Splits (%)", rangemode="tozero")
    return _base(fig, 340)


def build_curve(a):
    xs = np.arange(1, len(a.curve_val) + 1)
    fig = go.Figure()
    if len(a.curve_train):
        fig.add_trace(go.Scatter(x=xs, y=a.curve_train, mode="lines", name="Training", line=dict(color=COLORS["gbm"], width=2)))
    fig.add_trace(go.Scatter(x=xs, y=a.curve_val, mode="lines", name="Validierung (letzte 60 Trainingstage)", line=dict(color=WARN, width=2)))
    best = int(np.argmin(a.curve_val)) + 1
    fig.add_vline(x=best, line=dict(color="#7f7f7f", dash="dot"), annotation_text=f"bester Wert: Runde {best}", annotation_position="top right")
    fig.update_xaxes(title_text="Boosting-Runden (Bäume)")
    fig.update_yaxes(title_text="mittlerer absoluter Fehler (Log-Verhältnis)", rangemode="tozero")
    return _base(fig, 340)


def build_bars(a):
    ms = sorted(a.summary, key=lambda m: a.summary[m]["mase"])
    vals = [a.summary[m]["mase"] for m in ms]
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in ms], y=vals, marker=dict(color=[COLORS[m] for m in ms]), text=[de(v) for v in vals], textposition="outside", showlegend=False))
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="MASE 1 = saisonal naiv im Training", annotation_position="bottom right")
    fig.update_yaxes(title_text="MASE, Mittel über die Depots (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 360)


def build_horizon(a):
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in a.horizon_mae:
        fig.add_trace(go.Scatter(x=xs, y=a.horizon_mae[m], mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5 if m == "gbm" else 1.4), marker=dict(size=4)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 320).update_layout(legend=dict(orientation="h", y=-0.35))


def build_scatter(a):
    """Je Depot ein Punkt: MASE des Boostings gegen die von Holt-Winters; unter der Diagonalen gewinnt das Boosting."""
    x, y = a.depot_mase["hw_mult"], a.depot_mase["gbm"]
    hi = float(max(x.max(), y.max())) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="#7f7f7f", dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=COLORS["gbm"], size=8, opacity=0.75), text=[f"Depot {i}" for i in range(len(x))], showlegend=False))
    fig.update_xaxes(title_text="MASE Holt-Winters (je Depot)", range=[0, hi])
    fig.update_yaxes(title_text="MASE Boosting", range=[0, hi], scaleanchor="x", constrain="domain")
    return _base(fig, 340)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_features(res):
    rows = res["rows"]
    labels = [r["name"] for r in rows]
    vals = [r["mase"] for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=vals, error_y=dict(type="data", array=[r["se"] for r in rows]), marker=dict(color=COLORS["gbm"]), text=[de(v) for v in vals], textposition="outside", showlegend=False))
    for m in ("hw_mult", "regression", "oracle"):
        fig.add_hline(y=res["refs"][m], line=dict(color=COLORS[m], dash="dash" if m != "oracle" else "dot"), annotation_text=SHORT[m], annotation_position="top left" if m == "hw_mult" else "bottom left" if m == "regression" else "bottom right")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380)


def build_series_counts(res):
    xs = [str(n) for n in res["counts"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[res["rows"][n][0] for n in res["counts"]], error_y=dict(type="data", array=[res["rows"][n][1] for n in res["counts"]]), mode="lines+markers", name="Globales Boosting", line=dict(color=COLORS["gbm"], width=3)))
    for m in ("hw_mult", "regression", "snaive_k"):
        fig.add_hline(y=res["refs"][m], line=dict(color=COLORS[m], dash="dash"), annotation_text=SHORT[m] + " (je Depot, volle Historie)", annotation_position="top right" if m != "regression" else "bottom right")
    fig.update_xaxes(title_text="Zahl der Depots, aus denen das Modell lernt", type="category")
    fig.update_yaxes(title_text="MASE auf getrennten Test-Depots", rangemode="tozero")
    return _base(fig, 380)


def build_history(res):
    hs = list(res["histories"])
    xs = [str(h) for h in hs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[res["gbm"][0]] * len(hs), mode="lines+markers", name="Globales Boosting (aus anderen Depots)", line=dict(color=COLORS["gbm"], width=3)))
    for m in ("hw_mult", "regression"):
        fig.add_trace(go.Scatter(x=xs, y=[res["local"][(m, h)][0] for h in hs], error_y=dict(type="data", array=[res["local"][(m, h)][1] for h in hs]), mode="lines+markers", name=SHORT[m] + " (je Depot)", line=dict(color=COLORS[m], width=2)))
    fig.add_hline(y=res["snaive_k"], line=dict(color=COLORS["snaive_k"], dash="dash"), annotation_text="Wochenmittel", annotation_position="top right")
    fig.update_xaxes(title_text="Historie des neuen Depots vor dem Testjahr (Tage)", type="category")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.35))


def build_trend(rows):
    xs = [f"{r['trend']} %" for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("gbm_level", "Boosting, Ziel: Verhältnis zum Niveau", COLORS["gbm"], "solid"), ("gbm_none", "Boosting, Ziel: roher Wert", "#a5d6a7", "solid"), ("hw_mult", "Holt-Winters", COLORS["hw_mult"], "dash"), ("regression", "Regression", COLORS["regression"], "dash")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key][0] for r in rows], error_y=dict(type="data", array=[r[key][1] for r in rows]), mode="lines+markers", name=name, line=dict(color=color, width=2.5, dash=dash)))
    fig.update_xaxes(title_text="mittlerer Trend der Depots (Prozent je Jahr)", type="category")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.35))
