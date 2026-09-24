"""Boosting mit Lag-Merkmalen - ein Modell für alle Depots - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: das erste **globale** Modell und die erste **überwachte Lern**-Aufgabe der Linie. Ein Gradient-Boosting-Modell lernt aus Lag- und Kalendermerkmalen aller Depots gemeinsam
und prognostiziert jedes einzelne; die Vorgänger schätzten je Depot ein eigenes Modell.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import bf_constants as C
import bf_features as F
from bf_evaluation import Settings, analyse, features_experiment, history_experiment, series_experiment, trend_experiment
from bf_presets import ALL_GROUPS, PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from bf_visualization import SHORT, build_bars, build_curve, build_features, build_history, build_horizon, build_importance, build_origin, build_scatter, build_series, build_series_counts, build_trend

st.set_page_config(page_title="Boosting für Prognosen – Sebastian Hanisch", layout="wide")

GROUP_LABELS = {"lags": "Lag-Merkmale (letzte Werte)", "calendar": "Kalender (Wochentag, Jahrestag, Feiertag, Horizont)", "promo": "Aktionsplan (Aktion am Zieltag)"}


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _features(seeds):
    return features_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _series(counts, seeds):
    return series_experiment(counts=counts, seeds=seeds)


@st.cache_data(show_spinner=False)
def _history(histories, seeds):
    return history_experiment(histories=histories, seeds=seeds)


@st.cache_data(show_spinner=False)
def _trend(trends, seeds):
    return trend_experiment(trends=trends, seeds=seeds)


st.title("🌲 Boosting mit Lag-Merkmalen – ein Modell für alle Depots")
st.markdown(
    """
Bisher bekam **jedes Depot sein eigenes Modell**: Wochenmittel, Glättung, ARIMA, Regression - je auf der eigenen Historie geschätzt. Ein **globales Modell** dreht das um: **ein** Gradient-Boosting-Modell lernt aus den Lags, dem Kalender und dem Aktionsplan **aller Depots gemeinsam** und prognostiziert jedes einzelne. Die Demo
zeigt auf einem **Portfolio von Depots** (dieselben Tagesaufträge wie in den Vorgängern, jetzt mit eigenen Parametern je Depot), was das bringt - und wann nicht: welche Merkmale zählen, **wie viele Depots** das Modell braucht, wie es einem **neuen Depot mit kurzer Historie** hilft und wo Bäume an **Trends**
scheitern. Die Baumbibliothek ist selbst geschrieben (numpy) und im Test gegen `sklearn` geprüft.
"""
)
st.caption(
    "Sechstes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe: das **erste globale Modell** und die **erste überwachte Lernaufgabe** der Linie; alle Daten sind erzeugt. "
    "**Bezug zu OR:** wer hunderte Depots, Filialen oder Artikel prognostiziert, kann nicht hunderte Modelle pflegen - und braucht eine Prognose für Neuzugänge ohne Historie."
)

with st.expander("So funktioniert das globale Modell", expanded=True):
    st.markdown(
        """
1. **Zeilen.** Eine Zeile ist ein Tripel (Depot, Ursprung $t$, Horizont $j$): bekannt sind die Tage vor $t$, gesucht ist der Wert am Zieltag $t + j - 1$. So entsteht aus allen Depots eine große Tabelle - ein **direktes** Mehrschritt-Modell: der Horizont ist ein Merkmal.
2. **Merkmale.** **Lags:** die letzten sieben Tage, die letzten vier Werte des Zielwochentags, das 7-Tage-Mittel im Verhältnis zum 28-Tage-Niveau, dessen Verhältnis zum 91-Tage-Niveau, die Streuung. **Kalender:** Wochentag, Jahrestag (sin/cos), Feiertag, Tag danach, Horizont. **Aktionsplan:** Aktion am Zieltag.
3. **Ziel.** Das Verhältnis zum jüngsten Niveau im Log, $\\log\\big((y + 1) / (\\text{Niveau} + 1)\\big)$: so sind Depots verschiedener Größe vergleichbar, und die Bäume müssen keine Trends extrapolieren.
4. **Boosting.** Bäume nacheinander, jeder auf den Fehlern der bisherigen Summe ($F \\leftarrow F + \\eta\\,\\text{Baum}$), blattweise gewachsen, mit Histogrammen für die Split-Suche.
5. **Prognose.** Für jeden Ursprung des Testjahres und jeden Horizont eine Zeile; das Ergebnis wird zurückgerechnet. Zum Vergleich laufen **je Depot** Wochenmittel, Holt-Winters und die Regression auf Kalender und Aktionsplan aus den Vorgängern.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Portfolio**")
    n_depots = st.slider("Zahl der Depots", *bounds("depots_slider"), key="depots_slider", step=C.DEPOTS_STEP, help="Wie viele Depots das Portfolio hat; das Modell lernt aus allen (bis auf die letzten 60 Trainingstage).")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Rauschens; die Depots streuen um diesen Wert.")
    trend = st.slider("Trend (% je Jahr, Mittel)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Mittleres Wachstum der Depots in Prozent des Ausgangsniveaus je Jahr; die Depots streuen um diesen Wert.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Feiertags- und Aktionseffekte (Feiertage gleich für alle Depots, Aktionstage je Depot verschieden).")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird; das Modell lernt alle Horizonte gemeinsam.")
    st.markdown("**Das Modell**")
    groups = st.multiselect("Merkmalsgruppen", list(ALL_GROUPS), key="groups_select", format_func=lambda g: GROUP_LABELS[g], help="Welche Merkmale das Modell sieht. Mindestens eine Gruppe; ohne Auswahl rechnet die App mit dem Kalender.")
    normalize = st.selectbox("Ziel", list(C.NORMALIZE), key="normalize_select", format_func=lambda k: C.NORMALIZE_LABELS[k], help="Was das Modell vorhersagt: ein Verhältnis (log) zum jüngsten Niveau, zum Wochentagsmittel der letzten vier Wochen - oder der rohe Wert (dann sind die Lags Rohwerte und das Niveau ein Merkmal).")
    rounds = st.slider("Boosting-Runden", *bounds("rounds_slider"), key="rounds_slider", step=C.ROUNDS_STEP, help="Wie viele Bäume nacheinander gewachsen werden.")
    leaves = st.slider("Blätter je Baum", *bounds("leaves_slider"), key="leaves_slider", help="Höchstzahl der Blätter eines Baums (blattweises Wachsen).")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=C.LR_STEP, help="Wie stark jeder Baum in die Summe eingeht.")
    min_leaf = st.slider("Mindestzeilen je Blatt", *bounds("minleaf_slider"), key="minleaf_slider", step=C.MINLEAF_STEP, help="Kleinste Zahl an Trainingszeilen in einem Blatt.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Portfolio fest.")
    st.button("🎲 Neues Portfolio generieren", width="stretch", on_click=randomize_seed)

used_groups = [g for g in ALL_GROUPS if g in groups] or ["calendar"]
sync_query_params({"depots_slider": int(n_depots), "noise_slider": round(float(noise), 2), "trend_slider": int(trend), "events_slider": round(float(events), 2), "horizon_slider": int(horizon), "rounds_slider": int(rounds), "leaves_slider": int(leaves),
                   "lr_slider": round(float(lr), 2), "minleaf_slider": int(min_leaf), "groups_select": used_groups, "normalize_select": normalize, "seed_input": int(seed)})

settings = Settings(int(n_depots), round(float(noise), 2), round(float(events), 2), int(trend), int(horizon), int(rounds), int(leaves), round(float(lr), 2), int(min_leaf), "lags" in used_groups, "calendar" in used_groups, "promo" in used_groups, normalize, int(seed))
if not groups:
    st.warning("Keine Merkmalsgruppe gewählt: die App rechnet mit dem Kalender.")
with st.spinner("Das globale Modell wird trainiert ..."):
    a = analyse(settings)
port = a.port
sm = a.summary

# --- Ein Depot -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Depot und die Prognosen an einem Ursprung")
st.session_state["depot_slider"] = min(port.n - 1, max(0, st.session_state.get("depot_slider", 0)))
dep = int(st.slider("Depot", 0, port.n - 1, key="depot_slider", help="Welches Depot des Portfolios gezeigt wird."))
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, dep, origin), width="stretch", key="series_chart")
st.plotly_chart(build_origin(a, dep, origin), width="stretch", key="origin_chart")
st.caption(
    f"Depot {dep}: Niveau {de(port.level[dep], 0)} Aufträge, Rauschen {de(port.noise[dep], 2)}, Wochenmuster {de(port.weekly[dep], 2)}, Jahresmuster {de(port.yearly[dep], 2)}, Trend {de(port.trend[dep], 0)} % je Jahr; MASE: Boosting {de(a.depot_mase['gbm'][dep], 2)}, "
    f"Holt-Winters {de(a.depot_mase['hw_mult'][dep], 2)}, Regression {de(a.depot_mase['regression'][dep], 2)}. Die dicke Linie ist die Prognose des globalen Modells für die nächsten {settings.horizon} Tage; gestrichelt die lokalen Verfahren, grün gepunktet der wahre Erwartungswert."
)

st.markdown("---")

# --- Was das Modell gelernt hat ------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was das Modell gelernt hat")
g1, g2 = st.columns(2)
with g1:
    st.markdown("##### Welche Merkmale die Bäume nutzen")
    st.plotly_chart(build_importance(a), width="stretch", key="importance_chart")
with g2:
    st.markdown("##### Lernkurve")
    st.plotly_chart(build_curve(a), width="stretch", key="curve_chart")
best_round = int(np.argmin(a.curve_val)) + 1
st.caption(
    f"Links: Anteil am Gain (Verbesserung der Fehlerquadrate) der Splits je Merkmal, über alle {len(a.model.trees)} Bäume. Rechts: der mittlere absolute Fehler des Log-Verhältnisses auf den Trainingszeilen und auf den letzten 60 Trainingstagen (Validierung, dem Modell nicht gezeigt); "
    f"der Validierungsfehler ist nach Runde {best_round} am kleinsten ({de(float(a.curve_val[best_round - 1]), 4)}), nach {len(a.curve_val)} Runden {de(float(a.curve_val[-1]), 4)}. {len(a.model.trees)} Bäume mit zusammen {sum(t.n_leaves for t in a.model.trees)} Blättern."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über das Portfolio")
g, hw, rg, wm = sm["gbm"]["mase"], sm["hw_mult"]["mase"], sm["regression"]["mase"], sm["snaive_k"]["mase"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Globales Boosting", f"MASE {de(g, 2)}", delta=f"{g - hw:+.2f}".replace(".", ",") + " gegen Holt-Winters", delta_color="inverse", help="Mittel der MASE über alle Depots; der Pfeil vergleicht mit Holt-Winters aus Stück 2 (grün: besser).")
m2.metric("Holt-Winters (je Depot)", f"MASE {de(hw, 2)}", help="Stück 2, für jedes Depot auf den Trainingstagen geschätzt.")
m3.metric("Regression (je Depot)", f"MASE {de(rg, 2)}", help="Stück 4: lineare Regression im Log auf Wochentag, Trend, Jahresmuster, Feiertage und Aktionsplan, je Depot.")
m4.metric("Wochenmittel (je Depot)", f"MASE {de(wm, 2)}", help="Stück 1.")
m5.metric("Orakel", f"MASE {de(sm['oracle']['mase'], 2)}", help="Fehler der Prognose 'wahrer Erwartungswert': das Rauschen der Reihen. Kein Verfahren liegt im Mittel darunter.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": ("Orakel (wahrer Erwartungswert)" if m == "oracle" else C.METHOD_NAMES[m]), "MASE": de(sm[m]["mase"], 3), "MAE (Aufträge)": de(sm[m]["mae"], 1)} for m in sorted(sm, key=lambda m: sm[m]["mase"])]
st.dataframe(rows, hide_index=True)
wins = int(np.sum(a.depot_mase["gbm"] < a.depot_mase["hw_mult"]))
if a.best == "gbm":
    st.success(f"✅ Das globale Modell liegt vorn: MASE {de(g, 3)} gegen Holt-Winters {de(hw, 3)}, Regression {de(rg, 3)} und Wochenmittel {de(wm, 3)} (Orakel {de(sm['oracle']['mase'], 3)}); gegen Holt-Winters gewinnt es in {wins} von {port.n} Depots.")
elif g < hw:
    st.info(f"Das globale Modell schlägt Holt-Winters ({de(g, 3)} gegen {de(hw, 3)}; in {wins} von {port.n} Depots) und das Wochenmittel ({de(wm, 3)}), nicht aber die Regression je Depot ({de(rg, 3)}): sie kennt die Form des Vehikels (log-linear in Kalender und Aktionsplan) und hat zwei Jahre Historie je Depot; "
            "das Boosting muss die Form aus den Daten lernen. Die Experimente unten zeigen, wann sich das dreht.")
else:
    st.warning(f"⚠️ Hier verliert das globale Modell gegen Holt-Winters ({de(g, 3)} gegen {de(hw, 3)}). Es fehlen Depots, Merkmale oder Bäume: probieren Sie mehr Depots, alle Merkmalsgruppen oder das Ziel 'Verhältnis zum Niveau'.")
st.caption(f"{len(a.origins)} Ursprünge im Testjahr, je {settings.horizon} Tage Horizont, {port.n} Depots; MASE je Depot mit dem saisonal naiven Trainingsfehler als Nenner, dann gemittelt. Das Modell lernt aus den Zieltagen vor Tag {C.FIRST_TEST - 60}; die lokalen Verfahren aus allen {C.FIRST_TEST} Trainingstagen.")

h1, h2 = st.columns(2)
with h1:
    st.markdown("##### Wie der Fehler mit dem Horizont wächst")
    st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")
with h2:
    st.markdown("##### Wo gewinnt das Boosting? (je Depot)")
    st.plotly_chart(build_scatter(a), width="stretch", key="scatter_chart")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Welche Merkmale zählen?")
st.caption(f"{C.EXP_DEPOTS} Depots, Standardeinstellungen; das Modell mit verschiedenen Merkmalsgruppen neben den lokalen Verfahren. Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa eine halbe Minute.")
if st.button("Merkmale durchrechnen", key="features_start"):
    st.session_state["features_on"] = True
if st.session_state.get("features_on"):
    r = _features(C.EXP_SEEDS)
    st.plotly_chart(build_features(r), width="stretch", key="features_chart")
    by = {x["name"]: x["mase"] for x in r["rows"]}
    nm = [x["name"] for x in r["rows"]]
    imp = dict(zip(r["names"], r["importance"]))
    wd = imp["Wochentag"]
    wlag = sum(v for k, v in imp.items() if k.startswith("Wochentag-Lag"))
    last7 = sum(v for k, v in imp.items() if k.startswith("y(t-"))
    st.warning(
        f"**Befund:** Nur mit Lags erreicht das Modell {de(by[nm[0]], 3)}, nur mit dem Kalender {de(by[nm[1]], 3)}; beides zusammen {de(by[nm[2]], 3)} - so gut wie Holt-Winters ({de(r['refs']['hw_mult'], 3)}). Erst der **Aktionsplan** bringt es davor: {de(by[nm[3]], 3)}. "
        f"Die lokale Regression, die den Kalender und die Aktionen in genau der Form des Vehikels kennt, liegt bei {de(r['refs']['regression'], 3)}, das Orakel bei {de(r['refs']['oracle'], 3)}. Die Bäume nutzen vor allem den Zielwochentag ({pct(wd)} des Gains) und die Werte desselben Wochentags in den vier Vorwochen ({pct(wlag)}); "
        f"die letzten sieben Tage tragen zusammen nur {pct(last7, 1)} bei - bei diesem Muster sagen der Wochentag und sein Verlauf fast alles."
    )

st.markdown("---")

st.subheader("🔬 Wie viele Depots braucht das Modell?")
st.caption(f"Das Modell lernt aus {', '.join(str(n) for n in C.SERIES_COUNTS)} Depots und wird auf {C.EXP_TEST} davon getrennten Test-Depots geprüft; daneben die lokalen Verfahren auf denselben Depots mit voller Historie. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Depot-Zahl durchrechnen", key="series_start"):
    st.session_state["series_on"] = True
if st.session_state.get("series_on"):
    rs = _series(C.SERIES_COUNTS, C.EXP_SEEDS)
    st.plotly_chart(build_series_counts(rs), width="stretch", key="series_chart_exp")
    cs = list(rs["counts"])
    beat = [n for n in cs if rs["rows"][n][0] < rs["refs"]["hw_mult"]]
    st.warning(
        f"**Befund:** Aus {cs[0]} Depot lernt das Modell nicht genug ({de(rs['rows'][cs[0]][0], 2)} - schlechter als das Wochenmittel mit {de(rs['refs']['snaive_k'], 2)}), aus {cs[1]} erreicht es {de(rs['rows'][cs[1]][0], 2)}. "
        + (f"Ab {beat[0]} Depots liegt es vor Holt-Winters ({de(rs['rows'][beat[0]][0], 2)} gegen {de(rs['refs']['hw_mult'], 2)}) " if beat else "Es erreicht Holt-Winters nicht ")
        + f"und mit {cs[-1]} Depots bei {de(rs['rows'][cs[-1]][0], 2)}. Die lokale Regression ({de(rs['refs']['regression'], 2)}) bleibt vorn. **Ein globales Modell lohnt sich erst mit vielen Reihen** - und ist dann für jede weitere fast umsonst."
    )

st.markdown("---")

st.subheader("🔬 Ein neues Depot mit kurzer Historie")
st.caption(f"Das Modell lernt aus {C.EXP_TRAIN} anderen Depots; das neue Depot hat nur die letzten {', '.join(str(h) for h in C.HISTORIES)} Tage vor dem Testjahr, aus denen die lokalen Verfahren ihre Parameter schätzen ({C.EXP_TEST} neue Depots). Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Historien durchrechnen", key="history_start"):
    st.session_state["history_on"] = True
if st.session_state.get("history_on"):
    rh = _history(C.HISTORIES, C.EXP_SEEDS)
    st.plotly_chart(build_history(rh), width="stretch", key="history_chart")
    hs = list(rh["histories"])
    lo, hi = hs[0], hs[-1]
    st.warning(
        f"**Befund:** Das globale Modell kennt das neue Depot nicht und erreicht {de(rh['gbm'][0], 2)}, unabhängig von der Historie (es braucht nur die letzten 91 Tage als Merkmale). Mit {lo} Tagen Historie liegen Holt-Winters bei {de(rh['local'][('hw_mult', lo)][0], 2)} und die Regression (ohne Trend und Jahresmuster, die sich aus so wenig Daten nicht schätzen lassen) bei {de(rh['local'][('regression', lo)][0], 2)}; "
        f"auch mit einem Jahr sind es {de(rh['local'][('hw_mult', 365)][0], 2)} und {de(rh['local'][('regression', 365)][0], 2)}. Erst mit {hi} Tagen kommen die lokalen Verfahren heran: Holt-Winters {de(rh['local'][('hw_mult', hi)][0], 2)}, die Regression {de(rh['local'][('regression', hi)][0], 2)} liegt dann vor dem Modell. "
        "Der Wert des globalen Modells liegt bei Neuzugängen: es überträgt, was es aus anderen Depots gelernt hat."
    )

st.markdown("---")

st.subheader("🔬 Wo Bäume an Trends scheitern")
st.caption(f"{C.TREND_DEPOTS} Depots, mittlerer Trend {', '.join(str(t) for t in C.TREND_LEVELS)} % je Jahr; das Modell mit dem Ziel 'Verhältnis zum Niveau' und mit dem rohen Wert, neben Holt-Winters und der Regression. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Trend durchrechnen", key="trend_start"):
    st.session_state["trend_on"] = True
if st.session_state.get("trend_on"):
    rt = _trend(C.TREND_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_trend(rt), width="stretch", key="trend_chart")
    t0, t1 = rt[0], rt[-1]
    st.warning(
        f"**Befund:** Ein Baum sagt konstante Werte je Blatt voraus und kann über die Trainingswerte hinaus nichts fortsetzen. Mit dem **rohen Wert** als Ziel ist das Modell schon ohne Trend schlechter ({de(t0['gbm_none'][0], 3)} gegen {de(t0['gbm_level'][0], 3)} beim Verhältnis zum Niveau) und fällt bei {t1['trend']} % Trend auf {de(t1['gbm_none'][0], 3)}, "
        f"hinter Holt-Winters ({de(t1['hw_mult'][0], 3)}). Mit dem normierten Ziel sind es {de(t1['gbm_level'][0], 3)}: die Normierung lässt die Bäume Verhältnisse lernen, die im Testjahr wieder gelten, und das Modell liegt dann sogar vor Holt-Winters und der Regression ({de(t1['regression'][0], 3)}), die den Trend je Depot im Log schätzen und fortsetzen. "
        f"Die Normierung ist also das, was das Boosting trendfest macht."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt viele ähnliche Reihen** | Aus einem oder wenigen Depots lernt das Modell nicht genug; unähnliche Depots (anderes Muster) ziehen sich gegenseitig herunter. | Hierarchische Abstimmung (Stück 8), Kombination (Stück 9) |
| **Die Form ist unbekannt** | Kennt man die Form (hier: log-linear in Kalender und Aktionsplan), schlägt das einfache Modell mit den passenden Regressoren das Boosting; das Boosting gewinnt, wenn die Form unbekannt oder verwickelt ist. | Dynamische Regression (Stück 4) |
| **Bäume können extrapolieren** | Sie können es nicht: ohne Normierung auf das Niveau scheitern sie an Trends und Niveausprüngen. | Normierung, Regression mit Trend |
| **Der Zufall ist klein** | Die Lernkurve zeigt Überanpassung, wenn Runden und Blätter zu groß sind für die Zahl der Zeilen; Validierung und Seeds sind hier klein gehalten. | Regularisierung, Kreuzvalidierung |
| **Punktprognosen genügen** | Das Modell liefert einen Wert je Zeile; ein Intervall folgt aus einem Quantilverlust oder der konformen Kalibrierung. | Prognoseintervalle (Stück 7) |
| **Erzeugte Portfolios, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, **Boosting**, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zeilen und Ziel.** Zeile $(i, t, j)$: Depot $i$, Ursprung $t$, Horizont $j$, Zieltag $s = t + j - 1$. Niveau $\ell = \frac1{28}\sum_{k=1}^{28} y_{i,t-k}$. Ziel $z = \log\frac{y_{i,s} + 1}{\ell + 1}$; Prognose $\hat y = \max\big(e^{\hat z}(\ell + 1) - 1,\,0\big)$.
Merkmale: $y_{i,t-k}/(\ell + 1)$ für $k = 1..7$, $y_{i,s-7(k_0+m)}/(\ell + 1)$ für $m = 0..3$ mit $k_0 = \lceil j/7 \rceil$, das 7-Tage-Mittel durch $\ell + 1$, $\ell$ durch das 91-Tage-Niveau, die Streuung der letzten 28 Tage, Wochentag, $\sin/\cos(2\pi\,\mathrm{doy}_s/365)$, Feiertag, Tag danach, $j$, Aktion am Zieltag und Anteil der Aktionstage in den letzten sieben Tagen.

**Boosting** (quadratischer Fehler). $F_0 = \bar z$, $F_m = F_{m-1} + \eta\,f_m$; $f_m$ ist ein Baum auf den Residuen $g_r = F_{m-1}(x_r) - z_r$ mit Hesse 1. Split-Gain $\tfrac12\big[\tfrac{G_L^2}{n_L + \lambda} + \tfrac{G_R^2}{n_R + \lambda} - \tfrac{G^2}{n + \lambda}\big]$, Blattwert $-G/(n + \lambda)$;
Histogramm-Splits über 63 Quantil-Bins, blattweises Wachsen bis zur Blattzahl, Mindestzeilen je Blatt; das kleinere Kind wird histogrammiert, das größere per Differenz.

**Training.** Je Depot alle vier Tage ein Ursprung, je Ursprung zwei zufällige Horizonte; Zieltage vor Tag 670. Die Zeilen mit Zieltagen in den letzten 60 Trainingstagen sind die Validierung der Lernkurve. **MASE** je Depot: MAE geteilt durch den mittleren absoluten Fehler der saisonal naiven Prognose (Periode 7) auf den Tagen vor 730; Mittel über die Depots.
**Lokale Verfahren:** Wochenmittel über vier Wochen; Holt-Winters multiplikativ (Stück 2); OLS im Log auf Konstante, Wochentage, Trend, zwei Fourier-Paare, Feiertag, Tag danach, Aktion (Stück 4), Prognose $e^{x'\hat\beta}$.

Implementiert in `bf_tree.py` (Baumkern), `bf_gbm.py` (Boosting), `bf_features.py` (Zeilen, Merkmale, Ziel), `bf_baselines.py` (lokale Verfahren), `bf_scenario.py` (das Portfolio), `bf_evaluation.py` (Analyse, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
