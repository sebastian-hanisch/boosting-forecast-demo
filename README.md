# 🌲 Boosting mit Lag-Merkmalen – ein Modell für alle Depots

**[→ Demo live ausprobieren](https://sebastianhanisch-boosting-forecast-demo.streamlit.app/)**

Sechstes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Das **erste globale Modell** und die **erste überwachte Lernaufgabe** der Linie: Nachfolger der [Dynamischen Regression](https://github.com/sebastian-hanisch/dynamic-regression-demo), die für jedes Depot ein eigenes Modell schätzte.
Geplant sind fünf weitere Stücke (Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Bisher bekam **jedes Depot sein eigenes Modell**. Ein **globales Modell** dreht das um: **ein** Gradient-Boosting-Modell lernt aus den Lags, dem Kalender und dem Aktionsplan **aller Depots gemeinsam** und prognostiziert jedes einzelne. Die Demo zeigt auf einem **Portfolio von Depots** (dieselben Tagesaufträge wie in den Vorgängern, jetzt mit eigenen Parametern je Depot), was das bringt und wann nicht:
welche Merkmale zählen, **wie viele Depots** das Modell braucht, wie es einem **neuen Depot mit kurzer Historie** hilft und wo Bäume an **Trends** scheitern. Die Baumbibliothek ist selbst geschrieben (numpy, der Histogramm-Kern aus dem [LightGBM-Stück](https://github.com/sebastian-hanisch/lightgbm-demo)) und im Test gegen `sklearn` geprüft.

**Bezug zu OR:** wer hunderte Depots, Filialen oder Artikel prognostiziert, kann nicht hunderte Modelle pflegen – und braucht eine Prognose für Neuzugänge ohne Historie.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Das Boosting schlägt die lokalen Verfahren, weil es aus vielen Reihen lernt." Die Messung sagt: **es schlägt die Verfahren, die keine Einflussgrößen kennen, aber nicht die passend gebaute lokale Regression mit voller Historie.**

1. **Standardfall (40 Depots).** Das Boosting erreicht eine MASE von 0,82, Holt-Winters 0,89, das Wochenmittel 0,98; in allen 40 Depots liegt es vor Holt-Winters. Die **lokale Regression** (log-linear in Wochentag, Trend, Jahresmuster, Feiertag und Aktion, je Depot mit zwei Jahren Historie) liegt bei **0,77**, das Orakel bei 0,74. Sie kennt die Form des Vehikels; das Boosting muss sie aus den Daten lernen.
2. **Der Aktionsplan entscheidet.** Nur mit Lag-Merkmalen 0,89, nur mit Kalender 1,01, beides 0,84 (Holt-Winters 0,84), erst mit dem Aktionsplan **0,79**. Die Bäume nutzen vor allem den Zielwochentag (52 % des Gains) und die Werte desselben Wochentags in den vier Vorwochen (44 %); die letzten sieben Tage zusammen tragen weniger als 3 % bei.
3. **Ein globales Modell braucht Reihen.** Aus einem Depot lernt es 1,25 (schlechter als das Wochenmittel mit 0,91), aus drei 0,86, aus zehn 0,80 – ab zehn liegt es vor Holt-Winters (0,82) –, aus 30 0,78. Die Regression (0,71) bleibt vorn.
4. **Bei kurzer Historie liegt der Wert des Modells.** Für ein neues Depot, das nur die letzten 91 Tage kennt, erreicht Holt-Winters 1,29 und die Regression 1,33; das globale Modell (aus anderen Depots gelernt) 0,77, unabhängig von der Historie. Mit einem Jahr: 0,84 und 1,23; erst mit zwei Jahren (0,82 und 0,71) holen die lokalen Verfahren auf – die Regression zieht vorbei.
5. **Bäume können nicht extrapolieren – die Normierung hilft.** Bei 40 % Trend je Jahr erreicht das Modell mit dem **rohen Wert** als Ziel 1,14 (hinter Holt-Winters 1,10), mit dem **Verhältnis zum jüngsten Niveau** 1,02 – vor Holt-Winters und der Regression (1,09). Schon ohne Trend kostet das rohe Ziel: 0,76 gegen 0,71.

## Modell

- **Das Portfolio** (`bf_scenario.py`): 1 095 Tage je Depot (Ursprünge im letzten Jahr, ab Tag 730), wie in den Vorgängern multiplikativ aufgebaut (Niveau, Trend, Wochenmuster, Jahresmuster, Feiertage, Aktionen, log-normales Rauschen), aber mit eigenen Parametern je Depot (Niveau 30–500, Rauschen, Stärke von Wochen- und Jahresmuster, Trend). Feiertage gelten für alle Depots, Aktionstage nur je Depot. Der Erwartungswert je Tag ist das **Orakel**.
- **Zeilen und Merkmale** (`bf_features.py`): eine Zeile ist ein Tripel (Depot, Ursprung $t$, Horizont $j$), Ziel der Wert am Zieltag $s = t + j - 1$ – ein **direktes** Mehrschritt-Modell, der Horizont ist ein Merkmal. **Lags:** die letzten sieben Tage, die letzten vier Werte des Zielwochentags (im Abstand von Wochen, alle vor $t$), 7-Tage-Mittel durch 28-Tage-Niveau, 28- durch 91-Tage-Niveau, die Streuung.
  **Kalender:** Wochentag, $\sin/\cos$ des Jahrestags, Feiertag, Tag danach, Horizont. **Aktionsplan:** Aktion am Zieltag, Anteil der Aktionstage in den letzten sieben Tagen. **Ziel** (Regler): $\log\frac{y_s + 1}{\ell + 1}$ mit dem 28-Tage-Niveau $\ell$ (oder dem Wochentagsmittel der letzten vier Wochen) – oder der rohe Wert.
- **Boosting** (`bf_gbm.py`, `bf_tree.py`): quadratischer Fehler, $F_m = F_{m-1} + \eta f_m$, Bäume blattweise auf den Residuen (Hesse 1), Blattwert $-G/(H+\lambda)$, 63 Quantil-Bins, Differenz-Trick. Training auf 2 Ursprüngen je Depot alle 4 Tage (Zieltage vor Tag 670); die Tage 670–729 sind die Validierung der Lernkurve.
- **Lokale Vergleichsverfahren** (`bf_baselines.py`): Wochenmittel über vier Wochen (Stück 1), Holt-Winters multiplikativ (Stück 2), OLS im Log auf Wochentage, Trend, zwei Fourier-Paare, Feiertag, Tag danach, Aktion (Stück 4) – alle je Depot auf den 730 Trainingstagen.
- **Kennzahl** (`bf_evaluation.py`): **MASE** je Depot (MAE über alle Ursprünge und Horizonte des Testjahres, geteilt durch den saisonal naiven Trainingsfehler), gemittelt über die Depots. Das **Orakel** ist die Prognose "wahrer Erwartungswert".

## Methodik

- **Handrechnungen:** ein Split (Gain $\tfrac12(\tfrac43 + \tfrac43)$, Blattwerte $\pm\tfrac23$) und eine Boosting-Runde ($F_0 = 2$, Blätter $-\tfrac43$ / $+\tfrac43$ bei Lernrate 1); eine Merkmalszeile Wert für Wert (Lags, Wochentags-Lags für Horizonte 1 bis 28, Kalender, Aktion, Ziel); die Rückrechnung des Ziels für alle drei Zielvarianten; MASE-Nenner und Kennzahlen auf kleinen Beispielen.
- **Gegenprobe:** das Boosting gegen `sklearn.ensemble.HistGradientBoostingRegressor` (gleiche Blätter, Lernrate, Mindestzeilen, $\lambda$, Bins): der mittlere absolute Testfehler weicht um weniger als 6 % ab (in der Entwicklung 0,1232 gegen 0,1235); die Regression je Depot gegen eine Schleife und `numpy.linalg.lstsq`.
- **Eigenschaften:** der quadratische Fehler fällt mit jeder Runde; die Gain-Anteile summieren sich auf 1 und finden das einzige relevante Merkmal; **seltene Binärmerkmale (3 % Einsen) sind teilbar**; die Merkmale einer Zeile ändern sich nicht, wenn die Reihe ab dem Ursprung überschrieben wird; die Prognose auch nicht.
- **Ein Fund am Baumkern:** die Bin-Grenzen des LightGBM-Stücks lagen für seltene oder diskrete Merkmale so, dass Schwelle und Bin nicht zusammenpassten (die Merkmale waren unteilbar, das Modell ignorierte den Wochentag: MASE 1,47 statt 0,79). Der Kern hier bestimmt die Bins mit `side="left"`; derselbe Fehler steckt im LightGBM-Stück.
- **Statistik:** die Experimente mitteln über **drei feste Seeds** (Fehlerbalken = Standardfehler); die Preset-Zeilen sind **Einzelportfolios** (Seed 3).
- **Literatur** (nicht nachgebaut): Friedman 2001 (Gradient Boosting); Ke et al. 2017 (LightGBM); Montero-Manso/Hyndman 2021 (globale gegen lokale Modelle); Januschowski et al. 2020 (Kriterien für die Modellwahl).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Preset, 40 Depots, Seed 3) | MASE: Regression **0,77**, Boosting **0,82**, Holt-Winters 0,894, Wochenmittel 0,983; Orakel 0,735. Das Boosting schlägt Holt-Winters in 40 von 40 Depots. | `test_standard_preset` |
| **Welche Merkmale?** (30 Depots, 3 Seeds) | nur Lags 0,887; nur Kalender 1,014; Lags + Kalender 0,839; **+ Aktionsplan 0,792**. Holt-Winters 0,840, Regression 0,729, Orakel 0,709. Gain-Anteil: Wochentag 52 %, Wochentags-Lags 44 %, letzte sieben Tage < 3 %. | `test_features_experiment` |
| **Wie viele Depots?** (Lernen aus 1 / 3 / 10 / 30, Test auf 20 anderen Depots) | 1,25 / 0,86 / 0,795 / 0,775. Lokal (volle Historie): Wochenmittel 0,906, Holt-Winters 0,821, Regression 0,714. | `test_series_experiment` |
| **Neues Depot, kurze Historie** (Historie 91 / 182 / 365 / 730 Tage) | Boosting (aus 30 anderen Depots) 0,775 unabhängig von der Historie. Holt-Winters 1,285 / 0,906 / 0,839 / 0,821; Regression 1,328 / 1,244 / 1,231 / **0,714**. | `test_history_experiment` |
| **Trend und Ziel** (20 Depots, Trend 0 / 40 % je Jahr) | Ohne Trend: Boosting mit normiertem Ziel 0,708, roh 0,757, Holt-Winters 0,760, Regression 0,641. Bei 40 %: **normiert 1,022**, roh 1,144, Holt-Winters 1,096, Regression 1,087. | `test_trend_experiment` |
| Wenige Depots (Preset, 10 Depots) | Boosting 0,819 gegen Holt-Winters 0,866, in 9 von 10 Depots besser. | `test_few_depots_preset` |
| Nur Lag-Merkmale (Preset) | Boosting 0,944, schlechter als Holt-Winters (0,894), nur in einem Depot besser. | `test_lags_only_preset` |
| Ohne Aktionsplan (Preset) | Boosting 0,883 (Holt-Winters 0,894). | `test_no_promo_preset` |
| Starker Trend, rohes Ziel (Preset, +40 %) | Boosting 1,13, Holt-Winters 1,109, Regression 1,164, Orakel 0,913. | `test_strong_trend_raw_target_preset` |
| Viel Rauschen (Preset, 0,3) | Boosting 0,846, Holt-Winters 0,872, Regression 0,812, Orakel 0,802: der Vorsprung schrumpft. | `test_noisy_preset` |

Die Preset-Zeilen sind **Einzelportfolios** (Seed 3); belastbar sind die Zeilen über drei Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt viele ähnliche Reihen** | Aus einem oder wenigen Depots lernt das Modell nicht genug; unähnliche Depots ziehen sich gegenseitig herunter. | Hierarchische Abstimmung, Kombination (geplant) |
| **Die Form ist unbekannt** | Kennt man die Form, schlägt das einfache Modell mit den passenden Regressoren das Boosting; das Boosting gewinnt bei unbekannter oder verwickelter Form und bei kurzen Historien. | Dynamische Regression (Stück 4) |
| **Bäume können extrapolieren** | Sie können es nicht: ohne Normierung auf das Niveau scheitern sie an Trends. | Normierung, Regression mit Trend |
| **Der Zufall ist klein** | Überanpassung bei zu vielen Runden und Blättern; hier nur mit einer Validierung und drei Seeds geprüft, keine Kreuzvalidierung, keine Abstimmung der Parameter. | Regularisierung, Suche |
| **Punktprognosen genügen** | Das Modell liefert einen Wert je Zeile. | Prognoseintervalle (geplant) |
| **Erzeugte Portfolios, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, Aktionen 50 % Zuschlag); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, 76 Tests, einige Minuten wegen der Experimente): Baumkern und Boosting von Hand, gegen `sklearn` und in ihren Eigenschaften (fallender Fehler, Gain-Anteile, seltene Binärmerkmale), die Merkmale Wert für Wert und ohne Blick in die Zukunft, die lokalen Verfahren gegen Schleifen und `lstsq`, das Portfolio, die Auswertung (Formen, Orakel, MASE, Unabhängigkeit von der Zukunft),
Preset- und Permalink-Klemmen, AppTest-Rauchtests (Standard, jedes Preset, Depot- und Ursprungs-Regler, leere Merkmalsauswahl, Extremwerte, vier Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen mit Bändern und Rangfolgen; die Bäume können sich bei Gleichständen zwischen Plattformen um Rundung verschieben).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `bf_constants.py` | Regler-Grenzen, Verfahren, Experiment-Seeds |
| `bf_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `bf_scenario.py` | Das Portfolio (Depots, Kalender, Aktionen, Orakel) |
| `bf_features.py` | Zeilen, Merkmale, Ziel und Rückrechnung |
| `bf_tree.py`, `bf_gbm.py` | Histogramm-Baum und Boosting |
| `bf_baselines.py`, `bf_ets.py` | Wochenmittel, Holt-Winters, Regression je Depot |
| `bf_evaluation.py` | Analyse, MASE, vier Experimente |
| `bf_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Prognoseintervalle und Quantilverluste (Stück 7).
- Rekursive statt direkter Mehrschritt-Prognose; Kreuzvalidierung und Parametersuche; externe Bibliotheken für die Bäume (die Gegenprobe im Test nutzt `sklearn`).
- Depot-Kennungen als Merkmal und Transferlernen zwischen Portfolios.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: scikit-learn).
