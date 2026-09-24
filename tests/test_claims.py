"""Jede Zahl aus README und PRESET_HELP als Test. Das Boosting ist deterministisch bei festem Seed, aber Baumteilungen bei Gleichständen können sich zwischen numpy-Versionen und Plattformen um Rundungsfehler verschieben:
Einzelzahlen mit Bändern um die gerundete Angabe, dazu Rangfolgen mit Abstand (feedback_ci_platform_robust_tests)."""

from functools import lru_cache

import numpy as np
import pytest

import bf_constants as C
import bf_evaluation as E
import bf_presets as P

STD = "Standardfall: 40 Depots"


@lru_cache(maxsize=None)
def _preset(name):
    p = P.PRESETS[name]
    g = p["groups"]
    return E.analyse(E.Settings(p["n_depots"], p["noise"], p["events"], p["trend"], p["horizon"], p["rounds"], p["leaves"], p["lr"], p["min_leaf"], "lags" in g, "calendar" in g, "promo" in g, p["normalize"], p["seed"]))


def _m(a):
    return {k: v["mase"] for k, v in a.summary.items()}


def _wins(a):
    return int(np.sum(a.depot_mase["gbm"] < a.depot_mase["hw_mult"]))


def test_standard_preset():
    a = _preset(STD)
    m = _m(a)
    assert m["gbm"] == pytest.approx(0.82, abs=0.03) and m["hw_mult"] == pytest.approx(0.894, abs=0.02) and m["snaive_k"] == pytest.approx(0.983, abs=0.02) and m["regression"] == pytest.approx(0.77, abs=0.02) and m["oracle"] == pytest.approx(0.735, abs=0.01)
    assert m["regression"] < m["gbm"] < m["hw_mult"] < m["snaive_k"] and _wins(a) >= 35 and a.best == "regression"
    assert m["gbm"] - m["oracle"] < 0.15


def test_few_depots_preset():
    a = _preset("Wenige Depots (10)")
    m = _m(a)
    assert m["gbm"] == pytest.approx(0.819, abs=0.03) and m["hw_mult"] == pytest.approx(0.866, abs=0.02) and m["gbm"] < m["hw_mult"] and _wins(a) >= 7


def test_lags_only_preset():
    a = _preset("Nur Lag-Merkmale")
    m = _m(a)
    assert m["gbm"] == pytest.approx(0.944, abs=0.03) and m["gbm"] > m["hw_mult"] + 0.02 and _wins(a) <= 6


def test_no_promo_preset():
    a = _preset("Ohne Aktionsplan")
    m = _m(a)
    assert m["gbm"] == pytest.approx(0.883, abs=0.03) and abs(m["gbm"] - m["hw_mult"]) < 0.05 and m["gbm"] > _m(_preset(STD))["gbm"] + 0.03


def test_strong_trend_raw_target_preset():
    a = _preset("Starker Trend (+40 %), rohes Ziel")
    m = _m(a)
    assert m["gbm"] == pytest.approx(1.13, abs=0.04) and m["hw_mult"] == pytest.approx(1.109, abs=0.03) and m["gbm"] > m["hw_mult"] - 0.005 and m["oracle"] == pytest.approx(0.913, abs=0.02)


def test_noisy_preset():
    a = _preset("Viel Rauschen (0,3)")
    m = _m(a)
    assert m["gbm"] == pytest.approx(0.846, abs=0.03) and m["hw_mult"] == pytest.approx(0.872, abs=0.02) and m["regression"] == pytest.approx(0.812, abs=0.02) and m["oracle"] == pytest.approx(0.802, abs=0.01)
    assert m["hw_mult"] - m["gbm"] < _m(_preset(STD))["hw_mult"] - _m(_preset(STD))["gbm"]


def test_learning_curve_bottoms_out_before_the_end_in_the_standard_case():
    a = _preset(STD)
    best = int(np.argmin(a.curve_val)) + 1
    assert 30 <= best <= 80 and a.curve_val[-1] < a.curve_val[0] and a.curve_train[-1] < a.curve_train[0]


@lru_cache(maxsize=None)
def _features():
    return E.features_experiment(seeds=C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _series():
    return E.series_experiment(C.SERIES_COUNTS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _history():
    return E.history_experiment(C.HISTORIES, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _trend():
    return E.trend_experiment(C.TREND_LEVELS, C.EXP_SEEDS)


def test_features_experiment():
    r = _features()
    v = [x["mase"] for x in r["rows"]]
    assert [x["name"] for x in r["rows"]] == ["nur Lags", "nur Kalender", "Lags + Kalender", "Lags + Kalender + Aktionsplan"]
    assert v[0] == pytest.approx(0.887, abs=0.04) and v[1] == pytest.approx(1.014, abs=0.05) and v[2] == pytest.approx(0.839, abs=0.04) and v[3] == pytest.approx(0.792, abs=0.03)
    assert r["refs"]["hw_mult"] == pytest.approx(0.840, abs=0.02) and r["refs"]["regression"] == pytest.approx(0.729, abs=0.02) and r["refs"]["oracle"] == pytest.approx(0.709, abs=0.01)
    assert v[3] < v[2] < v[0] < v[1] and abs(v[2] - r["refs"]["hw_mult"]) < 0.05 and r["refs"]["regression"] < v[3]
    imp = dict(zip(r["names"], r["importance"]))
    wlag = sum(x for k, x in imp.items() if k.startswith("Wochentag-Lag"))
    last7 = sum(x for k, x in imp.items() if k.startswith("y(t-"))
    assert imp["Wochentag"] == pytest.approx(0.52, abs=0.08) and wlag == pytest.approx(0.44, abs=0.08) and last7 < 0.03


def test_series_experiment():
    r = _series()
    rows = {n: r["rows"][n][0] for n in r["counts"]}
    assert rows[1] == pytest.approx(1.25, abs=0.15) and rows[3] == pytest.approx(0.86, abs=0.05) and rows[10] == pytest.approx(0.795, abs=0.04) and rows[30] == pytest.approx(0.775, abs=0.04)
    assert r["refs"]["hw_mult"] == pytest.approx(0.821, abs=0.02) and r["refs"]["regression"] == pytest.approx(0.714, abs=0.02) and r["refs"]["snaive_k"] == pytest.approx(0.906, abs=0.02)
    assert rows[1] > r["refs"]["snaive_k"] + 0.1 and rows[1] > rows[3] > rows[10] > rows[30] - 0.01 and rows[30] < r["refs"]["hw_mult"] and rows[3] > r["refs"]["hw_mult"] and rows[30] > r["refs"]["regression"]


def test_history_experiment():
    r = _history()
    h = {k: v[0] for k, v in r["local"].items()}
    assert r["gbm"][0] == pytest.approx(0.775, abs=0.04) and r["snaive_k"] == pytest.approx(0.906, abs=0.02)
    assert h[("hw_mult", 91)] == pytest.approx(1.285, abs=0.25) and h[("hw_mult", 182)] == pytest.approx(0.906, abs=0.08) and h[("hw_mult", 365)] == pytest.approx(0.839, abs=0.03) and h[("hw_mult", 730)] == pytest.approx(0.821, abs=0.02)
    assert h[("regression", 91)] == pytest.approx(1.328, abs=0.15) and h[("regression", 182)] == pytest.approx(1.244, abs=0.15) and h[("regression", 365)] == pytest.approx(1.231, abs=0.1) and h[("regression", 730)] == pytest.approx(0.714, abs=0.02)
    assert r["gbm"][0] < h[("hw_mult", 365)] < h[("hw_mult", 182)] < h[("hw_mult", 91)] and r["gbm"][0] < h[("regression", 365)] and h[("regression", 730)] < r["gbm"][0] < h[("hw_mult", 730)]


def test_trend_experiment():
    r = {x["trend"]: x for x in _trend()}
    assert r[0]["gbm_level"][0] == pytest.approx(0.708, abs=0.04) and r[0]["gbm_none"][0] == pytest.approx(0.757, abs=0.04) and r[0]["hw_mult"][0] == pytest.approx(0.760, abs=0.03) and r[0]["regression"][0] == pytest.approx(0.641, abs=0.03)
    assert r[40]["gbm_level"][0] == pytest.approx(1.022, abs=0.05) and r[40]["gbm_none"][0] == pytest.approx(1.144, abs=0.05) and r[40]["hw_mult"][0] == pytest.approx(1.096, abs=0.04) and r[40]["regression"][0] == pytest.approx(1.087, abs=0.04)
    assert r[0]["gbm_level"][0] < r[0]["gbm_none"][0] and r[40]["gbm_level"][0] + 0.03 < r[40]["gbm_none"][0] and r[40]["gbm_none"][0] > r[40]["hw_mult"][0]
    assert r[40]["gbm_level"][0] < r[40]["hw_mult"][0] and r[40]["gbm_level"][0] < r[40]["regression"][0]
