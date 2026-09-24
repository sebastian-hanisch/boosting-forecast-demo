"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Depot-/Ursprungs-Regler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bf_constants as C
import bf_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 5 and len(at.get("plotly_chart")) == 7 and len(at.info) + len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_depot_and_origin_sliders_survive_a_smaller_portfolio():
    at = _run(depot_slider=39, origin_slider=1000)
    _ok(at)
    at.slider(key="depots_slider").set_value(10).run()
    _ok(at)
    assert at.session_state["depot_slider"] <= 9


def test_origin_slider_survives_a_longer_horizon():
    at = _run(origin_slider=1094 - 14)
    _ok(at)
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_empty_feature_groups_fall_back_to_the_calendar():
    at = _run(groups_select=[])
    _ok(at)
    assert any("Keine Merkmalsgruppe" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Portfolio generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["depots"] = "77"
    at.query_params["noise"] = "9"
    at.query_params["horizon"] = "0"
    at.query_params["groups"] = "promo,lags"
    at.query_params["norm"] = "quatsch"
    at.query_params["rounds"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["depots_slider"] == 80 and at.session_state["noise_slider"] == C.NOISE_MAX and at.session_state["horizon_slider"] == C.HORIZON_MIN
    assert at.session_state["groups_select"] == ["lags", "promo"] and at.session_state["normalize_select"] == "level" and at.session_state["rounds_slider"] == C.DEFAULT_ROUNDS


@pytest.mark.parametrize("kw", [dict(depots_slider=C.DEPOTS_MIN, rounds_slider=C.ROUNDS_MIN, leaves_slider=C.LEAVES_MIN), dict(horizon_slider=C.HORIZON_MAX, lr_slider=C.LR_MAX, minleaf_slider=C.MINLEAF_MAX), dict(horizon_slider=1, noise_slider=C.NOISE_MIN, events_slider=0.0),
                                dict(trend_slider=C.TREND_MIN, normalize_select="none"), dict(groups_select=["promo"], normalize_select="seasonal"), dict(groups_select=["lags"], normalize_select="none", trend_slider=C.TREND_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    monkeypatch.setattr(C, "EXP_DEPOTS", 10)
    monkeypatch.setattr(C, "EXP_TRAIN", 10)
    monkeypatch.setattr(C, "EXP_TEST", 5)
    monkeypatch.setattr(C, "TREND_DEPOTS", 10)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_features_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "features_start")
    assert at.session_state["features_on"] and any("Aktionsplan** bringt es" in w.value for w in at.warning)


def test_series_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "SERIES_COUNTS", (1, 3, 10))
    at = _run()
    _click(at, "series_start")
    assert at.session_state["series_on"] and any("erst mit vielen Reihen" in w.value for w in at.warning)


def test_history_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "history_start")
    assert at.session_state["history_on"] and any("kennt das neue Depot nicht" in w.value for w in at.warning)


def test_trend_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "trend_start")
    assert at.session_state["trend_on"] and any("trendfest" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
