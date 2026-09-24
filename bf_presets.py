"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. cr_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import bf_constants as C


def _bool(value):
    v = str(value).strip().lower()
    if v in ("1", "true", "ja", "yes"):
        return True
    if v in ("0", "false", "nein", "no"):
        return False
    raise ValueError(value)


def _groups(value):
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = [v.strip() for v in str(value).split(",") if v.strip()]
    if not parts or any(p not in ALL_GROUPS for p in parts):
        raise ValueError(value)
    return [g for g in ALL_GROUPS if g in parts]


def _normalize(value):
    v = str(value).strip().lower()
    if v not in C.NORMALIZE:
        raise ValueError(value)
    return v


ALL_GROUPS = ("lags", "calendar", "promo")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "depots_slider": SettingSpec("depots", int, C.DEFAULT_DEPOTS, C.DEPOTS_MIN, C.DEPOTS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "rounds_slider": SettingSpec("rounds", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX),
    "leaves_slider": SettingSpec("leaves", int, C.DEFAULT_LEAVES, C.LEAVES_MIN, C.LEAVES_MAX),
    "lr_slider": SettingSpec("lr", float, C.DEFAULT_LR, C.LR_MIN, C.LR_MAX),
    "minleaf_slider": SettingSpec("minleaf", int, C.DEFAULT_MIN_LEAF, C.MINLEAF_MIN, C.MINLEAF_MAX),
    "groups_select": SettingSpec("groups", _groups, list(ALL_GROUPS)),
    "normalize_select": SettingSpec("norm", _normalize, "level"),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_depots": "depots_slider", "noise": "noise_slider", "trend": "trend_slider", "events": "events_slider", "horizon": "horizon_slider", "rounds": "rounds_slider", "leaves": "leaves_slider", "lr": "lr_slider",
               "min_leaf": "minleaf_slider", "groups": "groups_select", "normalize": "normalize_select", "seed": "seed_input"}
STEPS = {"depots_slider": C.DEPOTS_STEP, "noise_slider": C.NOISE_STEP, "trend_slider": C.TREND_STEP, "events_slider": C.EVENTS_STEP, "rounds_slider": C.ROUNDS_STEP, "lr_slider": C.LR_STEP, "minleaf_slider": C.MINLEAF_STEP}


def _p(**kw):
    base = {"n_depots": C.DEFAULT_DEPOTS, "noise": C.DEFAULT_NOISE, "trend": C.DEFAULT_TREND, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON, "rounds": C.DEFAULT_ROUNDS, "leaves": C.DEFAULT_LEAVES, "lr": C.DEFAULT_LR,
            "min_leaf": C.DEFAULT_MIN_LEAF, "groups": list(ALL_GROUPS), "normalize": "level", "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: 40 Depots": _p(),
    "Wenige Depots (10)": _p(n_depots=10),
    "Nur Lag-Merkmale": _p(groups=["lags"]),
    "Ohne Aktionsplan": _p(groups=["lags", "calendar"]),
    "Starker Trend (+40 %), rohes Ziel": _p(trend=40, normalize="none"),
    "Viel Rauschen (0,3)": _p(noise=0.3),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = ",".join(value) if isinstance(value, (list, tuple)) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        v = PRESETS[name][key]
        st.session_state[state_key] = list(v) if isinstance(v, list) else v


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: 40 Depots": "Alle Merkmale, Ziel Verhältnis zum Niveau (Seed 3): Boosting 0,82 gegen Holt-Winters 0,89 und Wochenmittel 0,98, in allen 40 Depots vor Holt-Winters; die Regression je Depot liegt mit 0,77 vorn (Orakel 0,74).",
    "Wenige Depots (10)": "Nur zehn Depots zum Lernen: Boosting 0,82 gegen Holt-Winters 0,87, in 9 von 10 Depots besser; schon zehn Reihen reichen, aus einer oder drei nicht (Experiment 'Wie viele Depots').",
    "Nur Lag-Merkmale": "Ohne Kalender und Aktionsplan: Boosting 0,94, schlechter als Holt-Winters (0,89) und nur in einem Depot besser - der Aktionsplan fehlt, und ohne Kalender lernt der Baum den Wochentag nur über Lags.",
    "Ohne Aktionsplan": "Lags und Kalender, aber keine Aktion am Zieltag: Boosting 0,88, kaum besser als Holt-Winters (0,89); der Aktionsplan bringt es auf 0,82.",
    "Starker Trend (+40 %), rohes Ziel": "Ziel roher Wert statt Verhältnis: Boosting 1,13, hinter Holt-Winters (1,11); der Baum kann den Trend nicht fortsetzen. Mit dem Ziel 'Verhältnis zum Niveau' stellt sich das um.",
    "Viel Rauschen (0,3)": "Doppeltes Rauschen: alle Verfahren nähern sich dem Orakel (0,80); Boosting 0,85 gegen Holt-Winters 0,87, der Vorsprung schrumpft, die Regression bleibt vorn (0,81).",
}
