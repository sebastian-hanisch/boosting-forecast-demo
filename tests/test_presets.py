"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import bf_constants as C
import bf_evaluation as E
import bf_presets as P


def _settings(p):
    g = p["groups"]
    return E.Settings(p["n_depots"], p["noise"], p["events"], p["trend"], p["horizon"], p["rounds"], p["leaves"], p["lr"], p["min_leaf"], "lags" in g, "calendar" in g, "promo" in g, p["normalize"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_depots", "depots_slider"), ("noise", "noise_slider"), ("trend", "trend_slider"), ("events", "events_slider"), ("rounds", "rounds_slider"), ("lr", "lr_slider"), ("min_leaf", "minleaf_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        assert p["groups"] and set(p["groups"]) <= set(P.ALL_GROUPS) and p["normalize"] in C.NORMALIZE


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: 40 Depots"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("depots_slider") == (C.DEPOTS_MIN, C.DEPOTS_MAX) and P.bounds("lr_slider") == (C.LR_MIN, C.LR_MAX)
    assert set(P.STEPS) == {"depots_slider", "noise_slider", "trend_slider", "events_slider", "rounds_slider", "lr_slider", "minleaf_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_group_and_normalize_casters():
    assert P._groups("promo,lags") == ["lags", "promo"] and P._groups(["calendar"]) == ["calendar"] and P._normalize(" NONE ") == "none"
    for bad in ("", "x", "lags,foo"):
        try:
            P._groups(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        P._normalize("foo")
    except ValueError:
        return
    raise AssertionError("normalize")
