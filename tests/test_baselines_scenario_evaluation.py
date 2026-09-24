"""Lokale Verfahren gegen Schleifen und Handrechnung, Vehikel, Auswertung."""

import numpy as np
import pytest

import bf_baselines as B
import bf_constants as C
import bf_evaluation as E
import bf_scenario as S


@pytest.fixture(scope="module")
def port():
    return S.generate(4, 0.14, 0.5, 10.0, 7)


def test_snaive_k_by_loop(port):
    y, org, h = port.y[0], np.array([730, 800]), 10
    got = B.snaive_k(y, org, h)
    for a, t in enumerate(org):
        for j in range(h):
            assert got[a, j] == pytest.approx(np.mean([y[t - 7 * (i + 1) + (j % 7)] for i in range(4)]))


def test_snaive_k_uses_only_the_past(port):
    y = port.y[0].copy()
    org = np.array([760])
    a = B.snaive_k(y, org, 14)
    y[760:] = 0
    assert np.array_equal(a, B.snaive_k(y, org, 14))


def _ridge(Xt, z):
    A = Xt.T @ Xt
    return np.linalg.solve(A + 1e-3 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), Xt.T @ z)


def test_regression_matches_a_loop_and_lstsq(port):
    y = port.y[1]
    X = B.regression_design(port.dow, port.holiday, port.after, port.promo[1])
    org, h = np.array([730, 750]), 6
    got = B.regression_forecast(y, X, org, h)
    Xt, z = X[:C.FIRST_TEST], np.log(np.maximum(y[:C.FIRST_TEST], 1.0))
    beta = _ridge(Xt, z)
    for a, t in enumerate(org):
        for j in range(h):
            assert got[a, j] == pytest.approx(np.exp(X[t + j] @ beta))
    beta_ols = np.linalg.lstsq(Xt, z, rcond=None)[0]
    assert np.allclose(got[0], np.exp(X[730:736] @ beta_ols), rtol=0.03)


def test_short_history_drops_trend_and_yearly_terms(port):
    y = port.y[1]
    X = B.regression_design(port.dow, port.holiday, port.after, port.promo[1])
    org = np.array([730])
    cols = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14]
    assert X.shape[1] == 15
    short = B.regression_forecast(y, X, org, 5, hist=182)
    beta = _ridge(X[C.FIRST_TEST - 182:C.FIRST_TEST][:, cols], np.log(np.maximum(y[C.FIRST_TEST - 182:C.FIRST_TEST], 1.0)))
    assert np.allclose(short[0], np.exp(X[730:735][:, cols] @ beta))
    assert np.array_equal(B.regression_forecast(y, X, org, 5, hist=730), B.regression_forecast(y, X, org, 5))


def test_hw_with_full_history_equals_default(port):
    org = np.array([730, 740])
    a = B.hw_forecast(port.y[0], org, 7)
    assert a.shape == (2, 7) and np.allclose(a, B.hw_forecast(port.y[0], org, 7, hist=730)) and (a > 0).all()


def test_mase_scale_by_hand():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 3, 2, 3, 4, 5, 6, 9, 9.0])
    assert B.mase_scale(y, first=14) == pytest.approx(np.mean(np.abs(y[7:14] - y[0:7])))


def test_scenario_is_deterministic_and_seeded():
    a, b, c = S.generate(5, seed=1), S.generate(5, seed=1), S.generate(5, seed=2)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (5, C.N_DAYS)


def test_scenario_structure(port):
    assert (port.y >= 0).all() and (port.mu >= 1.0).all() and port.holiday.shape == (C.N_DAYS,)
    assert port.holiday.sum() == len(C.HOLIDAY_DOY) * 3 and set(np.unique(port.promo)) <= {0.0, 1.0}
    assert not np.array_equal(port.promo[0], port.promo[1])
    assert all(C.PROMO_LENGTH <= port.promo[i].sum() <= 9 * C.PROMO_LENGTH for i in range(port.n))
    assert len(set(np.round(port.level))) == port.n


def test_trend_parameter_shifts_growth():
    lo, hi = S.generate(30, trend_mean=-20.0, seed=0), S.generate(30, trend_mean=40.0, seed=0)

    def growth(p):
        return float(np.mean(p.mu[:, 730:].sum(axis=1) / p.mu[:, :365].sum(axis=1)))

    assert growth(hi) > growth(lo) + 0.4


def test_summarize_by_hand():
    actual = np.array([[[4.0, 6.0], [5.0, 5.0]]])
    fc = {"a": np.array([[[5.0, 5.0], [5.0, 8.0]]])}
    out, per, hor = E.summarize(fc, actual, np.array([2.0]))
    assert out["a"]["mae"] == pytest.approx(1.25) and out["a"]["mase"] == pytest.approx(0.625) and hor["a"] == pytest.approx([0.25, 1.0])


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(n_depots=10, horizon=7, rounds=30))


def test_analysis_shapes(analysis):
    a = analysis
    n_org = C.N_DAYS - 7 - C.FIRST_TEST + 1
    assert a.actual.shape == (10, n_org, 7) and all(f.shape == a.actual.shape for f in a.forecasts.values()) and len(a.origins) == n_org
    assert set(a.summary) == {"gbm", "hw_mult", "regression", "snaive_k", "oracle"} and len(a.curve_val) == 30 == len(a.curve_train)
    assert len(a.names) == a.model.n_features and (a.forecasts["gbm"] >= 0).all()


def test_oracle_is_the_best_forecast_and_actuals_match_the_series(analysis):
    a = analysis
    assert min(a.summary, key=lambda m: a.summary[m]["mase"]) == "oracle"
    t = a.origins[3]
    assert np.array_equal(a.actual[2, 3], a.port.y[2, t:t + 7]) and np.array_equal(a.forecasts["oracle"][2, 3], a.port.mu[2, t:t + 7])


def test_mase_is_mae_over_the_scale(analysis):
    a = analysis
    per = np.abs(a.forecasts["gbm"] - a.actual).mean(axis=(1, 2)) / a.scale
    assert np.allclose(per, a.depot_mase["gbm"]) and a.summary["gbm"]["mase"] == pytest.approx(per.mean())


def test_horizon_one_and_model_reuse():
    s = E.Settings(n_depots=10, horizon=1, rounds=10)
    a = E.analyse(s)
    assert a.actual.shape[2] == 1 and E._model(s.model_key) is E._model(s.model_key)


def test_gbm_forecast_ignores_the_future_of_the_series(analysis):
    a = analysis
    s = a.settings
    import dataclasses

    import bf_features as F
    import bf_gbm as G
    t = 800
    dep, org, hor = [3] * 7, [t] * 7, np.arange(1, 8)
    y2 = a.port.y.copy()
    y2[:, t:] = 0
    port2 = dataclasses.replace(a.port, y=y2)
    X1, _, l1 = F.build(a.port, dep, org, hor, s.groups, s.norm)
    X2, _, l2 = F.build(port2, dep, org, hor, s.groups, s.norm)
    assert np.allclose(F.to_orders(G.predict(a.model, X1), l1), F.to_orders(G.predict(a.model, X2), l2))
    i = t - a.origins[0]
    assert np.allclose(a.forecasts["gbm"][3, i], F.to_orders(G.predict(a.model, X1), l1))
