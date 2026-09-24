"""Zeilen, Merkmale und Ziel des globalen Modells: von Hand nachgerechnet, ohne Blick in die Zukunft."""

import dataclasses

import numpy as np
import pytest

import bf_constants as C
import bf_features as F
import bf_scenario as S


@pytest.fixture(scope="module")
def port():
    return S.generate(3, 0.14, 0.5, 10.0, 5)


def test_row_by_hand_normalised(port):
    dep, org, hor = 1, 300, 3
    s = org + hor - 1
    X, target, lvl = F.build(port, [dep], [org], [hor])
    y = port.y[dep]
    l28 = y[org - 28:org].mean()
    l91 = y[org - 91:org].mean()
    scale = l28 + 1
    row = X[0]
    names = F.feature_names()
    assert X.shape[1] == len(names)
    assert np.allclose(row[:7], [y[org - k] / scale for k in range(1, 8)])
    assert np.allclose(row[7:11], [y[s - 7 * (1 + i)] / scale for i in range(4)])
    assert row[11] == pytest.approx(y[org - 7:org].mean() / scale) and row[12] == pytest.approx(l28 / (l91 + 1)) and row[13] == pytest.approx(y[org - 28:org].std() / scale)
    g = lambda name: row[names.index(name)]
    assert g("Wochentag") == port.dow[s] and g("Jahrestag (sin)") == pytest.approx(np.sin(2 * np.pi * (s % 365) / 365)) and g("Horizont") == hor
    assert g("Aktion am Zieltag") == port.promo[dep, s] and g("Aktionen der letzten 7 Tage") == pytest.approx(port.promo[dep, org - 7:org].mean())
    assert target[0] == pytest.approx(np.log((y[s] + 1) / scale)) and lvl[0] == pytest.approx(l28)


@pytest.mark.parametrize("hor,k0", [(1, 1), (7, 1), (8, 2), (14, 2), (15, 3), (28, 4)])
def test_weekday_lags_stay_before_the_origin_and_share_the_weekday(port, hor, k0):
    org = 400
    X, _, _ = F.build(port, [0], [org], [hor], groups=("lags",), normalize=False)
    s = org + hor - 1
    assert np.allclose(X[0, 7:11], [port.y[0, s - 7 * (k0 + i)] for i in range(4)])
    assert s - 7 * k0 < org <= s - 7 * (k0 - 1)


def test_raw_target_adds_the_level_as_a_feature(port):
    X, target, lvl = F.build(port, [2], [500], [5], normalize=False)
    assert X.shape[1] == len(F.feature_names(normalize=False)) and X[0, -1] == pytest.approx(port.y[2, 472:500].mean())
    assert target[0] == port.y[2, 504] and lvl[0] == X[0, -1]


def test_seasonal_scale_is_the_weekday_mean(port):
    _, target, lvl = F.build(port, [0], [600], [4], normalize="seasonal")
    s = 603
    wl = np.mean([port.y[0, s - 7 * (1 + i)] for i in range(4)])
    assert target[0] == pytest.approx(np.log((port.y[0, s] + 1) / (wl + 1))) and lvl[0] == pytest.approx(wl)


@pytest.mark.parametrize("normalize", [True, False, "seasonal"])
def test_to_orders_inverts_the_target(port, normalize):
    dep, org, hor = np.array([0, 1, 2]), np.array([200, 400, 700]), np.array([1, 9, 14])
    _, target, lvl = F.build(port, dep, org, hor, normalize=normalize)
    assert np.allclose(F.to_orders(target, lvl, normalize), port.y[dep, org + hor - 1])


def test_features_do_not_see_the_future(port):
    t = 300
    X1, _, l1 = F.build(port, [1, 1], [t, t], [6, 14])
    y2 = port.y.copy()
    y2[:, t:] = np.random.default_rng(0).integers(0, 999, size=y2[:, t:].shape)
    X2, _, l2 = F.build(dataclasses.replace(port, y=y2), [1, 1], [t, t], [6, 14])
    assert np.array_equal(X1, X2) and np.array_equal(l1, l2)


def test_training_rows_are_before_the_cut_and_within_the_horizon(port):
    d, o, h = F.training_rows(port, np.arange(3), 14, stride=4, per_origin=2, last_target=670, seed=1)
    assert o.min() == F.FIRST_ORIGIN and (o + h - 1).max() < 670 and h.min() >= 1 and h.max() <= 14
    n_org = len(np.arange(F.FIRST_ORIGIN, 670 - 14 + 1, 4))
    assert len(d) == 3 * n_org * 2 and set(d) == {0, 1, 2}


def test_test_rows_cover_every_origin_and_horizon(port):
    d, o, h, org = F.test_rows(port, np.arange(2), 5)
    assert org[0] == C.FIRST_TEST and org[-1] + 5 == C.N_DAYS and len(d) == 2 * len(org) * 5
    assert np.array_equal(h[:10], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) and np.array_equal(o[:6], [730] * 5 + [731]) and d[len(d) // 2] == 1
