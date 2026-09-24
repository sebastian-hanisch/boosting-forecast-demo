"""Baumkern und Boosting gegen Handrechnung und gegen sklearn (nur als Gegenprobe in requirements-dev)."""

import numpy as np
import pytest

import bf_gbm as G
import bf_tree as T


def _grow(x, grad, **kw):
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    edges = T.build_bin_edges(X, 63)
    return T.grow(X, np.asarray(grad, dtype=float), np.ones(len(x)), edges, num_leaves=kw.pop("num_leaves", 2), min_child_samples=1, **kw), X


def test_one_split_by_hand():
    # Gradienten -1,-1,+1,+1: Teilung zwischen 2 und 3, GL=-2, HL=2, GR=2, HR=2, G=0, H=4, lambda=1
    tree, X = _grow([1, 2, 3, 4], [-1, -1, 1, 1])
    assert tree.n_leaves == 2 and tree.feature[0] == 0
    assert tree.gain[0] == pytest.approx(0.5 * (4 / 3 + 4 / 3 - 0.0))
    assert np.allclose(T.predict_value(tree, X), [2 / 3, 2 / 3, -2 / 3, -2 / 3])


def test_boosting_round_by_hand():
    # y = 0,0,4,4: F0 = 2, Residuen (F - y) = 2,2,-2,-2, Blatt links -4/(2+1), rechts +4/(2+1); Lernrate 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    ens = G.fit(X, [0, 0, 4, 4], num_leaves=2, min_child_samples=1, n_rounds=1, learning_rate=1.0)
    assert ens.f0 == pytest.approx(2.0)
    assert np.allclose(G.predict(ens, X), [2 - 4 / 3, 2 - 4 / 3, 2 + 4 / 3, 2 + 4 / 3])


def test_squared_error_falls_with_every_round():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=600)
    ens = G.fit(X, y, num_leaves=7, min_child_samples=10, n_rounds=30, learning_rate=0.2)
    mse = [np.mean((G.predict(ens, X, upto=k) - y) ** 2) for k in range(1, 31)]
    assert all(b <= a + 1e-12 for a, b in zip(mse, mse[1:])) and mse[-1] < 0.5 * mse[0]


def test_importances_sum_to_one_and_find_the_relevant_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = 3.0 * X[:, 1] + 0.05 * rng.normal(size=500)
    imp = G.importances(G.fit(X, y, num_leaves=7, min_child_samples=10, n_rounds=20))
    assert imp.sum() == pytest.approx(1.0) and int(np.argmax(imp)) == 1 and imp[1] > 0.9


def test_rare_binary_features_can_be_split():
    # Ein Merkmal, das nur in 3 % der Zeilen 1 ist, muss trotzdem teilbar sein (Bin-Grenzen mit side="left")
    rng = np.random.default_rng(2)
    flag = (rng.random(800) < 0.03).astype(float)
    X = np.stack([flag, rng.normal(size=800)], axis=1)
    y = 5.0 * flag + 0.1 * rng.normal(size=800)
    ens = G.fit(X, y, num_leaves=4, min_child_samples=5, n_rounds=20, learning_rate=0.3)
    p = G.predict(ens, X)
    assert p[flag == 1].mean() - p[flag == 0].mean() > 4.0


def test_validation_curves_are_returned_with_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=300)
    ens, tr, va = G.fit(X[:200], y[:200], n_rounds=15, min_child_samples=5, X_val=X[200:], y_val=y[200:])
    assert len(ens.trees) == 15 and tr.shape == (15,) and va.shape == (15,) and tr[-1] < tr[0] and va[-1] < va[0]


def test_matches_sklearn_histogram_boosting():
    sk = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.2 * rng.normal(size=2000)
    Xt = rng.normal(size=(2000, 5))
    yt = np.sin(2 * Xt[:, 0]) + Xt[:, 1] * Xt[:, 2] + 0.2 * rng.normal(size=2000)
    ours = G.predict(G.fit(X, y, num_leaves=15, min_child_samples=20, n_rounds=60, learning_rate=0.1), Xt)
    ref = sk.HistGradientBoostingRegressor(max_leaf_nodes=15, min_samples_leaf=20, max_iter=60, learning_rate=0.1, l2_regularization=1.0, max_bins=63, early_stopping=False, random_state=0).fit(X, y).predict(Xt)
    mae_o, mae_r = np.abs(ours - yt).mean(), np.abs(ref - yt).mean()
    assert abs(mae_o - mae_r) / mae_r < 0.06
