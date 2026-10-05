"""Orakel für den Histogramm-Baumkern und das Boosting: (1) ohne Regularisierung gegen sklearn `DecisionTreeRegressor(max_leaf_nodes=...)` (ebenfalls blattweise, bester Gain zuerst) und ein Boosting aus
sklearn-Bäumen auf den Residuen; (2) mit lambda, gamma, min_child_* gegen einen eigenen, erschöpfenden blattweisen Wachser, der jede Bin-Grenze direkt aus den Zeilen bewertet (kein Histogramm, keine Differenz).
Die Bin-Grenzen sind so fein, dass jede Lücke zwischen zwei Werten eine Kante hat; damit sind die Schwellen gleichwertig."""

import numpy as np
import pytest

import bf_gbm as G
import bf_tree as T


def _brute_leafwise(X, grad, hess, edges, num_leaves, lam, gamma, mcw, mcs):
    n, d = X.shape

    def best(idx):
        Gs, Hs, b = grad[idx].sum(), hess[idx].sum(), None
        for f in range(d):
            for th in edges[f]:
                lm = X[idx, f] <= th
                if lm.sum() < mcs or (~lm).sum() < mcs:
                    continue
                GL, HL = grad[idx][lm].sum(), hess[idx][lm].sum()
                if HL < mcw or Hs - HL < mcw:
                    continue
                g = 0.5 * (GL ** 2 / (HL + lam) + (Gs - GL) ** 2 / (Hs - HL + lam) - Gs ** 2 / (Hs + lam)) - gamma
                if g > 0 and (b is None or g > b[0] + 1e-12):
                    b = (g, f, th)
        return b

    ok = lambda idx: len(idx) >= 2 * max(mcs, 1)
    leaves = [np.arange(n)]
    cands = [best(leaves[0]) if ok(leaves[0]) else None]
    while len(leaves) < num_leaves:
        valid = [i for i, c in enumerate(cands) if c is not None]
        if not valid:
            break
        i = max(valid, key=lambda j: cands[j][0])
        _, f, th = cands[i]
        idx = leaves[i]
        lm = X[idx, f] <= th
        leaves[i], cands[i] = idx[lm], (best(idx[lm]) if ok(idx[lm]) else None)
        leaves.append(idx[~lm]); cands.append(best(idx[~lm]) if ok(idx[~lm]) else None)
    pred = np.zeros(n)
    for idx in leaves:
        pred[idx] = -grad[idx].sum() / (hess[idx].sum() + lam)
    return pred


def test_unregularised_tree_equals_sklearn_best_first_tree():
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(1701)
    for _ in range(12):
        n, d, L = int(rng.integers(8, 100)), int(rng.integers(1, 4)), int(rng.integers(2, 10))
        X, grad = rng.normal(size=(n, d)), rng.normal(size=n)
        tree = T.grow(X, grad, np.ones(n), T.build_bin_edges(X, 20 * n), num_leaves=L, lam=0.0, min_child_weight=0.0, min_child_samples=1)
        ref = sk.DecisionTreeRegressor(max_leaf_nodes=L, random_state=0).fit(X, -grad).predict(X)
        assert T.predict_value(tree, X) == pytest.approx(ref, abs=1e-9)
        if tree.n_leaves > 1:   # Gain der Wurzel = halbe Verringerung der Fehlerquadratsumme
            lm = X[:, tree.feature[0]] <= tree.threshold[0]
            sse = lambda v: np.sum((v - v.mean()) ** 2)
            assert tree.gain[0] == pytest.approx(0.5 * (sse(grad) - sse(grad[lm]) - sse(grad[~lm])), rel=1e-9)


def test_boosting_equals_a_loop_of_sklearn_trees_on_the_residuals(monkeypatch):
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(55)
    fine = T.build_bin_edges
    monkeypatch.setattr(T, "build_bin_edges", lambda X, mb: fine(X, 40 * len(X)))
    for _ in range(5):
        n, d, L, R, lr = int(rng.integers(30, 150)), int(rng.integers(1, 4)), int(rng.integers(2, 9)), int(rng.integers(1, 10)), float(rng.choice([0.1, 0.5, 1.0]))
        X = rng.normal(size=(n, d))
        y = np.sin(X[:, 0]) + 0.3 * rng.normal(size=n) + (X[:, -1] > 0)
        ens = G.fit(X, y, num_leaves=L, lam=0.0, min_child_samples=1, n_rounds=R, learning_rate=lr)
        F = np.full(n, y.mean())
        for _r in range(R):
            F = F + lr * sk.DecisionTreeRegressor(max_leaf_nodes=L, random_state=0).fit(X, y - F).predict(X)
        assert G.predict(ens, X) == pytest.approx(F, abs=1e-7)


def test_regularised_tree_equals_an_exhaustive_leafwise_grower():
    rng = np.random.default_rng(1702)
    for k in range(20):
        n, d, L = int(rng.integers(10, 50)), int(rng.integers(1, 4)), int(rng.integers(2, 9))
        X, grad = rng.normal(size=(n, d)), rng.normal(size=n)
        hess = np.ones(n) if k % 2 == 0 else rng.uniform(0.5, 2.0, size=n)
        lam, gamma, mcs, mcw = float(rng.choice([0.0, 1.0, 5.0])), float(rng.choice([0.0, 0.05, 0.5])), int(rng.choice([1, 2, 4])), float(rng.choice([0.0, 1.0]))
        edges = T.build_bin_edges(X, 40 * n)
        tree = T.grow(X, grad, hess, edges, num_leaves=L, lam=lam, gamma=gamma, min_child_weight=mcw, min_child_samples=mcs)
        assert T.predict_value(tree, X) == pytest.approx(_brute_leafwise(X, grad, hess, edges, L, lam, gamma, mcw, mcs), abs=1e-9)
