"""Auswertung: ein globales Boosting-Modell (ein Modell für alle Depots, Lag- und Kalendermerkmale) im Rolling-Origin-Vergleich mit den lokalen Verfahren der Vorgänger und vier Experimente (Merkmale, Zahl der Reihen,
kurze Historie, Trend und Normierung).

Ein Ursprung t heißt: bekannt sind die Tage 0..t-1 (und der Kalender samt Aktionsplan), prognostiziert werden die Tage t..t+h-1. Das Modell lernt aus den Trainingstagen aller Depots (Zieltage vor Tag 730 - 60; die letzten 60 Trainingstage sind
die Validierung für die Lernkurve); alle lokalen Verfahren schätzen ihre Parameter je Depot auf den Trainingstagen. Kennzahl: MASE je Depot (Nenner: saisonal naiver Fehler auf den Trainingstagen), gemittelt über die Depots."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bf_baselines as B
import bf_constants as C
import bf_features as F
import bf_gbm as G
import bf_scenario as S

VAL_DAYS = 60


@dataclass(frozen=True)
class Settings:
    n_depots: int = C.DEFAULT_DEPOTS
    noise: float = C.DEFAULT_NOISE
    events: float = C.DEFAULT_EVENTS
    trend: int = C.DEFAULT_TREND
    horizon: int = C.DEFAULT_HORIZON
    rounds: int = C.DEFAULT_ROUNDS
    leaves: int = C.DEFAULT_LEAVES
    lr: float = C.DEFAULT_LR
    min_leaf: int = C.DEFAULT_MIN_LEAF
    lags: bool = True
    calendar: bool = True
    promo: bool = True
    normalize: str = "level"
    seed: int = 3

    @property
    def portfolio_key(self):
        return (self.n_depots, self.noise, self.events, self.trend, self.seed)

    @property
    def groups(self):
        return tuple(g for g, on in (("lags", self.lags), ("calendar", self.calendar), ("promo", self.promo)) if on)

    @property
    def norm(self):
        return {"level": True, "seasonal": "seasonal", "none": False}[self.normalize]

    @property
    def model_key(self):
        return self.portfolio_key + (self.horizon, self.rounds, self.leaves, self.lr, self.min_leaf, self.groups, self.normalize)


@dataclass
class Analysis:
    settings: Settings
    port: S.Portfolio
    model: G.Ensemble
    curve_train: np.ndarray    # mittlerer absoluter Fehler (Log-Skala) auf den Trainingszeilen nach jeder Runde
    curve_val: np.ndarray      # ... auf den Validierungszeilen
    names: list
    origins: np.ndarray
    actual: np.ndarray         # (n, n_org, h)
    forecasts: dict            # Verfahren -> (n, n_org, h)
    scale: np.ndarray          # (n,) MASE-Nenner
    summary: dict              # Verfahren -> {"mase": ..., "mae": ...} gemittelt über die Depots
    depot_mase: dict           # Verfahren -> (n,)
    horizon_mae: dict          # Verfahren -> (h,) mittlerer MAE je Horizont, gemittelt über die Depots (durch die Skala geteilt)

    @property
    def best(self):
        return min((m for m in self.summary if m != "oracle"), key=lambda m: self.summary[m]["mase"])


@lru_cache(maxsize=32)
def _portfolio(key):
    n, noise, events, trend, seed = key
    return S.generate(n, noise, events, float(trend), seed)


def _scales(port):
    return np.array([B.mase_scale(port.y[i]) for i in range(port.n)])


@lru_cache(maxsize=32)
def _baselines(key, horizon, hist=None):
    """Lokale Verfahren und Orakel für alle Depots und Ursprünge des Testjahres: {Verfahren: (n, n_org, h)} und die Ursprünge."""
    port = _portfolio(key)
    org = np.arange(C.FIRST_TEST, C.N_DAYS - horizon + 1)
    out = {"hw_mult": [], "regression": [], "snaive_k": [], "oracle": []}
    for i in range(port.n):
        y = port.y[i]
        X = B.regression_design(port.dow, port.holiday, port.after, port.promo[i])
        out["hw_mult"].append(B.hw_forecast(y, org, horizon, hist))
        out["regression"].append(B.regression_forecast(y, X, org, horizon, hist))
        out["snaive_k"].append(B.snaive_k(y, org, horizon))
        out["oracle"].append(np.stack([port.mu[i, t:t + horizon] for t in org]))
    return {k: np.stack(v) for k, v in out.items()}, org


def train_model(port, depots, s, seed=0):
    """Globales Modell auf den Zeilen der angegebenen Depots (Zieltage vor Tag 730 - VAL_DAYS); Rückgabe: Modell, Lernkurven (Training, Validierung), Merkmalsnamen."""
    h = s.horizon
    d, o, hor = F.training_rows(port, depots, h, stride=4, per_origin=2, last_target=C.FIRST_TEST - VAL_DAYS, seed=seed)
    X, y, _ = F.build(port, d, o, hor, s.groups, s.norm)
    vd, vo, vh = F.training_rows(port, depots, h, stride=4, per_origin=1, first=C.FIRST_TEST - VAL_DAYS - h + 1, last_target=C.FIRST_TEST, seed=seed + 1)
    Xv, yv, _ = F.build(port, vd, vo, vh, s.groups, s.norm)
    ens, curve_train, curve_val = G.fit(X, y, num_leaves=s.leaves, n_rounds=s.rounds, learning_rate=s.lr, min_child_samples=s.min_leaf, seed=seed, X_val=Xv, y_val=yv)
    return ens, curve_train, curve_val, F.feature_names(s.groups, s.norm)


def predict_depots(ens, port, depots, s, step=1):
    """Prognosen in Aufträgen (n_dep, n_org, h) für die Ursprünge des Testjahres."""
    td, to, th, org = F.test_rows(port, depots, s.horizon, step=step)
    X, _, lvl = F.build(port, td, to, th, s.groups, s.norm)
    pred = F.to_orders(G.predict(ens, X), lvl, s.norm)
    return pred.reshape(len(depots), len(org), s.horizon), org


@lru_cache(maxsize=16)
def _model(mkey):
    n, noise, events, trend, seed, horizon, rounds, leaves, lr, min_leaf, groups, normalize = mkey
    s = Settings(n, noise, events, trend, horizon, rounds, leaves, lr, min_leaf, "lags" in groups, "calendar" in groups, "promo" in groups, normalize, seed)
    port = _portfolio(s.portfolio_key)
    return train_model(port, np.arange(port.n), s)


def summarize(forecasts, actual, scale):
    out, per = {}, {}
    for m, f in forecasts.items():
        mae_dep = np.abs(f - actual).mean(axis=(1, 2))
        per[m] = mae_dep / scale
        out[m] = {"mase": float(per[m].mean()), "mae": float(mae_dep.mean())}
    hor = {m: (np.abs(f - actual).mean(axis=1) / scale[:, None]).mean(axis=0) for m, f in forecasts.items()}
    return out, per, hor


def analyse(s):
    port = _portfolio(s.portfolio_key)
    ens, ctr, cva, names = _model(s.model_key)
    base, org = _baselines(s.portfolio_key, s.horizon)
    gbm, _ = predict_depots(ens, port, np.arange(port.n), s)
    actual = np.stack([np.stack([port.y[i, t:t + s.horizon] for t in org]) for i in range(port.n)])
    fc = {"gbm": gbm, **base}
    scale = _scales(port)
    summary, per, hor = summarize(fc, actual, scale)
    return Analysis(s, port, ens, ctr, cva, names, org, actual, fc, scale, summary, per, hor)


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


# --- Experiment 1: Merkmale ------------------------------------------------------------------------------------------------------------------

FEATURE_SETS = (
    ("nur Lags", dict(lags=True, calendar=False, promo=False)),
    ("nur Kalender", dict(lags=False, calendar=True, promo=False)),
    ("Lags + Kalender", dict(lags=True, calendar=True, promo=False)),
    ("Lags + Kalender + Aktionsplan", dict(lags=True, calendar=True, promo=True)),
)


def features_experiment(seeds=None, base=None):
    """MASE des Modells mit verschiedenen Merkmalsgruppen (Mittel über die Seeds) neben den lokalen Verfahren; dazu die Wichtigkeit der Merkmale des vollen Modells."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    res = {name: [] for name, _ in FEATURE_SETS}
    refs = {m: [] for m in ("hw_mult", "regression", "snaive_k", "oracle")}
    imp = []
    for sd in seeds:
        for name, kw in FEATURE_SETS:
            a = analyse(_replace(base, seed=sd, **kw))
            res[name].append(a.summary["gbm"]["mase"])
        for m in refs:
            refs[m].append(a.summary[m]["mase"])
        imp.append(G.importances(a.model))
    names = a.names
    return {"n_seeds": len(seeds), "rows": [{"name": n, **dict(zip(("mase", "se"), _mean_se(v)))} for n, v in res.items()], "refs": {m: _mean_se(v)[0] for m, v in refs.items()}, "names": names, "importance": np.mean(imp, axis=0)}


# --- Experiment 2: Zahl der Reihen ---------------------------------------------------------------------------------------------------------------


def series_experiment(counts=None, seeds=None, n_test=None):
    """Das Modell lernt aus n Depots und wird auf davon getrennten Test-Depots geprüft; daneben die lokalen Verfahren auf denselben Test-Depots (Mittel über die Seeds)."""
    counts = C.SERIES_COUNTS if counts is None else counts
    seeds = C.EXP_SEEDS if seeds is None else seeds
    n_test = C.EXP_TEST if n_test is None else n_test
    rows = {n: [] for n in counts}
    refs = {m: [] for m in ("hw_mult", "regression", "snaive_k")}
    for sd in seeds:
        total = max(counts) + n_test
        s = Settings(n_depots=total, seed=sd)
        port = _portfolio(s.portfolio_key)
        test = np.arange(max(counts), total)
        base, org = _baselines(s.portfolio_key, s.horizon)
        actual = np.stack([np.stack([port.y[i, t:t + s.horizon] for t in org]) for i in test])
        scale = _scales(port)[test]
        for n in counts:
            ens, _, _, _ = train_model(port, np.arange(n), s, seed=sd)
            pred, _ = predict_depots(ens, port, test, s)
            rows[n].append(float(np.mean(np.abs(pred - actual).mean(axis=(1, 2)) / scale)))
        for m in refs:
            refs[m].append(float(np.mean(np.abs(base[m][test] - actual).mean(axis=(1, 2)) / scale)))
    return {"n_seeds": len(seeds), "n_test": n_test, "counts": tuple(counts), "rows": {n: _mean_se(v) for n, v in rows.items()}, "refs": {m: _mean_se(v)[0] for m, v in refs.items()}}


# --- Experiment 3: kurze Historie -----------------------------------------------------------------------------------------------------------------


def history_experiment(histories=None, seeds=None, n_train=None, n_test=None):
    """Neue Depots mit nur hist Tagen Historie vor dem Testjahr: die lokalen Verfahren schätzen ihre Parameter auf diesen Tagen, das Modell kommt aus anderen Depots (Mittel über die Seeds)."""
    histories = C.HISTORIES if histories is None else histories
    seeds = C.EXP_SEEDS if seeds is None else seeds
    n_train = C.EXP_TRAIN if n_train is None else n_train
    n_test = C.EXP_TEST if n_test is None else n_test
    gb = []
    loc = {(m, h): [] for m in ("hw_mult", "regression") for h in histories}
    wm = []
    for sd in seeds:
        s = Settings(n_depots=n_train + n_test, seed=sd)
        port = _portfolio(s.portfolio_key)
        test = np.arange(n_train, n_train + n_test)
        org = np.arange(C.FIRST_TEST, C.N_DAYS - s.horizon + 1)
        actual = np.stack([np.stack([port.y[i, t:t + s.horizon] for t in org]) for i in test])
        scale = _scales(port)[test]
        ens, _, _, _ = train_model(port, np.arange(n_train), s, seed=sd)
        pred, _ = predict_depots(ens, port, test, s)
        gb.append(float(np.mean(np.abs(pred - actual).mean(axis=(1, 2)) / scale)))
        wm.append(float(np.mean(np.abs(np.stack([B.snaive_k(port.y[i], org, s.horizon) for i in test]) - actual).mean(axis=(1, 2)) / scale)))
        for h in histories:
            base, _ = _baselines(s.portfolio_key, s.horizon, h)
            for m in ("hw_mult", "regression"):
                loc[(m, h)].append(float(np.mean(np.abs(base[m][test] - actual).mean(axis=(1, 2)) / scale)))
    return {"n_seeds": len(seeds), "histories": tuple(histories), "gbm": _mean_se(gb), "snaive_k": _mean_se(wm)[0], "local": {k: _mean_se(v) for k, v in loc.items()}}


# --- Experiment 4: Trend und Normierung -------------------------------------------------------------------------------------------------------------


def trend_experiment(trends=None, seeds=None, base=None):
    """Zunehmender Trend: das Modell mit normiertem Ziel (Verhältnis zum jüngsten Niveau) und mit rohem Ziel gegen Holt-Winters und die Regression (Mittel über die Seeds)."""
    trends = C.TREND_LEVELS if trends is None else trends
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.TREND_DEPOTS) if base is None else base
    rows = []
    for tr in trends:
        res = {m: [] for m in ("gbm_level", "gbm_none", "hw_mult", "regression")}
        for sd in seeds:
            for name in ("level", "none"):
                a = analyse(_replace(base, trend=tr, seed=sd, normalize=name))
                res["gbm_" + name].append(a.summary["gbm"]["mase"])
            res["hw_mult"].append(a.summary["hw_mult"]["mase"])
            res["regression"].append(a.summary["regression"]["mase"])
        rows.append({"trend": tr, "n_seeds": len(seeds), **{m: _mean_se(v) for m, v in res.items()}})
    return rows
