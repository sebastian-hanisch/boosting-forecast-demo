"""Vehikel "Tagesaufträge mehrerer Depots": jedes Depot wie in den Vorgängern (Niveau, Trend, Wochenmuster, Jahresmuster, Feiertage, Aktionen, multiplikatives log-normales Rauschen), aber mit eigenen Parametern.
Der Kalender (Feiertage) ist für alle Depots derselbe, die Aktionstage sind je Depot verschieden. Der Erwartungswert je Tag (Orakel) steht mit im Ergebnis."""

from dataclasses import dataclass

import numpy as np

import bf_constants as C


@dataclass(frozen=True)
class Portfolio:
    y: np.ndarray             # (n, T) Tagesaufträge (ganze Zahlen)
    mu: np.ndarray            # (n, T) Erwartungswert ohne Rauschen
    holiday: np.ndarray       # (T,) 1 am Feiertag (für alle Depots)
    after: np.ndarray         # (T,) 1 am Tag nach dem Feiertag
    promo: np.ndarray         # (n, T) 1 an Aktionstagen
    dow: np.ndarray           # (T,) Wochentag, 0 = Montag
    level: np.ndarray         # (n,) Ausgangsniveau
    noise: np.ndarray         # (n,) Streuung des Rauschens
    weekly: np.ndarray        # (n,) Stärke des Wochenmusters
    yearly: np.ndarray        # (n,) Amplitude des Jahresmusters
    trend: np.ndarray         # (n,) Trend in Prozent je Jahr
    seed: int

    @property
    def n(self):
        return self.y.shape[0]


def calendar(n_days=C.N_DAYS):
    t = np.arange(n_days)
    holiday = np.isin(t % 365, C.HOLIDAY_DOY).astype(float)
    after = np.roll(holiday, 1)
    after[0] = 0.0
    return t % 7, holiday, after


def generate(n_depots=40, noise_mean=0.14, events=0.5, trend_mean=10.0, seed=0, n_days=C.N_DAYS):
    """noise_mean: mittlere Streuung des Rauschens; events: Stärke von Feiertagen und Aktionen; trend_mean: mittlerer Trend in Prozent je Jahr. Die Depots streuen um diese Werte."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dow, holiday, after = calendar(n_days)
    pattern = np.array(C.WEEKLY_PATTERN)
    pattern = pattern / pattern.mean()
    level = np.clip(C.LEVEL * np.exp(0.6 * rng.normal(size=n_depots)), 30.0, 500.0)
    noise = np.clip(noise_mean * np.exp(0.3 * rng.normal(size=n_depots)), 0.03, 0.6)
    weekly = rng.uniform(0.5, 1.5, size=n_depots)
    yearly = rng.uniform(0.0, 0.4, size=n_depots)
    trend = trend_mean + 10.0 * rng.normal(size=n_depots)
    y = np.zeros((n_depots, n_days))
    mu = np.zeros((n_depots, n_days))
    promo = np.zeros((n_depots, n_days))
    for i in range(n_depots):
        week_f = 1.0 + weekly[i] * (pattern[dow] - 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        year_f = 1.0 + yearly[i] * np.sin(2 * np.pi * (t % 365) / 365.0 + phase)
        trend_f = 1.0 + (trend[i] / 100.0) * t / 365.0
        holiday_f = 1.0 - events * C.HOLIDAY_DROP * holiday + events * C.HOLIDAY_REBOUND * after
        starts = rng.choice(np.arange(30, n_days - C.PROMO_LENGTH), size=C.PROMO_PER_YEAR * (n_days // 365), replace=False)
        for s in starts:
            promo[i, s:s + C.PROMO_LENGTH] = 1.0
        promo_f = 1.0 + events * 0.5 * promo[i]
        mu[i] = np.maximum(level[i] * np.maximum(trend_f, 0.05) * week_f * year_f * holiday_f * promo_f, 1.0)
        z = rng.normal(size=n_days)
        y[i] = np.maximum(np.rint(mu[i] * np.exp(noise[i] * z - 0.5 * noise[i] ** 2)), 0.0)
    return Portfolio(y, mu, holiday, after, promo, dow, level, noise, weekly, yearly, trend, int(seed))
