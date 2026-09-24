"""Konstanten der Boosting-Demo: Vehikel "Tagesaufträge mehrerer Depots" (Stück 6 der Zeitreihen-Prognose-Linie), Merkmale, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
SEASON_PERIOD = 7

# --- Glättung aus Stück 2 (Vergleich) ------------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98
DEFAULT_TREND = 10
DEFAULT_NOISE = 0.14
DEFAULT_EVENTS = 0.5

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------
DEPOTS_MIN, DEPOTS_MAX, DEPOTS_STEP, DEFAULT_DEPOTS = 10, 100, 10, 40
NOISE_MIN, NOISE_MAX, NOISE_STEP = 0.04, 0.4, 0.02
TREND_MIN, TREND_MAX, TREND_STEP = -20, 40, 5
EVENTS_MIN, EVENTS_MAX, EVENTS_STEP = 0.0, 1.0, 0.25
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
ROUNDS_MIN, ROUNDS_MAX, ROUNDS_STEP, DEFAULT_ROUNDS = 10, 200, 10, 80
LEAVES_MIN, LEAVES_MAX, DEFAULT_LEAVES = 3, 63, 15
LR_MIN, LR_MAX, LR_STEP, DEFAULT_LR = 0.02, 0.5, 0.02, 0.1
MINLEAF_MIN, MINLEAF_MAX, MINLEAF_STEP, DEFAULT_MIN_LEAF = 5, 200, 5, 20
NORMALIZE = ("level", "seasonal", "none")
NORMALIZE_LABELS = {"level": "Verhältnis zum 28-Tage-Niveau (log)", "seasonal": "Verhältnis zum Wochentagsmittel der letzten vier Wochen (log)", "none": "roher Wert"}

METHODS = ("gbm", "hw_mult", "regression", "snaive_k")
METHOD_NAMES = {"gbm": "Globales Boosting", "hw_mult": "Holt-Winters multiplikativ (Stück 2, je Depot)", "regression": "Regression auf Kalender und Aktionsplan (Stück 4, je Depot)", "snaive_k": "Wochenmittel (Stück 1, je Depot)"}

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_DEPOTS = 30
TREND_DEPOTS = 20
EXP_TEST = 20
EXP_TRAIN = 30
SERIES_COUNTS = (1, 3, 10, 30)
HISTORIES = (91, 182, 365, 730)
TREND_LEVELS = (0, 20, 40)
