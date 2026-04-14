# app/config.py
# All tunable constants for GASPmini v1.
# Keep values small so the simulation is easy to inspect.

# ── Grid ─────────────────────────────────────────────────────────────────────
GRID_WIDTH = 30
GRID_HEIGHT = 20

# ── Population ────────────────────────────────────────────────────────────────
POPULATION_SIZE = 20
INITIAL_GENE_COUNT = 12
FOOD_COUNT = 25
INTERIOR_WALL_COUNT = 30
TICKS_PER_EPOCH = 200

# ── Evolution fractions ───────────────────────────────────────────────────────
ELITE_FRACTION = 0.15
OFFSPRING_FRACTION = 0.75
RANDOM_NEW_FRACTION = 0.10

# ── Creature defaults ─────────────────────────────────────────────────────────
INITIAL_ENERGY = 30
ENERGY_GAIN_FROM_FOOD = 10
ENERGY_LOSS_PER_TICK = 1
HISTORY_LENGTH = 6
DEFAULT_LEARNING_RATE = 0.25
DEFAULT_REWARD_DECAY = 0.7
DEFAULT_EXPLORATION_RATE = 0.05

# ── Rewards ───────────────────────────────────────────────────────────────────
REWARD_EAT_FOOD = 10.0
REWARD_FAILED_MOVE = -1.0
REWARD_FAILED_EAT = -1.0
REWARD_IDLE = -0.2
REWARD_DEATH = -5.0
REWARD_STARVATION_TICK = -0.5

# ── Fitness ───────────────────────────────────────────────────────────────────
FITNESS_FOOD_WEIGHT = 3.0
FITNESS_AGE_WEIGHT = 0.05
FITNESS_FAILED_WEIGHT = -0.2

# ── Gene scoring ──────────────────────────────────────────────────────────────
GENE_MATCH_SCORE = 2.0
GENE_WILDCARD_SCORE = 0.0
GENE_MISMATCH_SCORE = -3.0

# ── Gene mutation rates ───────────────────────────────────────────────────────
MUTATION_RATE = 0.25          # probability a gene is mutated during reproduction
PARAM_MUTATION_RATE = 0.15    # probability a genome-level param is mutated
PARAM_MUTATION_DELTA = 0.05   # max change per mutation for float params

# ── Random default seed ───────────────────────────────────────────────────────
DEFAULT_SEED = 42

# ── Debug toggles ─────────────────────────────────────────────────────────────
DEBUG_SENSORS = False
DEBUG_GENE_SCORING = False
DEBUG_ACTIONS = False
DEBUG_REWARDS = False
DEBUG_EVOLUTION = False
