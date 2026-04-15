# app/config.py
# All tunable constants for GASPmini v1.
# Profile changes update the active module-level constants at runtime.

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SimulationProfile:
	label: str
	grid_width: int
	grid_height: int
	population_size: int
	initial_gene_count: int
	food_count: int
	interior_wall_count: int
	ticks_per_epoch: int
	elite_fraction: float
	offspring_fraction: float
	random_new_fraction: float
	initial_energy: int
	energy_gain_from_food: int
	energy_loss_per_tick: int
	history_length: int
	default_learning_rate: float
	default_reward_decay: float
	default_exploration_rate: float
	reward_eat_food: float
	reward_failed_move: float
	reward_idle: float
	reward_death: float
	reward_starvation_tick: float
	fitness_food_weight: float
	fitness_age_weight: float
	fitness_stationary_age_factor: float
	fitness_failed_weight: float
	gene_match_score: float
	gene_wildcard_score: float
	gene_mismatch_score: float
	mutation_rate: float
	param_mutation_rate: float
	param_mutation_delta: float
	learned_priority_imprint_factor: float
	debug_sensors: bool = False
	debug_gene_scoring: bool = False
	debug_actions: bool = False
	debug_rewards: bool = False
	debug_evolution: bool = False


DEFAULT_SEED = 42

PROFILE_DEFINITIONS: dict[str, SimulationProfile] = {
	'short_chaotic': SimulationProfile(
		label='Short chaotic runs',
		grid_width=30,
		grid_height=20,
		population_size=20,
		initial_gene_count=12,
		food_count=25,
		interior_wall_count=32,
		ticks_per_epoch=160,
		elite_fraction=0.15,
		offspring_fraction=0.70,
		random_new_fraction=0.15,
		initial_energy=28,
		energy_gain_from_food=10,
		energy_loss_per_tick=1,
		history_length=6,
		default_learning_rate=0.25,
		default_reward_decay=0.68,
		default_exploration_rate=0.08,
		reward_eat_food=10.0,
		reward_failed_move=-1.0,
		reward_idle=-0.2,
		reward_death=-5.0,
		reward_starvation_tick=-0.5,
		fitness_food_weight=3.0,
		fitness_age_weight=0.05,
		fitness_stationary_age_factor=0.2,
		fitness_failed_weight=-0.2,
		gene_match_score=2.0,
		gene_wildcard_score=0.0,
		gene_mismatch_score=-3.0,
		mutation_rate=0.30,
		param_mutation_rate=0.18,
		param_mutation_delta=0.05,
		learned_priority_imprint_factor=0.15,
	),
	'longer_strategic': SimulationProfile(
		label='Longer strategic',
		grid_width=36,
		grid_height=24,
		population_size=24,
		initial_gene_count=12,
		food_count=48,
		interior_wall_count=20,
		ticks_per_epoch=320,
		elite_fraction=0.20,
		offspring_fraction=0.75,
		random_new_fraction=0.05,
		initial_energy=45,
		energy_gain_from_food=12,
		energy_loss_per_tick=1,
		history_length=8,
		default_learning_rate=0.20,
		default_reward_decay=0.80,
		default_exploration_rate=0.03,
		reward_eat_food=10.0,
		reward_failed_move=-1.0,
		reward_idle=-0.1,
		reward_death=-5.0,
		reward_starvation_tick=-0.5,
		fitness_food_weight=3.0,
		fitness_age_weight=0.06,
		fitness_stationary_age_factor=0.2,
		fitness_failed_weight=-0.15,
		gene_match_score=2.0,
		gene_wildcard_score=0.0,
		gene_mismatch_score=-3.0,
		mutation_rate=0.18,
		param_mutation_rate=0.10,
		param_mutation_delta=0.04,
		learned_priority_imprint_factor=0.12,
	),
	'fast_evo_less_starvation': SimulationProfile(
		label='Faster evo signal less starvation',
		grid_width=30,
		grid_height=20,
		population_size=24,
		initial_gene_count=12,
		food_count=50,
		interior_wall_count=18,
		ticks_per_epoch=220,
		elite_fraction=0.25,
		offspring_fraction=0.70,
		random_new_fraction=0.05,
		initial_energy=40,
		energy_gain_from_food=12,
		energy_loss_per_tick=1,
		history_length=6,
		default_learning_rate=0.25,
		default_reward_decay=0.72,
		default_exploration_rate=0.05,
		reward_eat_food=10.0,
		reward_failed_move=-1.0,
		reward_idle=-0.15,
		reward_death=-5.0,
		reward_starvation_tick=-0.5,
		fitness_food_weight=3.5,
		fitness_age_weight=0.04,
		fitness_stationary_age_factor=0.2,
		fitness_failed_weight=-0.15,
		gene_match_score=2.0,
		gene_wildcard_score=0.0,
		gene_mismatch_score=-3.0,
		mutation_rate=0.22,
		param_mutation_rate=0.12,
		param_mutation_delta=0.05,
		learned_priority_imprint_factor=0.15,
	),
}

DEFAULT_PROFILE_ID = 'short_chaotic'
ACTIVE_PROFILE_ID = DEFAULT_PROFILE_ID


def get_profile_items() -> list[tuple[str, str]]:
	return [(profile_id, profile.label) for profile_id, profile in PROFILE_DEFINITIONS.items()]


def get_active_profile() -> SimulationProfile:
	return PROFILE_DEFINITIONS[ACTIVE_PROFILE_ID]


def apply_profile(profile_id: str) -> SimulationProfile:
	global ACTIVE_PROFILE_ID

	if profile_id not in PROFILE_DEFINITIONS:
		raise KeyError(f'Unknown profile: {profile_id}')

	profile = PROFILE_DEFINITIONS[profile_id]
	ACTIVE_PROFILE_ID = profile_id

	values = asdict(profile)
	globals().update({
		'GRID_WIDTH': values['grid_width'],
		'GRID_HEIGHT': values['grid_height'],
		'POPULATION_SIZE': values['population_size'],
		'INITIAL_GENE_COUNT': values['initial_gene_count'],
		'FOOD_COUNT': values['food_count'],
		'INTERIOR_WALL_COUNT': values['interior_wall_count'],
		'TICKS_PER_EPOCH': values['ticks_per_epoch'],
		'ELITE_FRACTION': values['elite_fraction'],
		'OFFSPRING_FRACTION': values['offspring_fraction'],
		'RANDOM_NEW_FRACTION': values['random_new_fraction'],
		'INITIAL_ENERGY': values['initial_energy'],
		'ENERGY_GAIN_FROM_FOOD': values['energy_gain_from_food'],
		'ENERGY_LOSS_PER_TICK': values['energy_loss_per_tick'],
		'HISTORY_LENGTH': values['history_length'],
		'DEFAULT_LEARNING_RATE': values['default_learning_rate'],
		'DEFAULT_REWARD_DECAY': values['default_reward_decay'],
		'DEFAULT_EXPLORATION_RATE': values['default_exploration_rate'],
		'REWARD_EAT_FOOD': values['reward_eat_food'],
		'REWARD_FAILED_MOVE': values['reward_failed_move'],
		'REWARD_IDLE': values['reward_idle'],
		'REWARD_DEATH': values['reward_death'],
		'REWARD_STARVATION_TICK': values['reward_starvation_tick'],
		'FITNESS_FOOD_WEIGHT': values['fitness_food_weight'],
		'FITNESS_AGE_WEIGHT': values['fitness_age_weight'],
		'FITNESS_STATIONARY_AGE_FACTOR': values['fitness_stationary_age_factor'],
		'FITNESS_FAILED_WEIGHT': values['fitness_failed_weight'],
		'GENE_MATCH_SCORE': values['gene_match_score'],
		'GENE_WILDCARD_SCORE': values['gene_wildcard_score'],
		'GENE_MISMATCH_SCORE': values['gene_mismatch_score'],
		'MUTATION_RATE': values['mutation_rate'],
		'PARAM_MUTATION_RATE': values['param_mutation_rate'],
		'PARAM_MUTATION_DELTA': values['param_mutation_delta'],
		'LEARNED_PRIORITY_IMPRINT_FACTOR': values['learned_priority_imprint_factor'],
		'DEBUG_SENSORS': values['debug_sensors'],
		'DEBUG_GENE_SCORING': values['debug_gene_scoring'],
		'DEBUG_ACTIONS': values['debug_actions'],
		'DEBUG_REWARDS': values['debug_rewards'],
		'DEBUG_EVOLUTION': values['debug_evolution'],
	})
	return profile


apply_profile(DEFAULT_PROFILE_ID)
