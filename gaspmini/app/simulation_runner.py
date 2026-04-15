# app/simulation_runner.py
# High-level simulation controller used by the GUI.
# Owns world state, drives tick/epoch transitions.

from __future__ import annotations

import copy
import random
from pathlib import Path

import app.config as config
from app.models import WorldState, Creature, Genome, CreatureEpochResult
from app.genome_store import genome_file_exists, load_genome_from_file, save_genome_to_file
from app.world import generate_world, make_rng, make_random_genome
from app.simulation import tick_world
from app.evolution import collect_epoch_results, evolve_next_generation
from app.evolution import compute_fitness
from app.logging_utils import log


class SimulationRunner:
    def __init__(
        self,
        seed: int | None = None,
        ticks_per_epoch: int | None = None,
        profile_id: str | None = None,
        custom_map_id: str | None = None,
    ) -> None:
        self.profile_id = profile_id or config.ACTIVE_PROFILE_ID
        config.apply_profile(self.profile_id)
        self.custom_map_id = custom_map_id or None
        self.seed = config.DEFAULT_SEED if seed is None else seed
        self.ticks_per_epoch = config.TICKS_PER_EPOCH if ticks_per_epoch is None else ticks_per_epoch
        self.world: WorldState | None = None
        self._rng: random.Random = make_rng(self.seed)
        self.epoch_history: list[dict] = []  # brief per-epoch summary
        self.best_genome_ever: Genome | None = None
        self.best_fitness_ever: float | None = None
        self.best_epoch_ever: int | None = None
        self.mode: str = 'evolution'
        self.testing_ground_genome: Genome | None = None
        self._testing_ground_completed_logged = False
        self.autosave_best_enabled = False
        self.autosave_best_path = str(Path('saved_best_creature.json'))
        self.inject_saved_best_enabled = False

    def set_profile(self, profile_id: str) -> None:
        config.apply_profile(profile_id)
        self.profile_id = profile_id
        self.ticks_per_epoch = config.TICKS_PER_EPOCH

    def set_custom_map(self, custom_map_id: str | None) -> None:
        self.custom_map_id = custom_map_id or None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, preserve_hall_of_fame: bool = False) -> None:
        """Start fresh from epoch 0 with a new (or the same) seed."""
        if seed is not None:
            self.seed = seed
        self._rng = make_rng(self.seed)
        self.epoch_history = []
        self._testing_ground_completed_logged = False
        if not preserve_hall_of_fame:
            self.best_genome_ever = None
            self.best_fitness_ever = None
            self.best_epoch_ever = None

        if self.mode == 'testing_ground':
            if self.testing_ground_genome is None:
                self.testing_ground_genome = self._select_testing_ground_genome()
            if self.testing_ground_genome is not None:
                self.world = generate_world(
                    epoch_index=0,
                    seed=self.seed,
                    genomes=[copy.deepcopy(self.testing_ground_genome)],
                    population_size=1,
                    custom_map_id=self.custom_map_id,
                )
                log(f"Testing Ground reset. Seed={self.seed}  Creatures={len(self.world.creatures)}")
                return
            self.mode = 'evolution'

        self.world = generate_world(
            epoch_index=0,
            seed=self.seed,
            genomes=self._build_reset_genomes(),
            custom_map_id=self.custom_map_id,
        )
        log(f"Simulation reset. Seed={self.seed}  Creatures={len(self.world.creatures)}")

    def enter_testing_ground(self) -> bool:
        genome = self._select_testing_ground_genome()
        if genome is None:
            return False
        self.mode = 'testing_ground'
        self.testing_ground_genome = copy.deepcopy(genome)
        self.reset(preserve_hall_of_fame=True)
        return True

    def exit_testing_ground(self) -> None:
        self.mode = 'evolution'
        self.testing_ground_genome = None
        self.reset(preserve_hall_of_fame=True)

    def is_testing_ground(self) -> bool:
        return self.mode == 'testing_ground'

    def is_run_complete(self) -> bool:
        if self.world is None:
            return False
        if self.is_testing_ground():
            return self.alive_count() == 0
        return self.is_epoch_over()

    # ── Tick ───────────────────────────────────────────────────────────────────

    def step_tick(self) -> None:
        """Advance the simulation by exactly one tick."""
        if self.world is None:
            self.reset(preserve_hall_of_fame=self.is_testing_ground())
        assert self.world is not None

        if self.is_testing_ground():
            if self.alive_count() == 0:
                self._log_testing_ground_complete()
                return
            tick_world(self.world)
            if self.alive_count() == 0:
                self._log_testing_ground_complete()
            return

        if self.world.tick_index >= self.ticks_per_epoch or self.alive_count() == 0:
            self.step_epoch()
            return

        tick_world(self.world)

    # ── Epoch ──────────────────────────────────────────────────────────────────

    def step_epoch(self) -> None:
        """Finish the current epoch and start a new one."""
        if self.world is None:
            self.reset(preserve_hall_of_fame=self.is_testing_ground())
        assert self.world is not None

        if self.is_testing_ground():
            while self.alive_count() > 0:
                tick_world(self.world)
            self._log_testing_ground_complete()
            return

        # Drain remaining ticks if needed
        while self.world.tick_index < self.ticks_per_epoch and self.alive_count() > 0:
            tick_world(self.world)

        # Collect results and log
        results = collect_epoch_results(self.world)
        top_result = results[0] if results else None
        self.epoch_history.append({
            'epoch': self.world.epoch_index,
            'results': results,
            'top_fitness': top_result.fitness if top_result else 0.0,
            'top_creature_id': top_result.creature_id if top_result else None,
        })
        alive = sum(1 for c in self.world.creatures if c.lifetime.alive)
        top = top_result.fitness if top_result else 0.0
        log(
            f"Epoch {self.world.epoch_index} ended. "
            f"Alive: {alive}/{len(self.world.creatures)}  "
            f"Best fitness this epoch: {top:.2f}"
        )

        self._update_hall_of_fame(results)

        # Evolve
        next_genomes = evolve_next_generation(self.world, results, self._rng)
        next_genomes = self._preserve_hall_of_fame(next_genomes)
        next_epoch = self.world.epoch_index + 1

        self.world = generate_world(
            epoch_index=next_epoch,
            seed=self.seed,
            genomes=next_genomes,
            custom_map_id=self.custom_map_id,
        )
        log(f"Epoch {next_epoch} started. Population: {len(self.world.creatures)}")

    # ── Queries ────────────────────────────────────────────────────────────────

    def is_epoch_over(self) -> bool:
        if self.world is None:
            return False
        if self.is_testing_ground():
            return self.alive_count() == 0
        return self.world.tick_index >= self.ticks_per_epoch or self.alive_count() == 0

    def alive_count(self) -> int:
        if self.world is None:
            return 0
        return sum(1 for c in self.world.creatures if c.lifetime.alive)

    def find_creature_at(self, x: int, y: int) -> Creature | None:
        if self.world is None:
            return None
        for c in self.world.creatures:
            if c.lifetime.alive and c.lifetime.x == x and c.lifetime.y == y:
                return c
        return None

    def epoch_best_fitnesses(self) -> list[tuple[int, float]]:
        return [
            (entry['epoch'], entry['top_fitness'])
            for entry in self.epoch_history
        ]

    def current_best_creature(self) -> Creature | None:
        if self.world is None or not self.world.creatures:
            return None
        return max(self.world.creatures, key=compute_fitness)

    def configure_best_genome_persistence(
        self,
        *,
        autosave_enabled: bool,
        autosave_path: str,
        inject_saved_best_enabled: bool,
    ) -> None:
        self.autosave_best_enabled = autosave_enabled
        self.autosave_best_path = autosave_path
        self.inject_saved_best_enabled = inject_saved_best_enabled

    def has_saved_best_genome(self, file_path: str | None = None) -> bool:
        path = file_path or self.autosave_best_path
        if not path:
            return False
        return genome_file_exists(path)

    def _update_hall_of_fame(self, results: list[CreatureEpochResult]) -> None:
        if self.world is None or not results:
            return

        top_result = results[0]
        if self.best_fitness_ever is not None and top_result.fitness <= self.best_fitness_ever:
            return

        top_creature = next(
            (c for c in self.world.creatures if c.creature_id == top_result.creature_id),
            None,
        )
        if top_creature is None:
            return

        self.best_fitness_ever = top_result.fitness
        self.best_epoch_ever = self.world.epoch_index
        self.best_genome_ever = copy.deepcopy(top_creature.genome)
        self._autosave_best_genome_if_enabled()

    def _preserve_hall_of_fame(self, next_genomes: list[Genome]) -> list[Genome]:
        if self.best_genome_ever is None or not next_genomes:
            return next_genomes

        if any(genome == self.best_genome_ever for genome in next_genomes):
            return next_genomes

        preserved = list(next_genomes)
        preserved[-1] = copy.deepcopy(self.best_genome_ever)
        return preserved

    def _select_testing_ground_genome(self) -> Genome | None:
        if self.best_genome_ever is not None:
            return copy.deepcopy(self.best_genome_ever)

        best_creature = self.current_best_creature()
        if best_creature is None:
            return None
        return copy.deepcopy(best_creature.genome)

    def _log_testing_ground_complete(self) -> None:
        if self._testing_ground_completed_logged or self.world is None:
            return
        self._testing_ground_completed_logged = True
        log(
            f"Testing Ground ended at tick {self.world.tick_index}. "
            f"Creature survived {self.world.tick_index} ticks."
        )

    def _autosave_best_genome_if_enabled(self) -> None:
        if not self.autosave_best_enabled or self.best_genome_ever is None or not self.autosave_best_path:
            return
        save_genome_to_file(self.best_genome_ever, self.autosave_best_path)
        log(f"Autosaved best creature to {self.autosave_best_path}")

    def _build_reset_genomes(self) -> list[Genome] | None:
        if self.mode != 'evolution' or not self.inject_saved_best_enabled or not self.autosave_best_path:
            return None
        if not genome_file_exists(self.autosave_best_path):
            return None

        loaded_genome = load_genome_from_file(self.autosave_best_path)
        genomes = [
            make_random_genome(self._rng)
            for _ in range(max(0, config.POPULATION_SIZE - 1))
        ]
        genomes.append(loaded_genome)
        return genomes
