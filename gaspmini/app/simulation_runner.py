# app/simulation_runner.py
# High-level simulation controller used by the GUI.
# Owns world state, drives tick/epoch transitions.

from __future__ import annotations

import copy
import random

from app.config import TICKS_PER_EPOCH, POPULATION_SIZE, DEFAULT_SEED
from app.models import WorldState, Creature, Genome, CreatureEpochResult
from app.world import generate_world, make_rng
from app.simulation import tick_world
from app.evolution import collect_epoch_results, evolve_next_generation
from app.logging_utils import log


class SimulationRunner:
    def __init__(self, seed: int = DEFAULT_SEED, ticks_per_epoch: int = TICKS_PER_EPOCH) -> None:
        self.seed = seed
        self.ticks_per_epoch = ticks_per_epoch
        self.world: WorldState | None = None
        self._rng: random.Random = make_rng(seed)
        self.epoch_history: list[dict] = []  # brief per-epoch summary
        self.best_genome_ever: Genome | None = None
        self.best_fitness_ever: float | None = None
        self.best_epoch_ever: int | None = None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Start fresh from epoch 0 with a new (or the same) seed."""
        if seed is not None:
            self.seed = seed
        self._rng = make_rng(self.seed)
        self.epoch_history = []
        self.best_genome_ever = None
        self.best_fitness_ever = None
        self.best_epoch_ever = None
        self.world = generate_world(
            epoch_index=0,
            seed=self.seed,
        )
        log(f"Simulation reset. Seed={self.seed}  Creatures={len(self.world.creatures)}")

    # ── Tick ───────────────────────────────────────────────────────────────────

    def step_tick(self) -> None:
        """Advance the simulation by exactly one tick."""
        if self.world is None:
            self.reset()
        assert self.world is not None

        if self.world.tick_index >= self.ticks_per_epoch or self.alive_count() == 0:
            self.step_epoch()
            return

        tick_world(self.world)

    # ── Epoch ──────────────────────────────────────────────────────────────────

    def step_epoch(self) -> None:
        """Finish the current epoch and start a new one."""
        if self.world is None:
            self.reset()
        assert self.world is not None

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
        )
        log(f"Epoch {next_epoch} started. Population: {len(self.world.creatures)}")

    # ── Queries ────────────────────────────────────────────────────────────────

    def is_epoch_over(self) -> bool:
        if self.world is None:
            return False
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

    def _preserve_hall_of_fame(self, next_genomes: list[Genome]) -> list[Genome]:
        if self.best_genome_ever is None or not next_genomes:
            return next_genomes

        if any(genome == self.best_genome_ever for genome in next_genomes):
            return next_genomes

        preserved = list(next_genomes)
        preserved[-1] = copy.deepcopy(self.best_genome_ever)
        return preserved
