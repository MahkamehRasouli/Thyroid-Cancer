"""
genetic_algorithm.py
---------------------
Implements Section 2.6 "Genetic Algorithm-Based Optimization of the EWMA
Control Chart": a GA that searches over (lambda, L) to minimize the number
of out-of-control observations, using tournament selection, blend
crossover, and Gaussian mutation, with elitism to preserve the best
chromosomes across generations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    GA_CROSSOVER_RATE,
    GA_ELITISM,
    GA_GENERATIONS,
    GA_L_BOUNDS,
    GA_LAMBDA_BOUNDS,
    GA_MUTATION_RATE,
    GA_MUTATION_STD,
    GA_POPULATION_SIZE,
    GA_RANDOM_STATE,
    GA_TOURNAMENT_SIZE,
)
from ewma import count_out_of_control


@dataclass
class GAResult:
    best_lambda: float
    best_L: float
    best_fitness: int
    history: list[float]  # best fitness per generation


def _clip(val: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(val, bounds[0], bounds[1]))


def _init_population(rng: np.random.Generator, pop_size: int) -> np.ndarray:
    lam = rng.uniform(*GA_LAMBDA_BOUNDS, size=pop_size)
    L = rng.uniform(*GA_L_BOUNDS, size=pop_size)
    return np.column_stack([lam, L])


def _fitness(chromosome: np.ndarray, x: np.ndarray, mu: float, sigma: float) -> int:
    lam, L = chromosome
    return count_out_of_control(x, mu, sigma, lam, L)


def _tournament_select(pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    best_idx = idx[np.argmin(fitness[idx])]  # lower fitness (fewer OOC points) wins
    return pop[best_idx].copy()


def _crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < GA_CROSSOVER_RATE:
        alpha = rng.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def _mutate(chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    chromosome = chromosome.copy()
    if rng.random() < GA_MUTATION_RATE:
        chromosome[0] += rng.normal(0, GA_MUTATION_STD)
        chromosome[0] = _clip(chromosome[0], GA_LAMBDA_BOUNDS)
    if rng.random() < GA_MUTATION_RATE:
        chromosome[1] += rng.normal(0, GA_MUTATION_STD * 2)  # L has a wider range
        chromosome[1] = _clip(chromosome[1], GA_L_BOUNDS)
    return chromosome


def optimize_ewma_parameters(
    x: np.ndarray,
    mu: float,
    sigma: float,
    population_size: int = GA_POPULATION_SIZE,
    generations: int = GA_GENERATIONS,
    random_state: int = GA_RANDOM_STATE,
) -> GAResult:
    """Run the GA to find (lambda, L) minimizing the number of out-of-control points.

    This directly implements the procedure described in Section 2.6: random
    initial population -> fitness = out-of-control count -> tournament
    selection -> blend crossover -> Gaussian mutation -> elitism -> repeat.
    """
    rng = np.random.default_rng(random_state)
    pop = _init_population(rng, population_size)
    history = []

    best_chromosome = None
    best_fitness = np.inf

    for _gen in range(generations):
        fitness = np.array([_fitness(c, x, mu, sigma) for c in pop])

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_chromosome = pop[gen_best_idx].copy()
        history.append(best_fitness)

        # Elitism: carry the top GA_ELITISM chromosomes forward unchanged
        elite_idx = np.argsort(fitness)[:GA_ELITISM]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < population_size:
            parent1 = _tournament_select(pop, fitness, rng, GA_TOURNAMENT_SIZE)
            parent2 = _tournament_select(pop, fitness, rng, GA_TOURNAMENT_SIZE)
            child1, child2 = _crossover(parent1, parent2, rng)
            child1 = _mutate(child1, rng)
            child2 = _mutate(child2, rng)
            new_pop.append(child1)
            if len(new_pop) < population_size:
                new_pop.append(child2)

        pop = np.array(new_pop)

    return GAResult(
        best_lambda=float(best_chromosome[0]),
        best_L=float(best_chromosome[1]),
        best_fitness=int(best_fitness),
        history=history,
    )
