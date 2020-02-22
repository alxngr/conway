from __future__ import annotations
from typing import Tuple, TypeVar, Type, Generic, List, Callable, Set
from abc import ABC, abstractmethod
from random import choice, choices, random
from enum import Enum
from heapq import nlargest
from statistics import mean
from copy import deepcopy
import numpy as np

from game import ConwayGame, Cell
import config

T = TypeVar('T', bound='Chromosome')


class SelectionType(Enum):
    ROULETTE = 0
    TOURNAMENT = 1


class GeneticAlgorithm(Generic[T]):

    def __init__(self,
                 initial_population: List[T],
                 threshold: float,
                 max_generations: int = 100,
                 mutation_chance: float = 0.01,
                 crossover_chance: float = 0.7,
                 selection_type: SelectionType = SelectionType.TOURNAMENT):
        self._population: List[T] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_chance: float = mutation_chance
        self._crossover_chance: float = crossover_chance
        self._selection_type: SelectionType = selection_type
        self._fitness_key: Callable = type(self._population[0]).fitness
        self.best_fitness_evolution: List[float] = []
        self.avg_fitness_evolution: List[float] = []

    def _pick_roulette(self, wheel: List[float]) -> Tuple[T, T]:
        """Use a probability distribution wheel to pick two parents."""

        return tuple(choices(self._population, weights=wheel, k=2))

    def _pick_tournament(self, num_participants: int) -> Tuple[T, T]:
        """Pick num_participants at random and take the best two."""

        participants: List[T] = choices(self._population, k=num_participants)
        return tuple(nlargest(2, participants, key=self._fitness_key))

    def _reproduce_and_replace(self) -> None:
        """Replace the population with a new generation of individuals."""

        new_population: List[T] = []

        while len(new_population) < len(self._population):
            # Pick two parents
            if self._selection_type == SelectionType.ROULETTE:
                parents: Tuple[T, T] = self._pick_roulette([x.fitness() for x in self._population])
            else:
                # Selecting from a two large sample of the initial population lead to a less diverse population
                parents: Tuple[T, T] = self._pick_tournament(int(0.5 * len(self._population)))

            # Potential crossover between the two parents
            if random() < self._crossover_chance:
                new_population.extend(parents[0].crossover(parents[1]))
            else:
                new_population.extend(parents)

        # If the population count was an odd number with have an extra individual
        if len(new_population) > len(self._population):
            new_population.pop()

        self._population = new_population

    def _mutate(self) -> None:
        """Mutate individuals with a _mutation_chance probability."""

        for individual in self._population:
            if random() < self._mutation_chance:
                individual.mutate()

    def run(self) -> T:
        """Run the algorithm for max_generations iterations and return the best individual found."""

        best: T = max(self._population, key=self._fitness_key)
        for generation in range(self._max_generations):

            if best.fitness() >= self._threshold:
                return best

            avg = mean(map(self._fitness_key, self._population))
            print(f'Generation {generation} '
                  f'Best {best.fitness()} '
                  f'Avg {avg}')

            self.best_fitness_evolution.append(best.fitness())
            self.avg_fitness_evolution.append(avg)

            self._reproduce_and_replace()
            self._mutate()

            highest: T = max(self._population, key=self._fitness_key)
            if highest.fitness() > best.fitness():
                best = highest

        return best


class Chromosome(ABC):

    @abstractmethod
    def fitness(self) -> float:
        ...

    @classmethod
    @abstractmethod
    def random_instance(cls: Type[T], *args) -> T:
        ...

    @abstractmethod
    def crossover(self: T, other: T) -> Tuple[T, T]:
        ...

    @abstractmethod
    def mutate(self) -> None:
        ...


class GenericConwayChromosome(Chromosome):

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape: Tuple[int, int] = shape
        self.game: ConwayGame = ConwayGame(initial_state=ConwayGame.get_random_state(size=shape))

    @abstractmethod
    def fitness(self) -> float:
        ...

    @classmethod
    @abstractmethod
    def random_instance(cls: GenericConwayChromosome, shape: Tuple[int, int]) -> GenericConwayChromosome:
        ...

    def crossover(self, other: GenericConwayChromosome) -> Tuple[GenericConwayChromosome, GenericConwayChromosome]:
        """
        Crossover between two Conway games initial states.

        The idea is that cell that are ON on one state can become ON on the other state and vice-versa.
        When one cell changes state it is turned OFF on the former one.
        """

        child1: GenericConwayChromosome = deepcopy(self)
        child2: GenericConwayChromosome = deepcopy(other)

        for cell in self.game.initial_state:
            if random() < 0.5:
                child1.game.initial_state.discard(cell)
                child2.game.initial_state.add(cell)

        for cell in other.game.initial_state:
            if random() < 0.5:
                child2.game.initial_state.discard(cell)
                child1.game.initial_state.add(cell)

        # Make sure games will be simulated again
        child1.game.reset()
        child2.game.reset()

        return child1, child2

    def mutate(self) -> None:
        """
        Mutation of one Conway game initial state.

        Every cell that is on has a probability to turn ON only one of its neighbors.
        It is then turned off.
        """

        new_initial_state: Set[Cell] = deepcopy(self.game.initial_state)

        for cell in self.game.initial_state:
            neighbors: List[Cell] = self.game.get_neighbors(cell=cell)
            neighbor = choice(neighbors)
            new_initial_state.add(neighbor)
            new_initial_state.discard(cell)

        self.game.initial_state = new_initial_state

        # Make sure game will be simulated again
        self.game.reset()


class SimpleConwayChromosome(GenericConwayChromosome):

    def fitness(self) -> float:
        """
        Try to find a simple estimation of the total entropy of a game.

        The idea is to use the frequency of each cell turning ON/OFF as a proxy for it.
        """

        if not self.game.has_been_simulated:
            self.game.simulate(show_progress=False)

        frequencies = np.zeros(shape=self.shape)
        for epoch in range(1, len(self.game.states)):
            # Cells that turned ON
            for cell_on in self.game.states[epoch]:
                if cell_on not in self.game.states[epoch - 1]:
                    frequencies[cell_on.i, cell_on.j] += 1
            # Cells that turned OFF
            for cell_on in self.game.states[epoch - 1]:
                if cell_on not in self.game.states[epoch]:
                    frequencies[cell_on.i, cell_on.j] += 1

        frequencies /= len(self.game.states)

        entropy = float(np.sum(frequencies))
        entropy /= self.shape[0] * self.shape[1]

        # Privilege game that last longer
        entropy *= len(self.game.states) / config.MAX_ITERATIONS

        return entropy

    @classmethod
    def random_instance(cls: SimpleConwayChromosome, shape: Tuple[int, int]) -> SimpleConwayChromosome:
        return SimpleConwayChromosome(shape=shape)
