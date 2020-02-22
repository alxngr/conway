from typing import Tuple, List

import matplotlib.pyplot as plt

from visualization import Visualization
from genetic import GeneticAlgorithm, SimpleConwayChromosome


class GeneticExperiment:

    def __init__(self, *,
                 name: str,
                 population_size: int,
                 individual_shape: Tuple[int, int],
                 selection_threshold: float,
                 max_generations: int,
                 mutation_chance: float,
                 crossover_chance: float) -> None:
        self._name: str = name
        self._population_size: int = population_size
        self._individual_shape: Tuple[int, int] = individual_shape
        self._selection_threshold: float = selection_threshold
        self._max_generations: int = max_generations
        self._mutation_chance: float = mutation_chance
        self._crossover_chance: float = crossover_chance

    def run(self) -> None:
        # Run experiment
        initial_population: List[SimpleConwayChromosome] = [
            SimpleConwayChromosome.random_instance(shape=self._individual_shape)
            for _ in range(self._population_size)
        ]
        ga: GeneticAlgorithm[SimpleConwayChromosome] = GeneticAlgorithm(
            initial_population=initial_population,
            threshold=self._selection_threshold,
            max_generations=self._max_generations,
            mutation_chance=self._mutation_chance,
            crossover_chance=self._crossover_chance
        )
        result: SimpleConwayChromosome = ga.run()

        # Save result
        result.game.save(model_name=self._name)
        Visualization.plot_game(game=result.game, model_name=self._name)
        Visualization.plot_number_of_cell_evolution(game=result.game)

        # Plot evolution of best fitness and avg fitness
        best = ga.best_fitness_evolution
        avg = ga.avg_fitness_evolution

        plt.plot(best, 'b')
        plt.plot(avg, 'r')

        plt.title('Evolution of fitness with generations (best and average)')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        plt.show()


if __name__ == '__main__':
    experiment: GeneticExperiment = GeneticExperiment(
        name='experiment_test_4',
        population_size=20,
        individual_shape=(50, 50),
        selection_threshold=0.8,
        max_generations=100,
        mutation_chance=0.5,
        crossover_chance=0.8
    )
    experiment.run()
