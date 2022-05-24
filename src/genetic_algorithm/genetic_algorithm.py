import os
import time
import random
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Union, Callable, List, Dict
from rich.table import Column
from rich.progress import Progress, BarColumn, TextColumn
from genetic_algorithm.reproduction import creation, mutation, crossover


class GA:

    def __init__(self,
                generations_count: int,
                population_count: int,
                function: Callable,
                params_bounds: Dict[str,List],
                initial_population: List[List] = None,
                fitness_threshold: int = None,
                maximize: bool = False,
                floating_point: bool = True,
                stochastic: bool = False,
                stochastic_iterations: int = 3,
                allow_gene_duplication: bool = True,
                crossover_percentage: float = 0.3,
                mutation_percentage: float = 0.7) -> None:

        # general
        self.generations_count = generations_count
        self.population_count = population_count
        self.function = function
        self.parameters_bounds = params_bounds
        self.parameters_count = len(params_bounds)
        self.initial_population = initial_population
        self.fitness_threshold = fitness_threshold
        self.treshold_reached = False

        for name, bounds in self.parameters_bounds.items():
            if floating_point:
                bounds[1] += 1e-6
            else:
                bounds[1] += 1

        # advanced
        self.maximize = maximize
        self.floating_point = floating_point
        self.stochastic = stochastic
        self.stochastic_iterations = stochastic_iterations
        self.allow_gene_duplication = allow_gene_duplication
        self.crossover_percentage = crossover_percentage
        self.mutation_percentage = mutation_percentage
        self.offspring_attempts = 3

        self.linux = True
        if os.name == 'nt':
            self.linux = False

        # results
        self.evolution_time = np.nan
        self.searched = set()
        self.fitness_array = []
        self.best_individual = []

        # validating arguments
        for name, bounds in self.parameters_bounds.items():
            min_val, max_val = bounds
            if min_val >= max_val:
                raise ValueError('Incorrect format of parameter bounds.')

        if self.initial_population is not None:

            self.initial_population_count = len(initial_population)
            self.population_count -= self.initial_population_count

            if len(self.initial_population) > self.population_count:
                raise ValueError('Initial population is bigger than defined population count.')

            for individual in self.initial_population:
                if len(individual) is not self.parameters_count:
                    raise ValueError('Individual parameters count is incorrect.')

                idx = 0
                for name, bounds in self.parameters_bounds.items():
                    min_val, max_val = bounds
                    param = individual[idx]

                    if not isinstance(param, int):
                        if self.floating_point:
                            if not isinstance(param, float):
                                raise ValueError('Initial population parameters must be of floating point or integer type.')
                        else:
                            raise ValueError('Initial population parameters must be of integer type.')

                    if not min_val <= param <= max_val:
                        raise ValueError('Parameter of individual is outside the bounds.')
                    idx += 1
                individual.append(np.nan)

        if self.stochastic:
            if self.stochastic_iterations < 3:
                raise ValueError('Number of stochastic iterations is to small.')

        total_percentage = crossover_percentage + mutation_percentage
        if total_percentage > 1:
            raise ValueError('The values assigned to crossover_percentage and mutation_percentage are too large.')


    def __was_searched(self, solution: List) -> bool:
        if not self.allow_gene_duplication:
            if tuple(solution) in self.searched:
                return True
        return False


    def __print_population(self, population: List) -> List:
        for key in self.parameters_bounds:
            print(' ', key, end=' ')
        print(' fitness')
        for individual in population:
            print(individual)


    def __create_population(self) -> List:
        population = []

        for _ in range(self.population_count):
            individual = creation(self.parameters_bounds, floating_point=self.floating_point)
            if not self.allow_gene_duplication:
                if individual in population:
                    for _ in range(self.offspring_attempts):
                        individual = creation(self.parameters_bounds, floating_point=self.floating_point)
                        if individual not in population:
                            break

            individual.append(np.nan)
            population.append(individual)

        if self.initial_population is not None:
            population += self.initial_population

        return population


    def __calculate_fitness(self, population: List) -> None:
        for idx in range(self.population_count):
            individual_fitness = np.nan
            individual = population[idx].copy()
            individual.pop()

            if self.stochastic:
                for i in range(self.stochastic_iterations):
                    individual_fitness += self.function( *individual )
                individual_fitness = int(individual_fitness / self.stochastic_iterations)
            else:
                individual_fitness = self.function( *individual )

            if self.fitness_threshold is not None:
                if self.maximize:
                    if individual_fitness > self.fitness_threshold:
                        self.treshold_reached =  True
                else:
                    if individual_fitness < self.fitness_threshold:
                        self.treshold_reached =  True

            if not self.allow_gene_duplication:
                # searched dont include fittnes
                self.searched.add(tuple(individual))

            population[idx][-1] = individual_fitness


    def __get_best_individual(self, population: List) -> Union[List,int]:
        best_individual = population[0].copy()
        self.best_individual = best_individual
        best_fitness = best_individual[-1]
        best_individual.pop()
        self.fitness_array.append(best_fitness)
        return best_individual, best_fitness


    def __reproduce_population(self, population: List) -> None:

        best_count = 1
        crossover_count = int(self.population_count * self.crossover_percentage) - 1
        mutation_count = int(self.population_count * self.mutation_percentage) - 1

        # leave best
        best_individual = population[0].copy()

        # crossover
        start = best_count
        end = start + crossover_count
        for i in range(start, end):
            parent_a = best_individual
            parent_b = population[i].copy()

            offspring = crossover(  self.parameters_bounds,
                                    parent_a,
                                    parent_b,
                                    floating_point=self.floating_point)

            if self.__was_searched(offspring):
                for _ in range(self.offspring_attempts):
                    offspring = creation(self.parameters_bounds, floating_point=self.floating_point)
                    if not self.__was_searched(offspring):
                        break

            offspring.append(np.nan)
            population[i] = offspring

        # mutation
        start = end
        end = start + mutation_count
        for i in range(start, end):

            offspring = mutation(self.parameters_bounds,
                                best_individual,
                                floating_point=self.floating_point)

            if self.__was_searched(offspring):
                for _ in range(self.offspring_attempts):
                    offspring = creation(self.parameters_bounds, floating_point=self.floating_point)
                    if not self.__was_searched(offspring):
                        break

            offspring.append(np.nan)
            population[i] = offspring

        # new
        start = end
        end = self.population_count
        for i in range(start, end):

            for _ in range(self.offspring_attempts + 1):
                offspring = creation(self.parameters_bounds, floating_point=self.floating_point)
                if not self.__was_searched(offspring):
                    break

            offspring.append(np.nan)
            population[i] = offspring


    def evolve(self, stop_value = None, verbose: int = 1) -> None:

        start_time = time.time()
        population = self.__create_population()

        text_column = TextColumn('{task.description}', table_column=Column(ratio=1))
        bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
        progress = Progress(text_column, bar_column, expand=True)

        with progress:
            for generation in progress.track(range(self.generations_count)):

                self.__calculate_fitness(population)

                population = sorted(population,
                                    key=lambda x: x[-1],
                                    reverse=self.maximize)

                best_params, best_fitness = self.__get_best_individual(population)

                if self.treshold_reached:
                    break

                self.__reproduce_population(population)

                if verbose > 0:
                    if verbose > 1:
                        if self.linux:
                            os.system('clear')
                        else:
                            os.system('cls')

                    learning_stats = 'Generation: [green]{0}[/green]/[green]{1}[/green]\nBest params: {2}\nBest fitness: {3}\n'

                    learning_stats = learning_stats.format( generation+1, self.generations_count,
                                                            best_params, best_fitness)

                    progress.print(learning_stats)

        end_time = time.time()
        self.evolution_time = end_time - start_time
        self.evolution_time = np.round(self.evolution_time, 2)


    def get_best(self) -> List:
        return self.best_individual


    def get_best_fitness(self) -> float:
        return self.fitness_array[-1]


    def get_evolution_time(self) -> float:
        return self.evolution_time


    def plot_learning_curve(self,
                            title: str = 'Fitness over generations',
                            xlabel: str = 'Generation',
                            ylabel: str = 'Fitness',
                            font_size: int = 12,
                            line_width: int = 2,
                            save_dir: str = None,
                            image_name: str = 'learning_curve.png',
                            ) -> List:

        if len(self.fitness_array) < 1:
            raise RuntimeError('plot_learning_curve() method can be called only for number of generations > 1')

        plt.plot(self.fitness_array, linewidth=line_width)
        plt.title(title, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.show()

        if save_dir is not None:
            plt.savefig(save_dir + image_name)

        return self.fitness_array


    def get_searched_list(self) -> List:
        if self.allow_gene_duplication:
            raise RuntimeError(
                'Searched list is not collected with allow_gene_duplication=True'
            )
        searched_list = list(self.searched.copy())
        searched_list = [list(solution) for solution in searched_list]
        return searched_list
