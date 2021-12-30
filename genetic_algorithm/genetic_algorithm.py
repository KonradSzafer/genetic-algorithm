import os
import time
import random
import numpy as np
import numpy.typing as npt
from typing import Union, Callable, List, Dict

from rich.table import Column
from rich.progress import Progress, BarColumn, TextColumn

from reproduction import creation, mutation, crossover

class GA:

    def __init__(self,
                generations_count: int,
                population_count: int,
                function: Callable,
                params_bounds: Dict[str,List],
                fitness_treshold: int = None,
                maximise: bool = False,
                floating_point: bool = False,
                stochastic: bool = False,
                stochastic_iterations: int = 3,
                complex_function: bool = False,
                crossover_percentage: float = 0.3,
                mutation_percentage: float = 0.7) -> None:

        # general
        self.generations_count = generations_count
        self.population_count = population_count
        self.function = function
        self.parameters_bounds = params_bounds
        self.parameters_count = len(params_bounds)
        self.fitness_treshold = fitness_treshold
        self.treshold_reached = False

        for name, bounds in self.parameters_bounds.items():
            if floating_point:
                bounds[1] += 1e-6
            else:
                bounds[1] += 1

        # advanced
        self.maximise = maximise
        self.floating_point = floating_point
        self.stochastic = stochastic
        self.stochastic_iterations = stochastic_iterations
        self.complex_function = complex_function
        self.crossover_percentage = crossover_percentage
        self.mutation_percentage = mutation_percentage

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

        if self.stochastic:
            if self.stochastic_iterations < 3:
                raise ValueError('Number of stochastic iterations is to small.')

        total_percentage = crossover_percentage + mutation_percentage
        if total_percentage > 1:
            raise ValueError('The values assigned to crossover_percentage and mutation_percentage are too large.')


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
            individual.append(np.nan)
            population.append(individual)
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

            if self.fitness_treshold is not None:
                if self.maximise:
                    if individual_fitness > self.fitness_treshold:
                        self.treshold_reached =  True
                else:
                    if individual_fitness < self.fitness_treshold:
                        self.treshold_reached =  True

            self.searched.add(tuple(individual + [individual_fitness]))
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

            offspring.append(np.nan)
            population[i] = offspring

        # mutation
        start = end
        end = start + mutation_count
        for i in range(start, end):

            offspring = mutation(self.parameters_bounds,
                                best_individual,
                                floating_point=self.floating_point)

            offspring.append(np.nan)
            population[i] = offspring

        # new
        start = end
        end = self.population_count
        for i in range(start, end):

            offspring = creation(self.parameters_bounds, floating_point=self.floating_point)

            offspring.append(np.nan)
            population[i] = offspring


    def evolve(self, stop_value = None, verbose: bool = True) -> None:

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
                                    reverse=self.maximise)

                best_params, best_fitness = self.__get_best_individual(population)

                if self.treshold_reached:
                    break

                self.__reproduce_population(population)

                if self.linux:
                    os.system('clear')
                else:
                    os.system('cls')

                # progress.print('Generation:', generation+1, '/', self.generations_count)
                # self.__print_population(population)
                print('Generation:', generation+1, '/', self.generations_count)
                print('\nBest params:', best_params)
                print('fitness:', best_fitness, end='\n\n')

        end_time = time.time()
        self.evolution_time = end_time - start_time
        self.evolution_time = np.round(self.evolution_time, 2)


    def get_best(self) -> List:

        return self.best_individual


    def get_best_fitness(self) -> float:

        return self.fitness_array[-1]


    def get_evolution_time(self) -> float:

        return self.evolution_time


    def get_learning_curve(self) -> List:

        return self.fitness_array


    def get_searched_list(self, include_fitness: bool = False) -> List:

        searched_list = list(self.searched.copy())
        searched_list = [list(solution) for solution in searched_list]
        if not include_fitness:
            for i in searched_list:
                searched_list.pop()
        return searched_list
