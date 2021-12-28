import os
import time
import random
import numpy as np
import numpy.typing as npt

from rich.table import Column
from rich.progress import Progress, BarColumn, TextColumn

from reproduction import creation, mutation, crossover

class GA:

    def __init__(self,
                generations_count, population_count,
                function,
                parameters_ranges,
                fitness_treshold=None,
                maximise=False,
                floating_point=False,
                stochastic=False,
                stochastic_iterations=3,
                crossover_percentage=0.3,
                mutation_percentage=0.7) -> None:

        # general
        self.generations_count = generations_count
        self.population_count = population_count
        self.function = function
        self.parameters_ranges = parameters_ranges
        self.parameters_count = len(parameters_ranges)
        self.fitness_treshold = fitness_treshold
        self.treshold_reached = False

        if not floating_point:
            for name, ranges in self.parameters_ranges.items():
                ranges[1] += 1

        # advanced
        self.maximise = maximise
        self.floating_point = floating_point
        self.stochastic = stochastic
        self.stochastic_iterations = stochastic_iterations
        self.crossover_percentage = crossover_percentage
        self.mutation_percentage = mutation_percentage

        self.linux = True
        if os.name == 'nt':
            self.linux = False

        # results
        self.evolution_time = np.nan
        self.fitness_array = []
        self.best_individual = []

        # error check
        for name, ranges in self.parameters_ranges.items():
            min_val, max_val = ranges
            if min_val >= max_val:
                raise

        total_percentage = crossover_percentage + mutation_percentage
        if total_percentage > 1:
            raise


    def __print_population(self, population):
        for key in self.parameters_ranges:
            print(' ', key, end=' ')
        print(' fitness')
        for individual in population:
            print(individual)


    def __create_population(self):
        population = []
        for _ in range(self.population_count):
            individual = creation(self.parameters_ranges, floating_point=self.floating_point)
            individual.append(np.nan)
            population.append(individual)
        return population


    def __calculate_fitness(self, population):
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

            population[idx][-1] = individual_fitness


    def __get_best_individual(self, population):
        best_individual = population[0].copy()
        self.best_individual = best_individual
        best_fitness = best_individual[-1]
        best_individual.pop()
        self.fitness_array.append(best_fitness)
        return best_individual, best_fitness


    def __reproduce_population(self, population):

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

            offspring = crossover(  self.parameters_ranges,
                                    parent_a,
                                    parent_b,
                                    floating_point=self.floating_point)

            offspring.append(np.nan)
            population[i] = offspring

        # mutation
        start = end
        end = start + mutation_count
        for i in range(start, end):

            offspring = mutation(self.parameters_ranges,
                                best_individual,
                                floating_point=self.floating_point)

            offspring.append(np.nan)
            population[i] = offspring

        # new
        start = end
        end = self.population_count
        for i in range(start, end):

            offspring = creation(self.parameters_ranges, floating_point=self.floating_point)

            offspring.append(np.nan)
            population[i] = offspring


    def evolve(self, stop_value=None, verbose=True) -> None:

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


    def get_evolution_time(self):

        return self.evolution_time


    def get_learning_curve(self):

        return self.fitness_array


    def get_best(self):

        return self.best_individual


    def get_best_fitness(self):

        return self.fitness_array[-1]
