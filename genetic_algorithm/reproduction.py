import random
import numpy as np


def creation(params_ranges, floating_point=True):
    parameters_ranges = params_ranges.copy()
    individual = []
    for name, ranges in parameters_ranges.items():
        min_val, max_val = ranges
        if floating_point:
            gene = np.random.uniform(min_val, max_val)
        else:
            gene = np.random.randint(min_val, max_val)
        individual.append(gene)
    return individual


def mutation(params_ranges, best_individual, mutation_rate=0.5, floating_point=True):
    parameters_ranges = params_ranges.copy()
    parent = best_individual.copy()
    offspring = []
    idx = 0
    for name, ranges in parameters_ranges.items():

        min_val, max_val = ranges
        abs_range = abs(max_val - min_val)

        if floating_point:
            mutation_value = abs_range * mutation_rate
        else:
            mutation_value = int(abs_range * mutation_rate)
        mutation_value /= 2

        gene = parent[idx]
        mutation_min_val = gene - mutation_value
        mutation_max_val = gene + mutation_value

        if mutation_min_val < min_val:
            mutation_min_val = min_val

        if mutation_max_val > max_val:
            mutation_max_val = max_val

        if floating_point:
            gene = np.random.uniform(mutation_min_val, mutation_max_val)
        else:
            gene = np.random.randint(mutation_min_val, mutation_max_val)

        offspring.append(gene)
        idx += 1

    return offspring


def crossover(params_ranges, parent_a, parent_b, floating_point=True):
    parameters_ranges = params_ranges.copy()
    offspring = []
    idx = 0
    for name, ranges in parameters_ranges.items():

        gene_a = parent_a[idx]
        gene_b = parent_b[idx]
        abs_range = abs(gene_a - gene_b)
        abs_range /= 2

        if gene_a == gene_b:
            gene = gene_a
        elif gene_a > gene_b:
            gene = gene_b + abs_range
        else:
            gene = gene_a + abs_range

        if floating_point:
            gene = float(gene)
        else:
            gene = int(gene)

        offspring.append(gene)
        idx += 1

    return offspring


if __name__ == '__main__':

    params_ranges = {
                    'x': (-10, 10),
                    'y': (-30, 30),
                    'z': (0, 50) }

    best_individual = [9, 10, 43, np.nan]

    offspring = mutation(params_ranges, best_individual)
    print(offspring)

    individual = [-9, 0, 30, np.nan]

    offspring = crossover(params_ranges, best_individual, individual)
    print(offspring)
