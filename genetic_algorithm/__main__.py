import matplotlib.pyplot as plt
from genetic_algorithm import GA

def function(x, y, z):
    value = 3*x - 2*y + 10*z
    return value

if __name__ == '__main__':

    params_ranges = {
                    'x': (-10, 10),
                    'y': (-30, 30),
                    'z': (0, 50) }

    GeneticAlgorithm = GA(  generations_count=100,
                            population_count=200,
                            function=function,
                            parameters_ranges=params_ranges,
                            maximise=True,
                            floating_point=True,
                            stochastic=False,
                            stochastic_iterations=3,
                            crossover_percentage=0.3,
                            mutation_percentage=0.7 )

    GeneticAlgorithm.evolve( verbose=True )

    optymalization_time = GeneticAlgorithm.get_evolution_time()
    print('\nOptymalization time:', optymalization_time, 's')

    learning_curve = GeneticAlgorithm.get_learning_curve()
    plt.plot(learning_curve)
    plt.show()

    result = GeneticAlgorithm.get_best()
    print('Result', result)

    best_fitness = GeneticAlgorithm.get_best_fitness()
    print('Best fitness', best_fitness)
