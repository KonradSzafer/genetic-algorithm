# genetic-algorithm

## Description

This is Genetic Algorithm implementation, written in Python.
The goal is to create a genetic algorithm that is quick to implement and easy to use.
This implementation introduces several new features that I came up with while working with it, such as support for stochastic problems and new types offspring creation.

## Documentation

### Instance initialization

```python
GeneticAlgorithm = GA(  generations_count=100,   # number of simulated generations
                        population_count=200,    # number of population in generation
                        function=function,       # function to optimize
                        params_bounds=bounds,    # variable bounds for each argument
                        fitness_threshold=None,  # threshold value of fitness for early stopping
                        maximize=False,          # decide to optimize values for minimizing or maximizing function output
                        floating_point=True,     # variables data type
                        stochastic=False,        # set True if function to optimize have stochastic nature
                        stochastic_iterations=3, # if function to optimize have stochastic nature, performs multiple calculations for every individual (>=3)
                        allow_gene_duplication=True, # allows the same genes to be calculated several times
                        crossover_percentage=0.3,# percentage of population reproduced by crossover
                        mutation_percentage=0.7  # percentage of population reproduced by mutation
                        )
```

### Methods

```python
GeneticAlgorithm.evolve( verbose = 1 )
```
Runs a solution search.
Verbosity mode. 0 = progress bar, 1 = generation number and best stats, 2 = adds console clear between generations.

```python
GeneticAlgorithm.get_best()
```
Returns best solution.

```python
GeneticAlgorithm.get_best_fitness()
```
Returns best solution fitness.

```python
GeneticAlgorithm.get_evolution_time()
```
Returns searching time in seconds.

```python
GeneticAlgorithm.plot_learning_curve(   title = 'Fitness over generations',
                                        xlabel = 'Generation',
                                        ylabel = 'Fitness',
                                        font_size = 12,
                                        line_width = 2,
                                        save_dir = None,
                                        image_name = 'learning_curve.png')
```
Plots learning curve and returns fitness array.
if save_dir is None, the image will not be saved.

```python
GeneticAlgorithm.get_searched_list()
```
Returns list of searched solutions.
