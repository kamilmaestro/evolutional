import multiprocessing

from deap import base
from deap import creator
from deap import tools
import random

selmethod = 'best'
crossover = 'heuristic'
mutation = 'gaussian'
minimalization = True

sizePopulation = 100
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 100

mean_val = 5
std = 10
indpb = 0.06

results = []


def individual(icls, start=-10, stop=10):
    genome = list()
    genome.append(random.uniform(start, stop))
    genome.append(random.uniform(start, stop))

    return icls(genome)


def fitnessFunction(individual):
    result = (individual[0] + 2 * individual[1] - 7) ** 2 + (2 * individual[0] + individual[1] - 5) ** 2
    return result,


def heuristic(ind1, ind2):
    if (ind1[0] - ind1[0]) * (ind2[1] - ind2[1]) < 0:
        return ind1, ind2

    k1 = random.random()
    k2 = random.random()
    x1_1 = k1 * abs(ind2[0] - ind1[0]) + min(ind2[0], ind1[0])
    x2_1 = k1 * abs(ind2[1] - ind1[1]) + min(ind2[1], ind1[1])
    x1_2 = k2 * abs(ind2[0] - ind1[0]) + min(ind2[0], ind1[0])
    x2_2 = k2 * abs(ind2[1] - ind1[1]) + min(ind2[1], ind1[1])

    ind1[0] = x1_1
    ind1[1] = x2_1
    ind2[0] = x1_2
    ind2[1] = x2_2

    return ind1, ind2


if minimalization:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)


if selmethod == 'tournament':
    toolbox.register("select", tools.selTournament, tournsize=3)
elif selmethod == 'best':
    toolbox.register("select", tools.selBest)
else:
    toolbox.register("select", tools.selStochasticUniversalSampling)


if crossover == 'onepoint':
    toolbox.register("mate", tools.cxOnePoint)
elif crossover == 'heuristic':
    toolbox.register("mate", heuristic)
else:
    toolbox.register("mate", tools.cxOrdered)


if mutation == 'gaussian':
    toolbox.register("mutate", tools.mutGaussian, mu=mean_val, sigma=std, indpb=indpb)
elif mutation == 'shuffle':
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
elif mutation == 'multflipbit':
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
else:
    toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=indpb)


with open('results.csv', 'w') as f:
    for result in results:
        f.write(str(result) + '\n')


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)