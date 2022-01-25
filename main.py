import pandas as pd
import multiprocessing

from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import random
from deap import tools, base, creator
import time
from sklearn.tree import DecisionTreeClassifier
from csvReader import getFileToAnalyze, FileType
from fileDecorator import dropColumns

selectionMethod = 'best'
crossover = 'onepoint'
minimization = False
sizePopulation = 50
probabilityMutation = 0.3
probabilityCrossover = 0.4
numberIteration = 100
numberElitism = 4
selection = True
classifier = 'AdaBoostClassifier'  # SVC KNeighborsClassifier DecisionTreeClassifier RandomForestClassifier GaussianNB AdaBoostClassifier
std = 0

fileType = FileType.PARKINSON
file = getFileToAnalyze(fileType)
yAxis = file['Status'] if fileType == FileType.PARKINSON else file['output']

if fileType == FileType.PARKINSON:
    pd.set_option('display.max_columns', None)
    file = dropColumns(file, 'Status', 'ID', 'Recording')

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(file)
    clf = SVC()
    scores = model_selection.cross_val_score(clf, df_norm, yAxis, cv=5, scoring='accuracy', n_jobs=-1)
    print(scores.mean())
else:
    file = dropColumns(file, 'output')

numberOfAttributes = len(file.columns)
print(numberOfAttributes)


def SVCParameters(numberFeatures, icls):
    genome = list()
    listKernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    k = random.uniform(0.1, 100)
    genome.append(k)
    genome.append(random.uniform(0.1, 5))
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)
    coeff = random.uniform(0.01, 10)
    genome.append(coeff)
    return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    if classifier == 'KNeighborsClassifier':
        estimator = KNeighborsClassifier(5)
    elif classifier == 'DecisionTreeClassifier':
        estimator = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'RandomForestClassifier':
        estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif classifier == 'GaussianNB':
        estimator = GaussianNB()
    elif classifier == 'AdaBoostClassifier':
        estimator = AdaBoostClassifier()
    else:
        estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                        coef0=individual[4], random_state=101)

    result_sum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        result_sum = result_sum + result
    return result_sum / split,


def mutationSVC(individual):
    numberParameter = random.randint(0, len(individual) - 1)
    if numberParameter == 0:
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = listKernel[random.randint(0, 3)]
    elif numberParameter == 1:
        k = random.uniform(0.1, 100)
        individual[1] = k
    elif numberParameter == 2:
        individual[2] = random.uniform(0.1, 5)
    elif numberParameter == 3:
        gamma = random.uniform(0.01, 5)
        individual[3] = gamma
    elif numberParameter == 4:
        coeff = random.uniform(0.1, 20)
        individual[2] = coeff


def ParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAtributtes)

    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    if classifier == 'KNeighborsClassifier':
        estimator = KNeighborsClassifier(5)
    elif classifier == 'DecisionTreeClassifier':
        estimator = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'RandomForestClassifier':
        estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif classifier == 'GaussianNB':
        estimator = GaussianNB()
    elif classifier == 'AdaBoostClassifier':
        estimator = AdaBoostClassifier()
    else:
        estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                        coef0=individual[4], random_state=101)

    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def SVCParametersFeatures(numberFeatures, icls):
    genome = list()
    listKernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    k = random.uniform(0.1, 100)
    genome.append(k)
    genome.append(random.uniform(0.1, 5))
    gamma = random.uniform(0.001, 1)
    genome.append(gamma)
    coeff = random.uniform(0.01, 1)
    genome.append(coeff)
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def mutationSVCFeatures(individual):
    numberParameter = random.randint(0, len(individual) - 1)
    if numberParameter == 0:
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = listKernel[random.randint(0, 3)]
    elif numberParameter == 1:
        k = random.uniform(0.1, 100)
        individual[1] = k
    elif numberParameter == 2:
        individual[2] = random.uniform(0.1, 5)
    elif numberParameter == 3:
        gamma = random.uniform(0.01, 1)
        individual[3] = gamma
    elif numberParameter == 4:
        coeff = random.uniform(0.1, 1)
        individual[2] = coeff
    else:
        if individual[numberParameter] == 0:
            individual[numberParameter] = 1
        else:
            individual[numberParameter] = 0


if minimization:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

if selection:
    toolbox.register('individual', SVCParametersFeatures, numberOfAttributes, creator.Individual)
    toolbox.register("evaluate", ParametersFeatureFitness, yAxis, file, numberOfAttributes)
    toolbox.register("mutate", mutationSVCFeatures)
else:
    toolbox.register('individual', SVCParameters, numberOfAttributes, creator.Individual)
    toolbox.register("evaluate", ParametersFitness, yAxis, file, numberOfAttributes)
    toolbox.register("mutate", mutationSVC)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

if selectionMethod == 'tournament':
    toolbox.register("select", tools.selTournament, tournsize=3)
elif selectionMethod == 'best':
    toolbox.register("select", tools.selBest)
elif selectionMethod == 'random':
    toolbox.register("select", tools.selRandom)
elif selectionMethod == 'worst':
    toolbox.register("select", tools.selWorst)
elif selectionMethod == 'roulette':
    toolbox.register("select", tools.selRoulette)
elif selectionMethod == 'doubletournament':
    toolbox.register("select", tools.selDoubleTournament, tournsize=3, parsimony=2, fitness_first=False)
else:
    toolbox.register("select", tools.selStochasticUniversalSampling)

# choosing crossover method
if crossover == 'onepoint':
    toolbox.register("mate", tools.cxOnePoint)
elif crossover == 'uniform':
    toolbox.register("mate", tools.cxUniform)
elif crossover == 'twopoint':
    toolbox.register("mate", tools.cxTwoPoint)
else:
    toolbox.register("mate", tools.cxOrdered)

population = toolbox.population(n=sizePopulation)
fitnesses = toolbox.map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

g = 0
results = []

t1 = time.time()
while g < numberIteration:
    g = g + 1
    print("-- Generation %i --" % g)

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    listElitism = []
    for x in range(0, numberElitism):
        listElitism.append(tools.selBest(population, 1)[0])

    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        if random.random() < probabilityCrossover:
            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probabilityMutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))
    population[:] = offspring + listElitism
    fits = [ind.fitness.values[0] for ind in population]

    length = len(population)
    mean = sum(fits) / length
    squaredSum = sum(x * x for x in fits)
    std = abs(squaredSum / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    results.append([mean, std, best_ind.fitness.values])

print("-- End of (successful) evolution --")
t2 = time.time()

with open('results_project_4.csv', 'w') as f:
    for result in results:
        f.write(str(result) + '\n')
    f.write(str(t2 - t1))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)
    print("Time: %s" % (t2 - t1))
