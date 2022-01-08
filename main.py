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
import math

from sklearn.tree import DecisionTreeClassifier

selmethod = 'best'
crossover = 'onepoint'
minimalization = False

sizePopulation = 50
probabilityMutation = 0.3
probabilityCrossover = 0.4
numberIteration = 100
numberElitism = 4
selection = True
classifier = 'AdaBoostClassifier'  # SVC KNeighborsClassifier DecisionTreeClassifier RandomForestClassifier GaussianNB AdaBoostClassifier

std = 0

data = 'data'  # data heart

if data == 'data':
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("./data.csv", sep=',')

    y = df['Status']
    df.drop('Status', axis=1, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    df.drop('Recording', axis=1, inplace=True)


    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    clf = SVC()
    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(scores.mean())
else:
    df = pd.read_csv("./heart.csv", sep=',')
    y = df['output']
    df.drop('output', axis=1, inplace=True)

numberOfAtributtes = len(df.columns)
print(numberOfAtributtes)

def SVCParameters(numberFeatures, icls):
    genome = list()
    #kernel
    listKernel = ["linear","rbf", "poly","sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    #c
    k = random.uniform(0.1, 100)
    genome.append(k)
    #degree
    genome.append(random.uniform(0.1, 5))
    #gamma
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)
    # coeff
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
    # w oparciu o macierze pomyłek https://www.dataschool.io/simple-guide-to-confusion-matrixterminology/
        result_sum = result_sum + result
    # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej
    return result_sum / split,


def mutationSVC(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer==0:
    # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0]=listKernel[random.randint(0, 3)]
    elif numberParamer==1:
    #C
        k = random.uniform(0.1,100)
        individual[1]=k
    elif numberParamer == 2:
    #degree
        individual[2]=random.uniform(0.1, 5)
    elif numberParamer == 3:
    #gamma
        gamma = random.uniform(0.01, 5)
        individual[3]=gamma
    elif numberParamer ==4:
    # coeff
        coeff = random.uniform(0.1, 20)
        individual[2] = coeff


def ParametersFeatureFitness(y,df,numberOfAtributtes,individual):
    split=5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop=[] #lista cech do usuniecia
    for i in range(numberOfAtributtes,len(individual)):
        if individual[i]==0: #gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i-numberOfAtributtes)

    dfSelectedFeatures=df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

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
        estimator = SVC(kernel=individual[0],C=individual[1],degree=individual[2],gamma=individual[3],
                        coef0=individual[4],random_state=101)

    # estimator = KNeighborsClassifier(n_neighbors=5)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        # w oparciu o macierze pomyłek https://www.dataschool.io/simple-guide-to-confusion-matrixterminology/
        resultSum = resultSum + result
        # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej
    return resultSum / split,



def SVCParametersFeatures(numberFeatures, icls):
    genome = list()
    # kernel
    listKernel = ["linear","rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    #c
    k = random.uniform(0.1, 100)
    genome.append(k)
    #degree
    genome.append(random.uniform(0.1,5))
    #gamma
    gamma = random.uniform(0.001,1)
    genome.append(gamma)
    # coeff
    coeff = random.uniform(0.01, 1)
    genome.append(coeff)
    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def mutationSVCFeatures(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer==0:
    # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0]=listKernel[random.randint(0, 3)]
    elif numberParamer==1:
    #C
        k = random.uniform(0.1,100)
        individual[1]=k
    elif numberParamer == 2:
    #degree
        individual[2]=random.uniform(0.1, 5)
    elif numberParamer == 3:
    #gamma
        gamma = random.uniform(0.01, 1)
        individual[3]=gamma
    elif numberParamer ==4:
    # coeff
        coeff = random.uniform(0.1, 1)
        individual[2] = coeff
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0


# choosing the fitness function
if minimalization:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

# toolbox prepration
toolbox = base.Toolbox()

if selection:
    toolbox.register('individual', SVCParametersFeatures, numberOfAtributtes, creator.Individual)
    toolbox.register("evaluate", ParametersFeatureFitness, y, df, numberOfAtributtes)
    toolbox.register("mutate", mutationSVCFeatures)
else:
    toolbox.register('individual', SVCParameters, numberOfAtributtes, creator.Individual)
    toolbox.register("evaluate", ParametersFitness, y, df, numberOfAtributtes)
    toolbox.register("mutate", mutationSVC)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# choosing the selection method
if selmethod == 'tournament':
    toolbox.register("select", tools.selTournament, tournsize=3)
elif selmethod == 'best':
    toolbox.register("select", tools.selBest)
elif selmethod == 'random':
    toolbox.register("select", tools.selRandom)
elif selmethod == 'worst':
    toolbox.register("select", tools.selWorst)
elif selmethod == 'roulette':
    toolbox.register("select", tools.selRoulette)
elif selmethod == 'doubletournament':
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

pop = toolbox.population(n=sizePopulation)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

g = 0
results = []

t1 = time.time()
while g < numberIteration:
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    listElitism = []
    for x in range(0, numberElitism):
        listElitism.append(tools.selBest(pop, 1)[0])

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < probabilityCrossover:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < probabilityMutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))
    pop[:] = offspring + listElitism
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
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