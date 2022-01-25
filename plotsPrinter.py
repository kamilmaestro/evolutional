from matplotlib import pyplot as plt


def getValuesFromFile(filename):
    meanResult = []
    stdResult = []
    valuesResult = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        timeResult = float(lines[-1][0])
        lines = lines[:-1]
        for line in lines:
            line = line[1:-1].split(',')
            meanResult.append(float(line[0]))
            stdResult.append(float(line[1]))
            valuesResult.append(float(line[2][2:-4]))

    return meanResult, stdResult, valuesResult, timeResult


def plotResults(title, values, show=False):
    epochs = [i + 1 for i in range(len(values))]
    plt.plot(epochs, values)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title + ' values')
    if show:
        plt.show()
    else:
        plt.savefig(f'{title}.png')


if __name__ == '__main__':
    mean, std, values, time = getValuesFromFile('results_project_4.csv')
    plotResults('Mean', mean, show=True)
    plotResults('Std', std, show=True)
    plotResults('Function', values, show=True)
