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

toolbox = base.Toolbox()

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)
    print("Time: %s" % (t2 - t1))