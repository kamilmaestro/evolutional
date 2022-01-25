from enum import Enum
import pandas as pd


class FileType(Enum):
    PARKINSON = 1
    HEART = 2


def getFileReadName(fileType):
    files = {
        FileType.PARKINSON: "./data.csv",
        FileType.HEART: "./heart.csv"
    }
    return files.get(fileType)


def getFileToAnalyze(fileType):
    filePath = getFileReadName(fileType)
    print("Reading file: " + filePath)
    return pd.read_csv(filePath, sep=',')
