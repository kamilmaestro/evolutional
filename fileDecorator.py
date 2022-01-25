def dropColumns(file, *columnNames):
    for columnName in columnNames:
        file.drop(columnName, axis=1, inplace=True)

    return file
