from numpy import array, mean, std, min, max

def zscore(X: array) -> array:
    return (X - mean(X)) / std(X)

def minMaxScale(X:array, metodo:str = "0", axis = 0):
    if metodo == "0":
        return (X - min(X, axis=axis)) / (max(X, axis=axis) - min(X, axis=axis))
    elif metodo == "-1":
        return 2 * (X - min(X, axis=axis)) / (max(X, axis=axis) - min(X, axis=axis)) - 1