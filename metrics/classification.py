from numpy import argmax, sum, array, repeat, arange, eye

# Calcula o número de previsões corretas para dados one-hot encoded
def correctOneHotEncod(Y:array, Y_pred:array):
    # Indice da classe 
    y_true_class = argmax(Y, axis=1)
    y_pred_class = argmax(Y_pred, axis=1)
    return sum(y_true_class == y_pred_class)

# Calcula a acurácia para dados one-hot encoded
def accuracy(Y:array, Y_pred:array):
    # Calcule o número de previsões corretas
    correct = correctOneHotEncod(Y, Y_pred)
    # Numero de amostras
    total = Y.shape[0]
    # Calcule a acurácia
    return correct / total


# Constroi as classes usando one-hot
def classesEncoder(X:array,clusters: int, one_hot_encoder:bool = True):
    n_amostras = X.shape[0]
    # Calcula a quantidade de classes existentes
    n_classes = n_amostras // clusters
    # Gera os rótulos
    y = repeat(arange(n_classes), clusters)
    if one_hot_encoder:
        return eye(n_classes)[y]
    else:
        return y.reshape(-1, 1)
    
from numpy import array,append, median, std, mean, min, max

def statics(Y:array):
    return {
        "min": min(Y),
        "max": max(Y),
        "mean": mean(Y),
        "median": median(Y),
        "std": std(Y)
    }