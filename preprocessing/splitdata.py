from numpy import array, random

def randomOrder(X:array, Y:array):
    # Verifica se o tamanho das linhas de X e Y são iguais
    linha_x = X.shape[0]
    linha_y = Y.shape[0]
    # Se o número de amostras forem iguais
    if linha_x == linha_y:
        # Gera indices aleatórios
        indices_random = random.permutation(linha_x)
        # Retorna os dados com ordem aleatoria
        x = X[indices_random,:]
        y = Y[indices_random,:]
        return x, y
    else:
        return None, None
def randomSplit(X:array, Y:array, test_size: float = 0.3):
    # Verifica se o tamanho das linhas de X e Y são iguais
    linha_x = X.shape[0]
    linha_y = Y.shape[0]
    # Se o número de amostras forem iguais
    if linha_x == linha_y:
        split = int(linha_x * (1 - test_size))
        
        # Gera indices aleatórios
        indices_random = random.permutation(linha_x)
        # Divide os dados em treino e teste
        x_treino = X[indices_random[:split]]
        y_treino = Y[indices_random[:split]]
        
        x_teste = X[indices_random[split:]]
        y_teste = Y[indices_random[split:]]
        return x_treino, y_treino, x_teste, y_teste
    else:
        return None, None, None, None