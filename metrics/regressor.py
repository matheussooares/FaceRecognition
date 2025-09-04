from numpy import array, sum, mean, std
from scipy.stats import kstest
def quadraticError(error:array):
    return 0.5 * (error**2)

# Erro entre o valore predito e o real
def erroPredict(Y:array, Y_pred:array):
    return Y-Y_pred

# Calcula o erro quadrático
def squaredError(Y:array, Y_pred:array):
    return (erroPredict(Y, Y_pred))**2

# Calcula a soma dos quadrados dos erros
def sumSquaredError(Y:array, Y_pred:array, axis:int = None):
    return sum(
        squaredError(
            Y=Y,
            Y_pred=Y_pred
        ),
            axis=axis
    )

# Calcula o erro médio quadrático
def meanSquaredError(Y:array, Y_pred:array):
    # Soma dos erros ao quadrado acumulados de todas as saidas
    erro_y = sumSquaredError(
        Y = Y, 
        Y_pred = Y_pred, 
        axis=1)
    # Calcula a média dos erros quadráticos
    return mean(erro_y)
    
    

# Calcula a estimativa não-polarizada do erro quarático
def unbiasedVarianceSquaredError(Y:array, Y_pred:array, axis:int = None):
    # Calcula a soma dos erros quadráticos
    SQE = sumSquaredError(
        Y=Y,
        Y_pred=Y_pred,
        axis = axis
    )
    # Número de amostras
    num_amostras = Y.shape[0]
    # Calcula a estimativa não-polarizada do erro quarático
    return (SQE/(num_amostras-2))

# Coeficiente de determinação
def coefficientDetermination(Y:array, Y_pred:array, adjusted:bool= False, p:int = 0):
    # Calcula a soma dos erros quadráticos
    SQe = sumSquaredError(
        Y=Y,
        Y_pred=Y_pred
    )
    # Calcula o total da variabilidade dos saídas em torno de sua média
    Syy = sum(
        (Y - mean(Y, axis=0))**2
    )
    if adjusted == True:
        # Calcula o numero de amostras
        num_amostras = Y.shape[0]
        print(num_amostras)
        return 1 - (SQe/(num_amostras-p))/(Syy/(num_amostras-1))
    elif adjusted == False:
        return 1 - (SQe/Syy)

# Calcula o teste de Kolmogorov-Smirnov
def metodoKolmogorovSmirnov(residuos:array):
    # Normalizar os resíduos para ter variância 1
    residuos_norm = residuos / std(residuos)
    # Realizar o teste de Kolmogorov-Smirnov
    H, p_value = kstest(residuos_norm.flatten(), 'norm')
    return H, p_value
    
