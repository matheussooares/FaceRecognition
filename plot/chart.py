import matplotlib.pyplot as plt
from numpy import array

def histograma(X:array, xlabel:str = "Valores", ylabel:str = "Frequência",title:str ='Histograma', **kargs):
    plt.hist(x = X, **kargs)
    # Adicionando título e rótulos aos eixos
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Exibindo o gráfico
    plt.show()
    
def dispersao(X:array, Y:array, xlabel:str = "X", ylabel:str = "Y", title:str = 'Gráfico de Dispersão', **kargs):
    plt.scatter(
        x = X,
        y = Y,
        **kargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot(X:array, Y:array, xlabel:str = "X", ylabel:str = "Y", title:str = 'Gráfico', **kargs):
    plt.plot(
        X,
        Y,
        **kargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def barras(X:array, Y:array, xlabel:str = "X", ylabel:str = "Y", title:str = 'Gráfico de Barras', **kargs):
    plt.bar(
        x = X,
        height = Y,
        **kargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()