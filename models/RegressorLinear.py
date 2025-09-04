from numpy import array, linalg, eye, concatenate, ones
from metrics.regressor import coefficientDetermination, meanSquaredError
from metrics.classification import accuracy, correctOneHotEncod

class MultipleRegressorLinear:
    def __init__(self, model:str = "regressor", regularizer:float=1e-5):
        # Seleciona classificação ou regressão
        self.model = model
        # Parametro de regularização do método dos mínimos quadráticos
        self.regularizer = regularizer
        # Escolhe o modelo
        self.init_model()
        
    def init_model(self):
        # Para regressão as predições seguem o método regressor
        if self.model == "regressor":
            self.predict = self.regressor
        elif self.model == "classifier":
            self.predict = self.classifier         
              
    # Calcula os regressores usando o método dos mínimos quadráticos
    def OrdinaryLeastSquares(self, X:array,  Y:array):
        # Calcula linha e coluna dos atributos
        num_linhas = X.shape[0]
        # Adiciona o bias
        X = concatenate((ones((num_linhas, 1)), X), axis=1)
        # Calcula o produto matricial 
        xtx = X.T @ X
        
        # Matriz com o coeficinte de regularização
        linha, coluna = xtx.shape
        # Gera os lambidas regularizados (Regularização de Thikonov)
        lambda_regularizer = eye(linha, coluna) * self.regularizer
        
        # Calcula os regressores usando o método dos minimos quadráticos
        beta = linalg.inv(xtx + lambda_regularizer) @ (X.T @ Y)
        self.beta =  beta
        # Armazena o número de regressores
        self.p = beta.shape[0]
    
    # Classificador Linear
    def classifier(self, X:array):
        # Calcula linha e coluna dos atributos
        num_linhas = X.shape[0]
        # Adiciona o bias
        x = concatenate((ones((num_linhas, 1)), X), axis=1)
        # Calcula as predições
        y = x @ self.beta
        return y
    
    # Regressor Linear
    def regressor(self, X:array):
        # Calcula linha e coluna dos atributos
        num_linhas = X.shape[0]
        # Adiciona o bias
        x = concatenate((ones((num_linhas, 1)), X), axis=1)
        # Calcula as predições
        y = x @ self.beta
        return y
    
    # Calcula as métricas do modelo
    def metricsfit(self, Y:array, Y_prev:array):
        self.metrics = dict()
        # Gera as métricas
        if self.model == "regressor":
            mse = meanSquaredError(
                Y=Y, 
                Y_pred=Y_prev
            )
            R2 = coefficientDetermination(
                Y=Y, 
                Y_pred=Y_prev
            )
        
            self.metrics["MSE"] = mse
            self.metrics["R2"] = R2  
        elif self.model == "classifier":
            acc = accuracy(
                Y=Y, 
                Y_pred=Y_prev
            )
            self.metrics["accuracy"] = acc     
    
    # Treina o modelo
    def fit(self, X:array,  Y:array):
        
        # Calcula os regressores
        self.OrdinaryLeastSquares(
            X = X,  
            Y = Y
        )
        
        # Predição das amostras de entrada
        y_predict = self.predict(X=X)
        
        # Calcula as metricas
        self.metricsfit(
            Y = Y, 
            Y_prev = y_predict
        )
        

