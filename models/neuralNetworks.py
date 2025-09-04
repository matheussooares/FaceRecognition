from numpy import array, random, concatenate, ones, mean, empty, vstack, tanh, median, std, append
from models.activationFunction import step, sigmoid, d_step, d_sigmoid, d_tanh
from preprocessing.splitdata import randomSplit, randomOrder
from metrics.regressor import meanSquaredError, coefficientDetermination, erroPredict
from metrics.classification import accuracy



# Classe da rede percetron logística
class LogisticPerceptronNetwork:
    def __init__(self, learning_rate:float =1e-2, weight_scale:float =1e-1, test_size:float = 0.2, max_iter:int = 1000000, tol: float = 1e-4, ativation:str = "sigmoid", localGradientConstant:float = 0.05, model:str = 'regressor', solver:str = "classico"):
        # Armazena o tipo de rede
        self.model = model
        # Escolhe algoritmo de aprendizado
        self.solver = solver
        # Passo de aprendizado
        self.learning_rate = learning_rate
        # Escala de inicialização dos pesos sinápticos
        self.weight_scale = weight_scale
        # Porcentagem de dados de teste
        self.test_size = test_size
        # Número máximo de épocas
        self.max_iter = max_iter
        # Nome da função de ativação
        self.ativation = ativation
        # Tolerância do erro para convergência
        self.tol = tol
        # Pesos sinápticos
        self.W = None
        # Número neuronios  de saídas
        self.q = None
        # Constante do gradiente local
        self.localGradientConstant = localGradientConstant
        # Função de ativação
        self.init_functionActivation(ativation)
        # Armazena as métricas dos modelos
        self.init_metrics()
        # Escolhe o modelo
        self.init_model()
        
    
    # Inicializa o método 
    def init_metrics(self):
        # Curva de aprendizado
        self.loss = array([])
        if self.model == "regressor":
            self.metrics = {
                "R2":array([])
            }
        elif self.model == "classifier":
            self.metrics = {
                "acuracia":array([])
            }
            
    
    def init_model(self):
        # Para regressão as predições seguem o método regressor
        if self.model == "regressor":
            self.predict = self.regressor
        elif self.model == "classifier":
            self.predict = self.classification
    
    # Escolhe a função de ativação
    def init_functionActivation(self, fun_ativation):
        # Seleciona a função de ativação
        if fun_ativation == "step":
            self.functionActivation = step
        elif fun_ativation == "sigmoid":
            self.functionActivation = sigmoid
        elif fun_ativation == "tanh":
            self.functionActivation = tanh
    
    # Inicializa os pesos sinápticos aleatoriamente
    def init_weights(self, q:int, p:int, weight_scale:float):
        # Pesos sinápticos aleatoriamente
        self.W = weight_scale*random.rand(q,p+1)
       
    # Regra de aprendizado
    def learningRule(self, d, y, X):      
        # calcula o erro entre a saída desejada e a saída prevista
        erro_i = erroPredict(
            Y=d, 
            Y_pred=y
        )
        # Adiciona o bias nas amostras
        x =  concatenate((-1*ones((1,1)), X), axis=0)
        # Aplica o aprendizado
        if self.solver == "deltaGeneralizado":
            # Derivada da função de ativação do gradiente local
            dev_y = self.derivativeActivation(y)
            # Aplica o aprendizado nos pesos
            self.W = self.W + self.learning_rate * (erro_i * dev_y) @ x.T
        elif self.solver == "classico":
            self.W = self.W + self.learning_rate * erro_i @ x.T
    
    # saídas das ativações dos neuronios
    def neuronActivation(self, X):
        # adiciona o bias nas amostras
        x =  concatenate((-1*ones((1,1)), X), axis=0)
        # Calcula o acumulo energético dos neurônios
        U = self.W@x
        # Saída dos neurionios por meio da função de ativação
        y = self.functionActivation(U)
        return y
    
    # Função que calcula a derivada da função de ativação
    def derivativeActivation(self, Y):
        if self.ativation == "sigmoid":
            return d_sigmoid(Y)
        elif self.ativation == "tanh":
            return d_tanh(Y)
        elif self.ativation == "step":
            return d_step(Y)
    
    # Função que calcula as métricas do modelo 
    def metricsFit(self, Y:array, Y_prev:array):
        # Gera as métricas
        if self.model == "regressor":
            # Erro médio quadratico
            mse = meanSquaredError(
                Y=Y, 
                Y_pred=Y_prev
            )*1/2
            
            # Coeficiente de determinação
            R2 = coefficientDetermination(
                Y=Y, 
                Y_pred=Y_prev
            )
            self.loss = append(self.loss, mse)
            r2 = append(self.metrics["R2"], R2)
            self.metrics["R2"] = r2
        elif self.model == "classifier":
            acc = accuracy(
                Y=Y, 
                Y_pred=Y_prev
            )
            self.loss = append(self.loss, acc)
            self.metrics["acuracia"] = self.loss
    
    # Treinamento da rede neural
    def fit(self, X:array, Y:array):
        # Calcula o número de atributos dos dados do modelo
        m, p = X.shape
        n, q = Y.shape
        
        # Armazena o número de saída da rede
        self.q = q
    
                              
        # Inicializa os pesos sinápticos
        self.init_weights(q, p, self.weight_scale)
                
        # Treina o modelo em cada época
        for epoca in range(1,self.max_iter+1):         
            # Modifica a ordem dos dados
            x_treino, y_treino = randomOrder(
                X = X, 
                Y = Y
            )
            
            # Número de amostras do conjunto de treino
            num_amostras = x_treino.shape[0]
            
            # Inicializa a matriz que armazena todas as predições do modelo  
            y_predic = empty((num_amostras, self.q))
            
            # Treina a rede com todos os dados de treino
            for t in range(num_amostras):
                # Seleciona a vetor de atributos da amostra atual 
                x = x_treino[t:t+1,:].T
                
                # Seleciona o vetor de saída real da amostra atual
                d = y_treino[t:t+1,:].T
                
                # Calcula o acumulo energético dos neurônios
                y = self.neuronActivation(x)
                # Calcula o passo de aprendizado
                self.learningRule(d,y, x)
                
                # Armazena a saída atual
                y_predic[t,:] =  y.T
            
            # Gera as métricas
            self.metricsFit(
                Y = y_treino, 
                Y_prev = y_predic)   

            print(f'{epoca}ª epoca de treinamento, Erro {self.loss[-1]}')
            
            
            if  self.convergenceCriteria() == True:
                print(f"Convergência atingida na época {epoca}")
                break
    
    def convergenceCriteria(self):
        if self.model == "regressor":
            if  self.loss[-1]<= self.tol:
                return True
            else:
                return False
        elif self.model == "classifier":
            if  self.loss[-1]>= self.tol:
                return True
            else:
                return False
    
    # Predição da rede neural
    def regressor(self, X):
        # Pega o número de amostras
        num_amostras = X.shape[0]
        # Inicializa as amostras   
        saida = empty((0, self.q))
        # Percorre todas as amostras
        for t in range(num_amostras):
            x = X[t:t+1,:].T
            # Calcula o acumulo energético dos neurônios
            y = self.neuronActivation(x)
            # Pega e armazena as saídas
            saida = vstack((saida, y.T))
            
        return saida
    
    # Calcula a acuracia de modelo em classificação
    def classification(self, X:array):
        # Pega o número de amostras
        num_amostras = X.shape[0]
        # Inicializa as amostras   
        saida = empty((0, self.q))
        # Percorre todas as amostras
        for t in range(num_amostras):
            x = X[t:t+1,:].T
            # Calcula o acumulo energético dos neurônios
            y = self.neuronActivation(x)
            # Pega e armazena as saídas
            saida = vstack((saida, y.T))
        return saida




class multilayerPerceptronNetwork():
    def __init__(self, hidden: dict = (2,), learning_rate:float =1e-2, weight_scale:float =1e-1, test_size:float = 0.2, max_iter:int = 1000000, tol: float = 1e-4, ativation:str = "sigmoid", localGradientConstant:float = 0.05, model:str = "regressor"):
        # Armazena o tipo de modelo
        self.model = model
        #Camadas ocultas
        self.hidden = hidden
        # Passo de aprendizado
        self.learning_rate = learning_rate
        # Escala de inicialização dos pesos sinápticos
        self.weight_scale = weight_scale
        # Porcentagem de dados de teste
        self.test_size = test_size
        # Número máximo de épocas
        self.max_iter = max_iter
        # Nome da função de ativação
        self.ativation = ativation
        # Tolerância do erro para convergência
        self.tol = tol
        # Número neuronios  de saídas
        self.q = None
        # Constante do gradiente local
        self.localGradientConstant = localGradientConstant
        # Função de ativação
        self.init_functionActivation(ativation)
        # Armazena as métricas dos modelos
        self.init_metrics()
        # Escolhe o modelo
        self.init_model()
        
    # Inicializa o método 
    def init_metrics(self):
        # Curva de aprendizado
        self.loss = array([])
        if self.model == "regressor":
            self.metrics = {
                "R2":array([])
            }
        elif self.model == "classifier":
            self.metrics = {
                "acuracia":array([])
           }
    # Inicializa o tipo de modelo regressão/classificação
    def init_model(self):
        # Para regressão as predições seguem o método regressor
        if self.model == "regressor":
            self.predict = self.regressor
        elif self.model == "classifier":
            self.predict = self.classification
            
    # Inicializa escolha da função de ativação
    def init_functionActivation(self, fun_ativation):
        if fun_ativation == "step":
            self.functionActivation = step
        elif fun_ativation == "sigmoid":
            self.functionActivation = sigmoid
        elif fun_ativation == "tanh":
            self.functionActivation = tanh
        else:
            raise ValueError("Função de ativação não suportada")
    
    # Função que calcula a derivada da função de ativação
    def derivativeActivation(self, Y):
        if self.ativation == "sigmoid":
            return d_sigmoid(Y)
        elif self.ativation == "tanh":
            return d_tanh(Y)
        elif self.ativation == "step":
            return 0  
    
    
    # Inicializa os pesos sinápticos aleatoriamente
    def init_weights(self, q:int, p:int):
        # Lista de pesos sinápticos
        W = list()
        # Número de camadas ocultas
        num_hidden_layers = len(self.hidden)
        self.num_hidden_layers = num_hidden_layers
        # numero de atributos mais o bies
        atributos = p + 1
        # Itera sobre as camadas criando os pesos sinápticos
        for i in range(num_hidden_layers+1):
            
            if i == num_hidden_layers: #  Última camada
                # Neuronios de saída igual ao número de saídas
                neuronios = q
            else: # Demais camadas ocultas
                # Neuronios da camada oculta conforme configuração
                neuronios = self.hidden[i]
            
            if i != 0: # atributos da camada oculta
                # A camada oculta tem o número de atributos + 1 (bias)
                atributos = self.hidden[i-1] + 1
    
            # inicializa os pesos sinápticos aleatoriamente da camada
            W.append(
                self.weight_scale*(2.0 * random.rand(neuronios,atributos) - 1.0)
            )
        # Armazena os pesos sinápticos de todas as camadas     
        self.weights = W
        return True
    
    # Função de ativação dos neurônios
    def neuronActivation(self, X, camada_n):
        # adiciona o bias nas amostras
        x =  concatenate((-1*ones((1,1)), X), axis=0)
        # Calcula o acumulo energético dos neurônios na camada n
        U = self.weights[camada_n]@x
        # Saída dos neurionios por meio da função de ativação
        y = self.functionActivation(U)
        return y       
    
    # Calcula a acuracia de modelo em classificação
    def classification(self, X:array):
        # Analisa o número de amostras
        num_amostras = X.shape[0]
        # Inicializa as saídas
        saida = empty((0, self.q))
        # Percorre os dados de entrada
        for t in range(num_amostras):
            # seleciona a vetor de atributos da amostra atual 
            x = X[t:t+1,:].T
            # Calcula o acumulo energético dos neurônios
            y = self.forwardDirection(x)[-1]
            saida = vstack((saida, y.T))            
        return saida

    # Função que calcula as métricas do modelo 
    def metricsFit(self, Y:array, Y_prev:array):
        # Gera as métricas
        if self.model == "regressor":
            # Erro médio quadratico
            mse = meanSquaredError(
                Y=Y, 
                Y_pred=Y_prev
            )*1/2
            
            # Coeficiente de determinação
            R2 = coefficientDetermination(
                Y=Y, 
                Y_pred=Y_prev
            )
            self.loss = append(self.loss, mse)
            r2 = append(self.metrics["R2"], R2)
            self.metrics["R2"] = r2
        elif self.model == "classifier":
            acc = accuracy(
                Y=Y, 
                Y_pred=Y_prev
            )
            self.loss = append(self.loss, acc)
            self.metrics["acuracia"] = self.loss
    
    # Treinamento da rede neural
    def fit(self, X, Y):
        # Calcula o tamanho dos atributos do modelo
        p = X.shape[1]
        # Calcula o tamanho da saída do modelo
        q = Y.shape[1]
        self.q = q        
        # Inicializa os pesos sinápticos aleatoriamente
        self.init_weights(q, p)
        
        # Aprendizado por epocas
        for epoca in range(1,self.max_iter+1):
            
            # Modifica a ordem dos dados
            x_treino, y_treino = randomOrder(
                X = X, 
                Y = Y
            )
            
            # Numero de amostras
            num_amostras_treino = x_treino.shape[0]
            
            # Inicializa a matriz que armazena todas as predições do modelo  
            y_predic = empty((num_amostras_treino, self.q))
            
            # Percorre o número de amostas de treino
            for t in range(num_amostras_treino):
                # seleciona a vetor de atributos da amostra atual 
                x = x_treino[t:t+1,:].T
                # Seleciona o vetor de saída real da amostra atual
                d = y_treino[t:t+1,:].T               
                # Calcula as ativações de cada camada
                Z = self.forwardDirection(X=x)
                # Armazena a camada de saída
                y_predic[t,:] = Z[-1].T
                
                # Propagação dos erros no sentido inverso
                self.backwardDirection(Z, d, self.learning_rate )
             
            # Calcula as metricas do modelo 
            self.metricsFit(
                Y = y_treino, 
                Y_prev = y_predic
            )
            print(f'{epoca}ª epoca de treinamento, Erro {self.loss[-1]}')
            
            # Analisa critério de converência
            if  self.convergenceCriteria() == True:
                print(f"Convergência atingida na época {epoca}")
                break
    
    def convergenceCriteria(self):
        if self.model == "regressor":
            if  self.loss[-1]<= self.tol:
                return True
            else:
                return False
        elif self.model == "classifier":
            if  self.loss[-1]>= self.tol:
                return True
            else:
                return False
                 
    # Função de propagação do erro no sentido inverso das camadas
    def backwardDirection(self, Z, d, eta):
        delta = 0
        # Percorre as camadas no sentido inverso
        for camada in range(self.num_hidden_layers, -1, -1):
            if camada == self.num_hidden_layers:
                # Erro da saída
                erro = (d - Z[camada+1])            
                #  Derivada da função de ativação dos neuronios da saída
                dy = self.derivativeActivation(Z[camada+1]) + 0.01
                # gradiente local dos neuronios de saida
                delta_y = erro * dy        
                # Adiciona o bais no vetor de entradas
                z = concatenate((-1*ones((1,1)), Z[camada]), axis=0)
                # Gradiente atual é atualizado
                delta = delta_y
            
            else:
                # Calcula a deridava da saída            
                dz = self.derivativeActivation(Z[camada+1]) + 0.01
                # Pega os pesos anteriores para calcular o gradiente local
                m_anterior = self.weights[camada+1][:,1:].T
                # Calcula o gradiente local dos neuronios
                delta_z = dz * (m_anterior @ delta)
                # Adiciona os bais no vetor de entrada
                z = concatenate((-1*ones((1,1)), Z[camada]), axis=0)
                # Gradiente atual é atualizado
                delta = delta_z
            
            # Aplica a regra de aprendizado nos pesos            
            self.weights[camada] = self.weights[camada] + self.learning_rate*(delta@z.T)    

    # Função de propragação das entradas no sentido para frente das camadas
    def forwardDirection(self,X):
        # Lista de saídas de cada camada
        Z = list()
        # Percorre as camadas ocultas e calcula as ativações
        for camada in range(self.num_hidden_layers+1):
            if camada == 0: # primeira camada
                # Inicializa a primeira camada com os dados de entrada
                x = X
                Z.append(x)
            else: # A camada oculta recebe a saída da camada anterior 
                x = z  
            # Calcula a ativação da camada n com a entrada
            z = self.neuronActivation(X = x, camada_n = camada)            
            # Armazena as ativações de cada camada
            Z.append(z)
        return Z
    
    def regressor(self, X:array):
        # Analisa o número de amostras
        num_amostras = X.shape[0]
        # Inicializa as saídas
        saida = empty((0, self.q))
        # Percorre os dados de entrada
        for t in range(num_amostras):
            # seleciona a vetor de atributos da amostra atual 
            x = X[t:t+1,:].T
            # Calcula o acumulo energético dos neurônios
            y = self.forwardDirection(x)[-1]
            saida = vstack((saida, y.T))            
        return saida