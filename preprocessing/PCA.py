from numpy import array, mean, cov, linalg, argsort, cumsum, zeros_like
from scipy import stats


class principalComponentAnalysis:
    def __init__(self, X:array):
        # Aplica o pca
        self.pca(X = X)
        pass
    
    def pca(self, X:array):
        # Centralizar os dados
        X_centered = X - mean(X, axis=0)
       
        # Matriz de covariância de X  
        cov_matrix = cov(X_centered.T)
        
        # Autovalores (w) e autovetores (v)
        w, V = linalg.eigh(cov_matrix)
        
        # Ordenar em ordem decrescente
        idx = argsort(w)[::-1]  # inverte a ordem para decrescente
        w_sorted = w[idx]
        V_sorted = V[:, idx]  # reorganiza colunas de V
        

        # Armazena os autovalores e autovetores ordenados
        self.Eigenvalue = w_sorted
        self.Eigenvector = V_sorted
    
    def explainedVariance(self):
        # Retorna a variância explicada por cada componente principal
        total_variance = sum(self.Eigenvalue)
        # Percentual da variância explicada
        percen_varianca = (self.Eigenvalue / total_variance) * 100
        # Variância acumulada
        return  cumsum(percen_varianca)
    
    def transform(self, X:array, n_components:int):
        # Projeta os dados nos primeiros n_components componentes principais
        X_centered = X - mean(X, axis=0)
        return X_centered @ self.Eigenvector[:, :n_components]

# X = array([
#     [2.5, 2.4],
#     [0.5, 0.7],
#     [2.2, 2.9],
#     [1.9, 2.2],
#     [3.1, 3.0],
#     [2.3, 2.7],
#     [2.0, 1.6],
#     [1.0, 1.1],
#     [1.5, 1.6],
#     [1.1, 0.9]
# ])
# pca = principalComponentAnalysis(X=X)
# X_pca = pca.transform(X=X, n_components=2)
# print(X_pca)
def boxCox(X:array):
    df_pca_yeojohnson = zeros_like(X)
    # Aplicar box-cox
    transformed_data, lambda_value = stats.yeojohnson(X)
    # Aplicar Yeo-Johnson em cada coluna (componente principal)
    for i in range(X.shape[1]):
        df_pca_yeojohnson[:, i], _ = stats.yeojohnson(X[:, i])
    return transformed_data, lambda_value