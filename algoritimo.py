import numpy as np

# Função objetivo a ser minimizada (exemplo: esfera f(x) = sum(x_i^2))
def funcao_objetivo(x):
    return np.sum(x**2)

def algoritmo_vagalume(f_obj, dim, n_vagalumes, max_geracoes, lb, ub, alpha, beta0, gamma):
    # Inicialização da população
    populacao = np.random.uniform(lb, ub, (n_vagalumes, dim))
    intensidade = np.array([f_obj(ind) for ind in populacao])
    
    for geracao in range(max_geracoes):
        for i in range(n_vagalumes):
            for j in range(n_vagalumes):
                # Se o vagalume j é mais brilhante (menor custo) que o i
                if intensidade[j] < intensidade[i]:
                    r = np.linalg.norm(populacao[i] - populacao[j])
                    beta = beta0 * np.exp(-gamma * (r**2))
                    # Movimenta o vagalume i em direção a j
                    populacao[i] += beta * (populacao[j] - populacao[i]) + alpha * (np.random.rand(dim) - 0.5)
                    # Limita aos espaços de busca
                    populacao[i] = np.clip(populacao[i], lb, ub)
                    intensidade[i] = f_obj(populacao[i])
                    
    # Encontra o melhor resultado
    melhor_idx = np.argmin(intensidade)
    return populacao[melhor_idx], intensidade[melhor_idx]

# Parâmetros principais
dimensao = 2
n_pop = 20
geracoes = 50
limite_inf = -5.0
limite_sup = 5.0
alfa = 0.2     # Aleatoriedade
beta0 = 1.0    # Atratividade base
gama = 1.0     # Coeficiente de absorção da luz

melhor_posicao, melhor_custo = algoritmo_vagalume(
    funcao_objetivo, dimensao, n_pop, geracoes, limite_inf, limite_sup, alfa, beta0, gama
)

print("Melhor posição:", melhor_posicao)
print("Melhor custo:", melhor_custo)
