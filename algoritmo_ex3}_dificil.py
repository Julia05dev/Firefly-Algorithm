import numpy as np

def funcao_objetivo(x):
    # Função Esfera: o mínimo global é 0 em x = [0, 0, ...]
    return np.sum(x**2)

def algoritmo_vagalume(func_obj, dim, n_vagalumes, iteracoes, lb, ub, alfa, gama, beta0):
    # Inicializa a população de vaga-lumes aleatoriamente nos limites
    populacao = np.random.uniform(lb, ub, (n_vagalumes, dim))
    intensidade = np.array([func_obj(ind) for ind in populacao])
    
    # Melhor global encontrado
    melhor_idx = np.argmin(intensidade)
    melhor_posicao = populacao[melhor_idx].copy()
    melhor_valor = intensidade[melhor_idx]
    
    for t in range(iteracoes):
        for i in range(n_vagalumes):
            for j in range(n_vagalumes):
                # Se o vaga-lume j é mais brilhante (menor valor na minimização) que i
                if intensidade[j] < intensidade[i]:
                    # Distância euclidiana entre os vaga-lumes i e j
                    r = np.linalg.norm(populacao[i] - populacao[j])
                    # Atração diminui com a distância (absorção da luz gama)
                    beta = beta0 * np.exp(-gama * (r**2))
                    # Movimenta o vaga-lume i em direção a j + componente aleatória (alfa)
                    mutacao = alfa * (np.random.rand(dim) - 0.5)
                    populacao[i] = populacao[i] + beta * (populacao[j] - populacao[i]) + mutacao
                    # Garante que continua dentro dos limites
                    populacao[i] = np.clip(populacao[i], lb, ub)
                    # Atualiza a intensidade
                    intensidade[i] = func_obj(populacao[i])
                    
                    # Atualiza o melhor global
                    if intensidade[i] < melhor_valor:
                        melhor_valor = intensidade[i]
                        melhor_posicao = populacao[i].copy()
                        
        # Reduz o fator de aleatoriedade ao longo das gerações
        alfa *= 0.99
        
    return melhor_posicao, melhor_valor

# Parâmetros do Algoritmo
dimensoes = 3          # Número de variáveis (dimensões do problema)
n_vagalumes = 25       # Tamanho da população
n_iteracoes = 50       # Número de gerações/iterações
limite_inf = -5.0      # Limite inferior de busca
limite_sup = 5.0       # Limite superior de busca
alfa_inicial = 0.5     # Controle da aleatoriedade do movimento
gama = 1.0             # Coeficiente de absorção da luz
beta0 = 1.0            # Atratividade base na distância r=0

# Execução
melhor_pos, melhor_val = algoritmo_vagalume(
    funcao_objetivo, dimensoes, n_vagalumes, n_iteracoes, 
    limite_inf, limite_sup, alfa_inicial, gama, beta0
)

print(f"Melhor posição encontrada: {melhor_pos}")
print(f"Valor mínimo da função: {melhor_val}")
