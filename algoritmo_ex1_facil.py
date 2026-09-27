import time
import numpy as np
import matplotlib.pyplot as plt

# 1. DADOS DO PROBLEMA (NÍVEL FÁCIL)
# Tempos de processamento das 30 tarefas
tempos_tarefas = np.array([
    12, 5, 9, 7, 4, 11, 8, 6, 10, 3,
    7, 9, 5, 6, 4, 8, 12, 3, 7, 8,
    3, 7, 8, 11, 4, 9, 10, 13, 5, 2
])

num_tarefas = len(tempos_tarefas)# 30
num_maquinas = 5 #5 máquinas idênticas


def calcular_makespan(solucao):
    """
    solucao: array com o ID da máquina (0 a 4) atribuída a cada tarefa.
    Retorna o Makespan (C_max) da alocação.
    """
    cargas_maquinas = np.zeros(num_maquinas)
    for id_tarefa, id_maquina in enumerate(solucao):
        cargas_maquinas[id_maquina] += tempos_tarefas[id_tarefa]
    return np.max(cargas_maquinas)


# ALGORITMO DO VAGA LUME DISCRETO
def algoritmo_vagalume_facil(n_vagalumes=30, iteracoes=100, alfa=0.5, gama=0.1, beta0=1.0):
    inicio_tempo = time.time()

    # inicialização aleatória da população 
    populacao = np.random.uniform(0, num_maquinas - 1e-3, (n_vagalumes, num_tarefas))
    
    # converte para inteiros
    intensidade = np.array([calcular_makespan(np.floor(ind).astype(int)) for ind in populacao])

    # registra o histórico da melhor solução para o gráfico de evolução
    historico_evolucao = []
    
    melhor_idx = np.argmin(intensidade)
    melhor_solucao = np.floor(populacao[melhor_idx]).astype(int)
    melhor_makespan = intensidade[melhor_idx]

    historico_evolucao.append(melhor_makespan)

    for t in range(iteracoes):
        for i in range(n_vagalumes):
            for j in range(n_vagalumes):
                # se o vaga-lume j é melhor (menor makespan) que i
                if intensidade[j] < intensidade[i]:
                    r = np.linalg.norm(populacao[i] - populacao[j])
                    beta = beta0 * np.exp(-gama * (r**2))
                    mutacao = alfa * (np.random.rand(num_tarefas) - 0.5)
                    
                    # movimentação contínua
                    populacao[i] = populacao[i] + beta * (populacao[j] - populacao[i]) + mutacao
                    populacao[i] = np.clip(populacao[i], 0, num_maquinas - 1e-3)
                    
                    #avaliação discreta
                    solucao_discreta = np.floor(populacao[i]).astype(int)
                    intensidade[i] = calcular_makespan(solucao_discreta)

                    if intensidade[i] < melhor_makespan:
                        melhor_makespan = intensidade[i]
                        melhor_solucao = solucao_discreta.copy()

        alfa *= 0.98  # Decaimento do fator de aleatoriedade
        historico_evolucao.append(melhor_makespan)

    tempo_execucao = time.time() - inicio_tempo
    return melhor_solucao, melhor_makespan, tempo_execucao, historico_evolucao


# executando e apresentando os resultados
solucao, makespan, tempo_exec, historico = algoritmo_vagalume_facil()

# organizando os resultados por máquina
maquinas = {m: [] for m in range(num_maquinas)}
cargas = np.zeros(num_maquinas, dtype=int)

for id_tarefa, id_maquina in enumerate(solucao):
    maquinas[id_maquina].append(id_tarefa + 1)  # Tarefas de 1 a 30
    cargas[id_maquina] += tempos_tarefas[id_tarefa]

print("#" * 80)
print("RESULTADOS - NÍVEL FÁCIL")
print("#" * 80)
print(f"b) Valor final do Makespan (C_max): {makespan}")
print(f"c) Tempo de execução: {tempo_exec:.4f} segundos\n")
print("a) Atribuição final de tarefas às máquinas:")
for m in range(num_maquinas):
    print(f"   Máquina {m + 1}: Tarefas {maquinas[m]} | Carga total: {cargas[m]}")

# d) gráfico da evolução da solução
plt.figure(figsize=(8, 4))
plt.plot(historico, color='blue', linewidth=2)
plt.title("Evolução do Makespan (Algoritmo do Vaga-lume - Nível Fácil)")
plt.xlabel("Iteração")
plt.ylabel("Makespan (C_max)")
plt.grid(True)
plt.tight_layout()
plt.show()