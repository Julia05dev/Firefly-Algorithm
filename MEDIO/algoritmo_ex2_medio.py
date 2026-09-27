import time
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# 1. dados do problema (nível médio)
# tempos de processamento das 50 tarefas
tempos_tarefas = np.array([
    17, 9, 20, 12, 28, 16, 22, 15, 18, 22,
    19, 23, 11, 27, 14, 21, 17, 24, 26, 15,
    13, 10, 15, 16, 22, 18, 21, 20, 23, 17,
    23, 20, 22, 16, 18, 12, 26, 14, 20, 11,
    20, 18, 23, 15, 17, 8, 21, 14, 23, 12
])

# capacidades das 6 máquinas
capacidades_maquinas = np.array([18, 22, 25, 16, 28, 14])

num_tarefas = len(tempos_tarefas) # 50
num_maquinas = 6 # 6 máquinas com capacidades diferentes


def calcular_makespan(solucao):
    """
    solucao: array com o id da máquina (0 a 5) atribuída a cada tarefa.
    retorna o makespan (c_max) da alocação considerando as capacidades.
    """
    cargas_maquinas = np.zeros(num_maquinas)
    for id_tarefa, id_maquina in enumerate(solucao):
        cargas_maquinas[id_maquina] += tempos_tarefas[id_tarefa]
        
    # calcula o tempo efetivo dividindo a soma das tarefas pela capacidade
    tempos_efetivos = cargas_maquinas / capacidades_maquinas
    return np.max(tempos_efetivos)


# algoritmo do vaga lume discreto
def algoritmo_vagalume_medio(n_vagalumes=40, iteracoes=150, alfa=0.5, gama=0.1, beta0=1.0):
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
                    
                    # avaliação discreta
                    solucao_discreta = np.floor(populacao[i]).astype(int)
                    intensidade[i] = calcular_makespan(solucao_discreta)

                    if intensidade[i] < melhor_makespan:
                        melhor_makespan = intensidade[i]
                        melhor_solucao = solucao_discreta.copy()

        alfa *= 0.98  # decaimento do fator de aleatoriedade
        historico_evolucao.append(melhor_makespan)

    tempo_execucao = time.time() - inicio_tempo
    return melhor_solucao, melhor_makespan, tempo_execucao, historico_evolucao


# executando e apresentando os resultados
solucao, makespan, tempo_exec, historico = algoritmo_vagalume_medio()

# organizando os resultados por máquina
maquinas = {m: [] for m in range(num_maquinas)}
cargas = np.zeros(num_maquinas, dtype=int)

for id_tarefa, id_maquina in enumerate(solucao):
    maquinas[id_maquina].append(id_tarefa + 1)  # tarefas de 1 a 50
    cargas[id_maquina] += tempos_tarefas[id_tarefa]
    
# tempos efetivos para a impressão final
tempos_efetivos_finais = cargas / capacidades_maquinas

print("#" * 80)
print("resultados - nível médio")
print("#" * 80)
print(f"b) valor final do makespan (c_max): {makespan:.4f}")
print(f"c) tempo de execução: {tempo_exec:.4f} segundos\n")
print("a) atribuição final de tarefas às máquinas:")
for m in range(num_maquinas):
    print(f"   máquina {m + 1} (cap: {capacidades_maquinas[m]}): tarefas {maquinas[m]}")
    print(f"      -> carga total: {cargas[m]} | tempo efetivo: {tempos_efetivos_finais[m]:.4f}")

# d) gráfico da evolução da solução
plt.figure(figsize=(8, 4))
plt.plot(historico, color='blue', linewidth=2)
plt.title("evolução do makespan (algoritmo do vaga-lume - nível médio)")
plt.xlabel("iteração")
plt.ylabel("makespan (c_max)")
plt.grid(True)
plt.tight_layout()
pasta_script = os.path.dirname(os.path.abspath(__file__))
caminho_imagem = os.path.join(pasta_script, "evolucao_medio.png")
plt.savefig(caminho_imagem)
plt.show()