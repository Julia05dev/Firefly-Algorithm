import time
import numpy as np
#import matplotlib.pyplot as plt
import os

np.random.seed(42)

# 1. dados do problema (nível difícil)
# tempos de processamento das 40 tarefas
tempos_tarefas = np.array([
    25, 17, 20, 12, 28, 16, 22, 15, 18, 30,
    19, 23, 11, 27, 14, 21, 17, 24, 26, 19,
    13, 10, 15, 28, 22, 18, 21, 30, 23, 17,
    25, 20, 22, 16, 18, 12, 26, 14, 27, 11
])

prioridade = {
    0: [1, 2],    
    2: [3, 4],    
    4: [5],       
    7: [8],       
    9: [10],      
    11: [12],     
    14: [15],     
    16: [17],     
    18: [19],     
    20: [21],
    22: [23],     
    24: [25],     
    26: [27],     
    28: [29],     
    30: [31],     
    32: [33],     
    34: [35],     
    36: [37],     
    38: [39]      
}


num_tarefas = len(tempos_tarefas) # 40
num_maquinas = 5 # 5 máquinas


def calcular_makespan(solucao):
    """
    solucao: array com o id da máquina (0 a 4) atribuída a cada tarefa.
    retorna o makespan (c_max) da alocação considerando as capacidades.
    """
    # grau de entrada de cada tarefa, garantir que todas as prioridades já estão feitas
    grau_entrada = np.zeros(num_tarefas, dtype=int)
    dependentes = {i: [] for i in range(num_tarefas)}
    
    for u, sucs in prioridade.items():
        for v in sucs:
            grau_entrada[v] += 1
            dependentes[u].append(v)
            
    tempo_termino_tarefas = np.zeros(num_tarefas)
    tempo_disponivel_maquina = np.zeros(num_maquinas)
    tarefas_prontas = [i for i in range(num_tarefas) if grau_entrada[i] == 0]
    processadas = 0
    ordem_execucao = list(range(num_tarefas))
    
    while processadas < num_tarefas:
        # verificar tarefas que ainda não foram feitas
        candidatas = [t for t in tarefas_prontas if t in ordem_execucao]
        if not candidatas:
            break
            
        u = candidatas[0]
        ordem_execucao.remove(u)
        processadas += 1
        
        m = solucao[u]
        
        tempo_inicio = tempo_disponivel_maquina[m]
        for pred, sucs in prioridade.items():
            if u in sucs:
                tempo_inicio = max(tempo_inicio, tempo_termino_tarefas[pred])
                
        tempo_termino = tempo_inicio + tempos_tarefas[u]
        tempo_termino_tarefas[u] = tempo_termino
        tempo_disponivel_maquina[m] = tempo_termino
        
        for v in dependentes[u]:
            grau_entrada[v] -= 1
            if grau_entrada[v] == 0:
                tarefas_prontas.append(v)
                
    return np.max(tempo_disponivel_maquina)


# algoritmo do vagalume discreto
def algoritmo_vagalume_dificil(n_vagalumes=40, iteracoes=150, alfa=0.5, gama=0.1, beta0=1.0):
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
solucao, makespan, tempo_exec, historico = algoritmo_vagalume_dificil()

# organizando os resultados por máquina
maquinas = {m: [] for m in range(num_maquinas)}
cargas = np.zeros(num_maquinas, dtype=int)

for id_tarefa, id_maquina in enumerate(solucao):
    maquinas[id_maquina].append(id_tarefa + 1)  # tarefas de 1 a 40
    cargas[id_maquina] += tempos_tarefas[id_tarefa]
    
# tempos efetivos para a impressão final
tempos_efetivos_finais = cargas

print("#" * 80)
print("resultados - nível difícil")
print("#" * 80)
print(f"b) valor final do makespan (c_max): {makespan:.4f}")
print(f"c) tempo de execução: {tempo_exec:.4f} segundos\n")
print("a) atribuição final de tarefas às máquinas:")
for m in range(num_maquinas):
    print(f"   máquina {m + 1}: tarefas {maquinas[m]}")
    print(f"      -> carga total: {cargas[m]} | tempo efetivo: {tempos_efetivos_finais[m]:.4f}")

# d) gráfico da evolução da solução
