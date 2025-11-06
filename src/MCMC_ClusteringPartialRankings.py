#ALGORITHM 4 — MCMC for Clustering Partial Rankings (from the article)
#
#Input:
#    - Observed rankings π₁, ..., π_N (possibly partial)
#    - Number of clusters C
#    - Initial parameters {ρ_c, α_c, z_j}
#
#Repeat for each MCMC iteration:
#    1. (Gibbs step) Update cluster assignments z_j:
#        For each ranking π_j:
#            Compute P(z_j = c | π_j, ρ_c, α_c) ∝ exp(-α_c * d(π_j, ρ_c))
#            Sample z_j from this distribution.
#
#    2. (Metropolis–Hastings) Update consensus rankings ρ_c:
#        For each cluster c:
#            Propose new ρ_c' using "leap-and-shift" move.
#            Accept or reject based on posterior probability ratio.
#
#    3. (Optional) Update dispersion parameters α_c:
#        Sample α_c from posterior given cluster assignments.
#
#Until convergence.
#
#Output:
#    - Estimated consensus rankings ρ_c
#    - Cluster assignments z_j
#    - Dispersion parameters α_c

import numpy as np
import random
from itertools import permutations
from collections import Counter

# Função de distância Kendall
# (d(π_j, ρ_c) no pseudoalgoritmo)
def kendall_distance(r1, r2):
    common_items = [x for x in r1 if x in r2]
    pairs = 0
    n = len(common_items)
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = common_items[i], common_items[j]
            # compara apenas se ambos existem em r2
            if a1 in r2 and a2 in r2:
                if (r1.index(a1) - r1.index(a2)) * (r2.index(a1) - r2.index(a2)) < 0:
                    pairs += 1
    return pairs

# Geração de rankings sintéticos — simula π_j ~ Mallows(ρ_c, α_c)
# (equivalente à parte "simulate rankings" do artigo)
def generate_mallows_centered(center, alpha, n_samples, top_k=None):
    n = len(center)
    rankings = []
    for _ in range(n_samples):
        # Perturbação proporcional a alpha
        perm = center.copy()
        n_swaps = np.random.poisson(alpha)
        for _ in range(n_swaps):
            i, j = np.random.choice(range(n), 2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        if top_k:
            perm = perm[:top_k]
        rankings.append(perm)
    return rankings

# Cálculo de ranking de consenso ρ_c — aproximação determinística
# (substitui o passo Metropolis–Hastings do artigo)
def consensus_ranking(rankings, all_items):
    scores = Counter()
    for r in rankings:
        for pos, item in enumerate(r):
            scores[item] += len(all_items) - pos
    ordered = [x for x, _ in scores.most_common()]
    return ordered

# Definição dos pilotos e parâmetros da simulação
pilotos = ["VER", "NOR", "LEC", "HAM", "RUS", "PIA", "SAI", "PER", "ALO", "STR"]
n_pilotos = len(pilotos)
n_corridas = 30
top_k = 8  # ranking parcial

# Dois clusters com consensos distintos
consenso_cluster1 = pilotos.copy()  # pilotos rápidos
consenso_cluster2 = pilotos[::-1]  # pilotos técnicos

# Geração dos rankings simulados — π_j
rankings_cluster1 = generate_mallows_centered(consenso_cluster1, alpha=2, n_samples=15, top_k=top_k)
rankings_cluster2 = generate_mallows_centered(consenso_cluster2, alpha=2, n_samples=15, top_k=top_k)
rankings = rankings_cluster1 + rankings_cluster2
true_labels = [0] * 15 + [1] * 15  # rótulos verdadeiros (somente para checagem)

# Inicialização dos parâmetros (ρ_c, α_c, z_j)
C = 2  # número de clusters
assignments = np.random.choice(C, n_corridas)
clusters = {c: [] for c in range(C)}

# LOOP MCMC — Etapas 1 e 2 do pseudoalgoritmo
for iteration in range(50):  # iterações do "MCMC"

    # (1) Atualizar atribuições de cluster z_j - Gibbs step
    clusters = {c: [] for c in range(C)}
    for i, r in enumerate(rankings):
        distances = []
        for c in range(C):
            if clusters[c]:
                rho = consensus_ranking(clusters[c], pilotos)
                d = kendall_distance(r, rho)
            else:
                d = np.random.rand()
            distances.append(d)
        new_cluster = np.argmin(distances)  # escolhe cluster com menor distância
        assignments[i] = new_cluster
        clusters[new_cluster].append(r)

    # Atualizar rankings de consenso ρ_c - MH step (simplificado)
    consensos = []
    for c in range(C):
        if clusters[c]:
            rho = consensus_ranking(clusters[c], pilotos)
            consensos.append(rho)
        else:
            # se cluster estiver vazio, inicializa aleatoriamente
            consensos.append(random.sample(pilotos, len(pilotos)))

# Exibir os 30 rankings simulados (π_j)
print("\nCorridas simuladas (Rankings π_j)")
for i, r in enumerate(rankings):
    cluster_real = "Cluster 1 (pistas rápidas)" if true_labels[i] == 0 else "Cluster 2 (pistas técnicas)"
    print(f"Corrida {i+1:02d} | {cluster_real}: {' > '.join(r)}")

# Saída - equivalente ao "posterior mean" do artigo
print("\nResultados do MCMC F1")
for c, rho in enumerate(consensos):
    print(f"\nCluster {c + 1} - Ranking de consenso estimado:")
    print(" > ".join(rho[:top_k]))

print("\nAtribuições finais das corridas:")
print(assignments) # 0 -> Cluster 1 (Rápida) e 1 -> Cluster 2 (Técnica)
