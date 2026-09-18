#!/usr/bin/env python3
"""detect_redites.py — Detection de redites semantiques entre articles.
Utilise all-MiniLM-L6-v2 (sentence-transformers) + DBSCAN.
Seuil de similarite cosinus = 0.85 pour eliminer les faux positifs.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

SEUIL = 0.85

articles = {
    'securite-mas': 'Securite des systemes multi-agents LLM : Taxonomie A-I-R et defis ouverts. Le SoK 2609.00595 systematise la securite des systemes multi-agents LLM via le cadre A-I-R (Adversaire-Interface-Risque). Adversaire : 4 positions identifiees. Interface : 6 interfaces d interaction (partage de memoire, appel d outils, communication directe). Risque : 7 risques systemiques (propagation de biais, fuite de donnees, collusion). Contrat de defense en 5 parties : Cible du chemin, Observation, Intervention, Frontiere de confiance, Recuperation. Lacunes : Fermeture des chemins d attaque inter-agents et recuperation post-incident restent des defis ouverts. AutoGen : garde-fous locaux. LangGraph : sandboxing d outils, mais pas de traceage inter-agents. CrewAI : roles preconfigures, mais pas de detection de collusion. Opportunite : Auditeur de chemins d attaque pour MAS, traceage bout-en-bout des interactions inter-agents. Benchmark de securite contrefactuelle.',
    'qmix': 'Optimisation de topologie multi-agents par QMIX : Vers une orchestration adaptive. Agent Q-Mix 2604.00344 reformule la selection de topologie multi-agents comme un probleme d apprentissage par renforcement multi-agents cooperatif (MARL) avec factorisation QMIX. Amelioration de +1,6 point de precision sur Humanity s Last Exam (20,8% vs 19,2%). Architecture : Encodeur GNN sensible a la topologie, Memoire GRU, Tetes Q par agent. Paradigme CTDE : entrainement centralise, execution decentralisee. Actions de communication : chaque agent choisit une action, induisant conjointement un graphe de communication par tour. Fonction de recompense : equilibre precision de la tache vs cout en tokens. Comparaison : AutoGen statique, LangGraph statique, CrewAI statique, Agent Q-Mix apprise et optimisee. Robustesse aux defaillances d agents. Generalisation : performance sur des taches inedites ? Cout d entrainement : MARL + QMIX necessite des ressources computationnelles importantes. Interpretabilite : les topologies apprises sont-elles explicables ? Securite : comment garantir que les topologies apprises ne creent pas de chemins d attaque (cf 2609.00595). Opportunite : Optimiseur de topologie multi-agents integre a Accretio.',
    'aos': 'Agent Operating Systems (AOS) : Integrer les plans de controle agentiques dans les OS traditionnels. Les agents, entites durables et probabilistes, depassent les hypotheses des OS traditionnels (processus deterministes, flux de controle explicite). Decomposition en 5 modules : Ordonnanceurs, Contexte/Memoire, Registres d outils, Politiques de confiance, Observabilite/Audit. Criteres d evaluation : Auditabilite, Determinisme, Comprehensibilite. Limites : article theorique, pas de benchmarks chiffres, pas d implementation open-source. Protocoles MCP et A2A : MCP developpe par Anthropic pour acces standardise aux outils, A2A developpe par Google pour coordination entre agents. Lacune : pas d interoperabilite native entre MCP et A2A. Gouvernance et observabilite : application des politiques, gestion d etat, tracabilite, transparence. Opportunite : couche d abstraction systemique manquant aux frameworks existants (LangGraph, AutoGen). Plan de controle unifie pour les SMA, combinant modules AOS, protocoles MCP et A2A, mecanismes de gouvernance.',
    'orchestration-1': 'Orchestration des systemes multi-agents : Vers un plan de controle auditable et interoperable. Les systemes multi-agents emergent comme l architecture dominante pour l automatisation de l IA en entreprise en 2026. Taux d echec en production entre 41% et 86,7%, dont 79% proviennent de problemes de specification et de coordination. Couche d orchestration unifiee integrant MCP et A2A avec gouvernance auditable et detection des conflits semantiques. Divergence d intention semantique : agents cooperants developpent des interpretations incoherentes d objectifs partages. Semantic Consensus Framework (SCF) : middleware sensible aux processus, six modules. Couche de contexte de processus, graphe d intention semantique, moteur de detection de conflits, protocole de resolution de consensus. AutoGen, LangGraph, CrewAI : cadres d orchestration open-source. PwC Agent OS, Accenture Trusted Agent Huddle : exemples industriels.',
    'orchestration-2': 'Orchestration des systemes multi-agents : Vers un plan de controle auditable et interoperable. Agent Operating Systems 2606.01508 introduit les AOS comme nouvelle abstraction systeme pour l ere des LLM. Agent Q-Mix 2604.00344 reformule la selection de topologie comme probleme d apprentissage par renforcement multi-agents cooperatif. Factorisation QMIX : chaque agent choisit des actions de communication qui induisent conjointement un graphe de communication par tour. Architecture : Encodeur GNN, Memoire GRU, Tetes Q par agent, paradigme CTDE. Resultats : +1.6% de precision sur Humanity s Last Exam (Gemini-3.1-Flash-Lite) vs LangGraph et Microsoft Agent Framework. Protocoles MCP et A2A : pas d interoperabilite native entre MCP et A2A. Gouvernance et observabilite : tracabilite, transparence, application des politiques, gestion d etat. Lacunes : auditabilite, cout en tokens, conflits semantiques. Plan de controle unifie combinant modules AOS, protocoles MCP et A2A, mecanismes de gouvernance.',
}

def main():
    print('=' * 70)
    print('DETECTION DE REDITES SEMANTIQUES')
    print(f'Modele : all-MiniLM-L6-v2 | Seuil cosinus : {SEUIL}')
    print('=' * 70)
    labels = list(articles.keys())
    textes = list(articles.values())
    print(f'\nNombre de documents : {len(textes)}')
    for i, label in enumerate(labels):
        print(f'  [{i}] {label} ({len(textes[i])} chars)')
    print('\nChargement du modele all-MiniLM-L6-v2...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Encodage des documents...')
    embeddings = model.encode(textes, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    print('\n' + '=' * 70)
    print('SCORES DE SIMILARITE MAX ENTRE PAIRES')
    print('=' * 70)
    paires = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = sim_matrix[i][j]
            paires.append((labels[i], labels[j], score))
    paires.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{'Doc A':<20} {'Doc B':<20} {'Score':<10} {'Seuil 0.85?'}")
    print('-' * 65)
    for a, b, score in paires:
        flag = 'OUI' if score >= SEUIL else 'non'
        print(f'{a:<20} {b:<20} {score:<10.4f} {flag}')
    print('\n' + '=' * 70)
    print('CLUSTERING DBSCAN (distance = 1 - cosinus)')
    print('=' * 70)
    eps = 1 - SEUIL
    print(f'eps = {eps}')
    clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
    distances = 1 - sim_matrix
    np.fill_diagonal(distances, 0)
    labels_cluster = clustering.fit_predict(distances)
    print(f'\nClusters trouves : {set(labels_cluster)}')
    print(f"{'Document':<20} {'Cluster'}")
    print('-' * 35)
    for i, label in enumerate(labels):
        cl = labels_cluster[i]
        cl_str = f'Cluster {cl}' if cl >= 0 else 'Bruit (isole)'
        print(f'{label:<20} {cl_str}')
    print('\n' + '=' * 70)
    print('ANALYSE DES PAIRES AU-DESSUS DU SEUIL 0.85')
    print('=' * 70)
    found = False
    for a, b, score in paires:
        if score >= SEUIL:
            print(f'  PAIRE {a} <-> {b} (score={score:.4f}) : AU-DESSUS DU SEUIL')
            found = True
    if not found:
        print('  Aucune paire au-dessus du seuil 0.85.')
    print('\n' + '=' * 70)
    print('CALIBRATION : NOMBRE DE PAIRES A DIFFERENTS SEUILS')
    print('=' * 70)
    for seuil in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        count = sum(1 for _, _, s in paires if s >= seuil)
        print(f'  Seuil {seuil:.2f} : {count} paire(s)')
    print('\n' + '=' * 70)
    print('FIN DE L ANALYSE')
    print('=' * 70)

if __name__ == '__main__':
    main()
