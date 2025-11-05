"""
PROGETTO INGEGNERIA DELLA CONOSCENZA - WINE QUALITY
Università di Bari - A.A. 2024/2025
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd

# Import moduli del progetto
from supervisedLearning import supervised_learning
from neuralNetworks import neural_networks_pipeline
from bayesianNetwork import probabilistic_models_pipeline
from knowledge import WineQualityKB, evaluate_kb_on_dataset
from sklearn.model_selection import train_test_split
from unsupervisedLearning import kmeans_clustering
from unified_visualization import UnifiedVisualizer

# Configurazione logging
warnings.filterwarnings('ignore')
logging.getLogger('sklearnex').setLevel(logging.ERROR)


# Caricamento e preprocessing dataset vino (red + white)
def load_wine_dataset(path_merged='winequality-merged.csv',
                      path_red='winequality-red.csv',
                      path_white='winequality-white.csv',
                      merge=True):
    if merge and os.path.exists(path_merged):
        print(f"[LOAD] Caricamento dataset merged: {path_merged}")
        df = pd.read_csv(path_merged, sep=';')
        return preprocess_data(df, dataset_name="Merged")

    # Carica dataset separati
    if not os.path.exists(path_red) or not os.path.exists(path_white):
        raise FileNotFoundError(
            f"Dataset non trovati! Cercare: {path_red}, {path_white}"
        )

    print(f"[LOAD] Caricamento dataset RED: {path_red}")
    red = pd.read_csv(path_red, sep=';')
    print(f"[LOAD] Caricamento dataset WHITE: {path_white}")
    white = pd.read_csv(path_white, sep=';')

    # Aggiungi colonna per tipo vino
    red['wine_type'] = 0  # Red = 0
    white['wine_type'] = 1  # White = 1

    # Concatena
    df = pd.concat([red, white], ignore_index=True)
    print(f"[MERGE] Dataset merged: {df.shape[0]} campioni totali")

    # Salva merged per future esecuzioni
    df.to_csv(path_merged, sep=';', index=False)
    print(f"[SAVE] Dataset unito salvato: {path_merged}")

    return preprocess_data(df, dataset_name="Merged (Red + White)")


# Preprocessing dataset
def preprocess_data(df, dataset_name="Dataset"):
    print(f"\n[PREPROC] Preprocessing dataset: {dataset_name}")

    # Binarizza quality: <= 5 (Bassa/0) vs >= 6 (Alta/1)
    df['quality_binary'] = (df['quality'] >= 6).astype(int)

    # Colonne da escludere
    cols_to_drop = ['quality', 'quality_binary']
    if 'type' in df.columns:
        cols_to_drop.append('type')

    # Rimuovi colonne vuote
    unnamed_cols = [col for col in df.columns if 'unnamed' in col.lower()]
    cols_to_drop.extend(unnamed_cols)

    # Feature e target
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    y = df['quality_binary'].values

    # Conversione a numerico (per gestire eventuali string)
    print(f"[PREPROC] Conversione colonne a tipo numerico...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Gestione NaN
    if X.isnull().any().any():
        nan_count = X.isnull().sum().sum()
        print(f"[WARNING] {nan_count} valori NaN rilevati")
        print(f"[PREPROC] Riempimento NaN con media della colonna...")
        X = X.fillna(X.mean())

    # Converti a numpy array
    X = X.values

    print(f"\n[PREPROC] Caricamento completato")
    print(f"  Campioni: {X.shape[0]}")
    print(f"  Feature: {X.shape[1]}")
    print(f"  Classe 0 (Bassa qualità): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  Classe 1 (Alta qualità): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")

    return X, y


def main():
    """Pipeline principale per il progetto Wine Quality
    1) Caricamento dataset
    2) Apprendimento non supervisionato: K-Means Clustering
    3) Apprendimento supervisionato: KNN, Decision Tree, Random Forest, Gradient Boosting, SVM
    4) Reti Neurali: MLP leggero con Early Stopping
    5) Modelli Probabilistici: Naive Bayes, Discrete Bayesian
    6) Knowledge Base: Regole enologiche dominio-specifiche + Forward Chaining
    7) Generazione visualizzazioni unificate in PNG
    """

    print("\n" + "=" * 100)
    print("PROGETTO INGEGNERIA DELLA CONOSCENZA - WINE QUALITY")
    print("Università di Bari - Anno Accademico 2024/2025")
    print("=" * 100)

    # ==========================
    # 1) CARICAMENTO DATASET
    # ==========================
    print("\n" + "=" * 100)
    print("1. CARICAMENTO DATASET")
    print("=" * 100)

    try:
        X, y = load_wine_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # ==========================
    # 2) APPRENDIMENTO NON SUPERVISIONATO - K-MEANS CLUSTERING
    # ==========================
    print("\n" + "=" * 100)
    print("2. APPRENDIMENTO NON SUPERVISIONATO - K-MEANS CLUSTERING")
    print("=" * 100)

    try:
        clustering_results = kmeans_clustering(X, k_min=2, k_max=11)
        k_elbow = clustering_results['k_elbow']
        print(f"\n[CLUSTER] k (Elbow/WSS): {k_elbow}")

        # Report dimensioni cluster sull'intero dataset
        unique, counts = np.unique(clustering_results['cluster_labels'], return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        print(f"Numero di cluster (k_elbow): {clustering_results['n_clusters']}")
        print(f"Dimensioni cluster (sull'intero dataset): {cluster_sizes}")

        # k ottimale (Elbow/WSS)
        k_optimal = clustering_results['n_clusters']
        print(f"\n[CLUSTER] k ottimale (Elbow/WSS): {k_optimal}")

    except Exception as e:
        print(f"[WARNING] Clustering fallito: {e}")
        clustering_results = None

    # ==========================
    # 3) APPRENDIMENTO SUPERVISIONATO
    # ==========================
    print("\n" + "=" * 100)
    print("3. APPRENDIMENTO SUPERVISIONATO")
    print("Algoritmi: KNN, Decision Tree, Random Forest, Gradient Boosting, SVM")
    print("Tecniche: Cross-Validation 5-fold + RandomizedSearchCV")
    print("=" * 100)

    try:
        supervised_results = supervised_learning(X, y)

        # Salva summary
        print("\n[SAVE] Riepilogo Supervised Learning...")
        with open('supervised_summary.txt', 'w', encoding='utf-8') as f:
            f.write("SUPERVISED LEARNING - RIEPILOGO\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"{'Algoritmo':<30} {'Test F1':<12} {'Test Acc':<12} {'Gap':<12} {'MCC':<12}\n")
            f.write("-" * 90 + "\n")

            for name, res in supervised_results.items():
                f.write(f"{name:<30} {res['test_f1']:<12.4f} {res['test_accuracy']:<12.4f} "
                        f"{res['overfitting_gap']:<12.4f} {res['test_mcc']:<12.4f}\n")

            best_model = max(supervised_results.keys(),
                             key=lambda x: supervised_results[x]['test_f1'])
            f.write(f"\nMIGLIOR MODELLO: {best_model}\n")
            f.write(f"Test F1-Score: {supervised_results[best_model]['test_f1']:.4f}\n")

        print("[OK] Salvato: supervised_summary.txt")

    except Exception as e:
        print(f"[ERROR] Supervised Learning fallito: {e}")

    # ==========================
    # 4) RETI NEURALI
    # ==========================
    print("\n" + "=" * 100)
    print("4. RETI NEURALI")
    print("Architettura MLP leggera (64-32) + Early Stopping")
    print("=" * 100)

    mlp_results = neural_networks_pipeline(X, y)
    print(f"\n[MLP] Training completato")
    print(f"  Test F1-Score: {mlp_results['test_f1']:.4f}")
    print(f"  Test Accuracy: {mlp_results['test_accuracy']:.4f}")
    print(f"  Overfitting Gap: {mlp_results['overfitting_gap']:.4f}")

    # ==========================
    # 5) MODELLI PROBABILISTICI
    # ==========================
    print("\n" + "=" * 100)
    print("5. MODELLI PROBABILISTICI")
    print("Naive Bayes + DiscreteBayesianNetwork con struttura DAG")
    print("=" * 100)

    try:
        prob_results = probabilistic_models_pipeline(X, y)
        print(f"\n[PROB] Training completato")

        # Visualizza e salva risultati su PNG
        if prob_results:
            viz = UnifiedVisualizer()
            viz.plot_probabilistic_results(prob_results,
                                           output_path='visualization/probabilistic_results.png')
            print(f"[VIZ] Risultati salvati in 'visualization/probabilistic_results.png'")

    except Exception as e:
        print(f"[WARNING] Modelli probabilistici saltati: {e}")
        prob_results = None

    # ==========================
    # 6) KNOWLEDGE BASE
    # ==========================
    print("\n" + "=" * 100)
    print("6. KNOWLEDGE BASE - DOMAIN EXPERT RULES")
    print("Forward Chaining + Regole enologiche dominio-specifiche")
    print("=" * 100)

    try:
        # Inizializza KB
        kb = WineQualityKB()
        print(f"\n[KB]Knowledge Base inizializzata")
        print(f"  Numero di regole: {len(kb.rules)}")
        print(f"  Regole caricate:")
        for rule in kb.rules:
            print(f"    • {rule['name']}: {rule['conclusion']} (confidence={rule['confidence']:.2f})")

        # Converti X_test a lista di dict con nomi feature
        # Estrai nomi feature dai dati originali
        feature_names = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]

        # Crea lista di dict per test set
        feature_dict_list = []
        for row in X_test:
            wine_dict = {name: value for name, value in zip(feature_names[:len(row)], row)}
            feature_dict_list.append(wine_dict)

        kb_results = evaluate_kb_on_dataset(kb, feature_dict_list, y_test)

        print(f"\n[KB] Valutazione completata su {len(y_test)} campioni")
        print(f"  KB Accuracy: {kb_results['kb_accuracy']:.4f}")
        print(f"  KB Precision: {kb_results['kb_precision']:.4f}")
        print(f"  KB Recall: {kb_results['kb_recall']:.4f}")
        print(f"  KB F1-Score: {kb_results['kb_f1']:.4f}")
        print(f"  KB Coverage (regole applicate): {kb_results['kb_coverage']:.1%}")
        print(f"  Confidence media: {kb_results['avg_confidence']:.4f}")

        # Confronto KB vs Best ML
        print("\n" + "-" * 100)
        print("CONFRONTO KB vs BEST ML MODEL")
        print("-" * 100)

        best_ml_name = max(supervised_results.keys(),
                           key=lambda x: supervised_results[x]['test_f1'])
        best_ml_f1 = supervised_results[best_ml_name]['test_f1']

        delta_f1 = best_ml_f1 - kb_results['kb_f1']

        print(f"\nBest ML Model: {best_ml_name}")
        print(f"  ML F1-Score: {best_ml_f1:.4f}")
        print(f"\nKnowledge Base:")
        print(f"  KB F1-Score: {kb_results['kb_f1']:.4f}")
        print(f"\nΔF1 (ML - KB): {delta_f1:+.4f}")

        if delta_f1 > 0:
            print(f"ML migliore della KB di {delta_f1:.1%}")
        elif delta_f1 < -0.05:
            print(f"KB COMPETITIVA! Migliore di ML di {abs(delta_f1):.1%}")
        else:
            print(f"KB E ML COMPARABILI (differenza < 0.5%)")

        # Salva risultati KB
        with open('kb_results.txt', 'w', encoding='utf-8') as f:
            f.write("KNOWLEDGE BASE - RISULTATI\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Accuracy:  {kb_results['kb_accuracy']:.4f}\n")
            f.write(f"Precision: {kb_results['kb_precision']:.4f}\n")
            f.write(f"Recall:    {kb_results['kb_recall']:.4f}\n")
            f.write(f"F1-Score:  {kb_results['kb_f1']:.4f}\n")
            f.write(f"Coverage:  {kb_results['kb_coverage']:.1%}\n")
            f.write(f"\nCONFRONTO KB vs {best_ml_name}\n")
            f.write(f"ML F1:     {best_ml_f1:.4f}\n")
            f.write(f"KB F1:     {kb_results['kb_f1']:.4f}\n")
            f.write(f"ΔF1:       {delta_f1:+.4f}\n")

        print(f"\n[SAVE] Salvato: kb_results.txt")

    except Exception as e:
        print(f"[ERROR] Knowledge Base fallito: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 100)
    print("7. GENERAZIONE VISUALIZZAZIONI IN PNG")
    print("=" * 100)

    try:
        viz = UnifiedVisualizer()
        viz.generate_all_visualizations(
            supervised_txt='supervised_summary.txt',
            kb_txt='kb_results.txt'
        )
        print("[OK] Visualizzazioni generate in 'visualization/'")
    except Exception as e:
        print(f"[WARN] Visualizzazioni non generate: {e}")

    print("\n Pipeline completa!")


if __name__ == "__main__":
    main()
