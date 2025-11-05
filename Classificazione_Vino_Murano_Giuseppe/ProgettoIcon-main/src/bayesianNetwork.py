"""
Moduli per la classificazione con modelli probabilistici:
- Naive Bayes (Gaussian e Multinomial)
- Bayesian Network con struttura DAG predefinita
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination


# Funzione di utilità per fit con BDeu prior
def fit_bn_with_bdeu(model, data_discretized):
    model.fit(data_discretized, estimator=BayesianEstimator,
              prior_type='BDeu', equivalent_sample_size=1)
    return model


# Classificatore Naive Bayes
def naive_bayes_classifier(X, y):
    print("\n" + "=" * 80)
    print("MODELLI PROBABILISTICI - Naive Bayes")
    print("=" * 80)
    print(f"\nDataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    print("Assunzioni: Indipendenza condizionale feature | target")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} campioni")
    print(f"Test set: {X_test.shape[0]} campioni")

    # Gaussian Naive Bayes
    print("\n--- Gaussian Naive Bayes ---")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)

    acc_gnb = accuracy_score(y_test, y_pred_gnb)
    f1_gnb = f1_score(y_test, y_pred_gnb, average='weighted')

    print(f"Accuracy: {acc_gnb:.4f}")
    print(f"F1-score (weighted): {f1_gnb:.4f}")

    # Multinomial Naive Bayes
    print("\n--- Multinomial Naive Bayes ---")
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train_disc, y_train)
    y_pred_mnb = mnb.predict(X_test_disc)

    acc_mnb = accuracy_score(y_test, y_pred_mnb)
    f1_mnb = f1_score(y_test, y_pred_mnb, average='weighted')

    print(f"Accuracy: {acc_mnb:.4f}")
    print(f"F1-score (weighted): {f1_mnb:.4f}")

    # Scegli il migliore
    if f1_gnb >= f1_mnb:
        best_model = gnb
        best_type = "Gaussian"
        best_f1 = f1_gnb
        best_acc = acc_gnb
    else:
        best_model = mnb
        best_type = "Multinomial"
        best_f1 = f1_mnb
        best_acc = acc_mnb

    print(f"\n[OK] Miglior modello: {best_type} NB (F1={best_f1:.4f})")

    return {
        'model': best_model,
        'type': best_type,
        'accuracy': best_acc,
        'f1_score': best_f1,
        'y_test': y_test,
        'y_pred': y_pred_gnb if best_type == "Gaussian" else y_pred_mnb
    }


# Classificatore Bayesian Network con struttura DAG
def bayesian_network_classifier(X, y, feature_names=None):
    print("\n" + "=" * 80)
    print("BAYESIAN NETWORK - Struttura DAG e Inferenza")
    print("=" * 80)

    # Discretizzazione
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    X_disc = discretizer.fit_transform(X)

    # Nomi feature
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X.shape[1])]

    df = pd.DataFrame(X_disc, columns=feature_names)
    df['target'] = y.astype(int)
    for col in df.columns:
        df[col] = df[col].astype(int)

    print(f"\nDataset discretizzato: {df.shape[0]} campioni, {df.shape[1] - 1} feature")

    # Split
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['target'])

    # Usa nomi di dominio se disponibili, altrimenti fallback
    edges = []
    if all(feat in df.columns for feat in ['alcohol', 'volatile acidity', 'sulphates']):
        edges = [
            ('alcohol', 'target'),
            ('volatile acidity', 'target'),
            ('sulphates', 'target'),
        ]
    else:
        # fallback ai primi/ultimi indici
        first = feature_names[0]
        second = feature_names[1] if len(feature_names) > 1 else feature_names[0]
        last = feature_names[-1]
        edges = [(first, 'target'), (second, 'target'), (last, 'target')]

    print(f"\nStruttura DAG definita: {len(edges)} archi")
    for edge in edges:
        print(f"  {edge[0]} → {edge[1]}")

    # Crea BN
    model = DiscreteBayesianNetwork(edges)

    # Stima parametri con MLE
    print("\nStima parametri (Maximum Likelihood)...")
    model.fit(train_df, estimator=MaximumLikelihoodEstimator)

    print(f"CPD stimate: {len(model.get_cpds())}")

    # Inferenza su test set
    print("\nInferenza con Variable Elimination...")
    inference = VariableElimination(model)

    y_test = test_df['target'].values
    y_pred = []

    # Predizione per ogni campione di test
    sample_size = min(100, len(test_df))
    test_sample = test_df.head(sample_size)

    for idx, row in test_sample.iterrows():
        evidence = {feat: int(row[feat]) for feat in feature_names if feat in model.nodes()}

        # Query: P(target | evidence)
        result = inference.query(variables=['target'], evidence=evidence)
        predicted_class = result.values.argmax()
        y_pred.append(predicted_class)

    y_test_sample = test_sample['target'].values

    # Metriche
    acc = accuracy_score(y_test_sample, y_pred)
    f1 = f1_score(y_test_sample, y_pred, average='weighted')

    print(f"\nRisultati su {sample_size} campioni test:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-score: {f1:.4f}")

    return {
        'model': model,
        'inference_engine': inference,
        'accuracy': acc,
        'f1_score': f1,
        'n_test_samples': sample_size
    }


# Pipeline completa modelli probabilistici
def probabilistic_models_pipeline(X, y):
    print("\n" + "=" * 80)
    print("PIPELINE MODELLI PROBABILISTICI")
    print("=" * 80)

    # 1. Naive Bayes
    nb_results = naive_bayes_classifier(X, y)

    results = {
        'naive_bayes': nb_results
    }

    # 2. Bayesian Network
    bn_results = bayesian_network_classifier(X, y)
    if bn_results:
        results['bayesian_network'] = bn_results

        print("\n" + "=" * 80)
        print("CONFRONTO MODELLI PROBABILISTICI")
        print("=" * 80)
        print(f"Naive Bayes:       F1={nb_results['f1_score']:.4f}")
        print(f"Bayesian Network:  F1={bn_results['f1_score']:.4f}")

    return results
