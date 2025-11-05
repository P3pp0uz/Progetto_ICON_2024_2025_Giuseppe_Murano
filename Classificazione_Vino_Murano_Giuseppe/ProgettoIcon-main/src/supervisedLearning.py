'''
Supervised Learning con regolarizzazione e RandomizedSearchCV
per ottimizzare Random Forest. Include KNN, Decision Tree, Gradient Boosting, SVM.
'''

import numpy as np

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("[OK] Intel Extension for Scikit-Learn attivata\n")
except ImportError:
    print("[WARNING] Intel Extension non trovata\n")

from sklearn.model_selection import (
    cross_validate, StratifiedKFold, train_test_split, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import randint
from unified_visualization import UnifiedVisualizer

visualizer = UnifiedVisualizer()


def optimize_random_forest(X_train, y_train):
    """
    Ottimizza Random Forest tramite RandomizedSearchCV su 50iterazioni e 3-fold CV.
    Restituisce il modello ottimizzato e i risultati dell'ottimizzazione.
    """

    print("\n━━━ OTTIMIZZAZIONE RANDOM FOREST ━━━")
    print("Metodo: RandomizedSearchCV (50 iterazioni, 3-fold CV)")

    # Spazio parametri da esplorare
    param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [6, 8, 10, 12],
        'min_samples_split': randint(10, 25),
        'min_samples_leaf': randint(5, 15),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy']
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=50,  # 50 combinazioni random
        cv=3,  # 3-fold per velocità
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Training Random Forest con RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    print(f"\n✓ Ottimizzazione completata!")
    print(f"Best CV F1-Score: {random_search.best_score_:.4f}")
    print(f"Best hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    results = {
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }

    return random_search.best_estimator_, results


def supervised_learning(X, y):
    """
    Pipeline apprendimento supervisionato con:
    - Regolarizzazione per ridurre overfitting
    - Random Forest ottimizzato con RandomizedSearchCV
    - Parametri ottimizzati per compromesso bias-variance
    """

    print("\n" + "=" * 80)
    print("SUPERVISED LEARNING")
    print("Tecniche applicate:")
    print(" - Regolarizzazione anti-overfitting")
    print(" - RandomizedSearchCV per RF optimization")
    print("=" * 80)

    # Split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nDataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Distribuzione train: {np.bincount(y_train)}")
    print(f"Distribuzione test: {np.bincount(y_test)}")

    X_train_balanced, y_train_balanced = X_train, y_train
    print("\n━━━ DATI SENZA OVERSAMPLING ━━━")
    print(f"Train (originale): {X_train_balanced.shape[0]} campioni")
    print(f"Distribuzione: {np.bincount(y_train_balanced)}")

    # Preprocessing
    print("\n━━━ PREPROCESSING ━━━")
    k_features = min(19, X.shape[1])
    print(f"SelectKBest: top {k_features} feature")
    selector = SelectKBest(f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
    X_test_selected = selector.transform(X_test)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # OTTIMIZZA RANDOM FOREST CON RANDOMIZEDSEARCHCV
    rf_optimized, rf_results = optimize_random_forest(X_train_scaled, y_train_balanced)

    # Modelli REGOLARIZZATI per ridurre overfitting
    print("\n━━━ MODELLI REGOLARIZZATI ━━━")

    models = {
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            metric='euclidean'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=7,
            max_features='sqrt',
            random_state=42
        ),
        'Random Forest (Optimized)': rf_optimized,  # MODELLO OTTIMIZZATO
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ),
        'SVM': SVC(
            C=5,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Training e valutazione
    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"ALGORITMO: {name}")
        print(f"{'=' * 80}")

        # Skip fit per RF (già fatto in optimize_random_forest)
        if name != 'Random Forest (Optimized)':
            model.fit(X_train_scaled, y_train_balanced)

        # Predizioni
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metriche
        train_acc = accuracy_score(y_train_balanced, y_pred_train)
        train_f1 = f1_score(y_train_balanced, y_pred_train, average='weighted')

        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_mcc = matthews_corrcoef(y_test, y_pred_test)

        # Cross-validation (solo per modelli non RF ottimizzato)
        if name == 'Random Forest (Optimized)':
            cv_f1_mean = rf_results['best_cv_score']
            cv_f1_std = 0.0  # Non disponibile da RandomizedSearchCV
        else:
            cv_scores = cross_validate(
                model, X_train_scaled, y_train_balanced, cv=cv,
                scoring=['f1_weighted', 'accuracy'],
                return_train_score=False
            )
            cv_f1_mean = cv_scores['test_f1_weighted'].mean()
            cv_f1_std = cv_scores['test_f1_weighted'].std()

        # Stampa
        print(f"Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        print(f"Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f}")
        print(f"Test MCC: {test_mcc:.4f}")
        print(f"CV F1 (5-fold): {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print(f"Overfitting gap: {train_f1 - test_f1:.4f}")

        # Plot (riaddestra su train originale per coerenza)
        output_file = f"visualization/{name.lower().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}_eval.png"
        model_for_plot = type(model)(**model.get_params()) if name != 'Random Forest (Optimized)' else model

        # Usa subset di train originale per plot
        X_train_original = X_train_scaled[:len(y_train)]

        if name != 'Random Forest (Optimized)':
            model_for_plot.fit(X_train_original, y_train)

        metrics_dict = visualizer.plot_model_evaluation(
            model=model_for_plot,
            X_train=X_train_original,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            model_name=f"{name}",
            output_path=output_file
        )

        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'test_mcc': test_mcc,
            'cv_f1_mean': cv_f1_mean,
            'cv_f1_std': cv_f1_std,
            'overfitting_gap': train_f1 - test_f1,
            'plot_metrics': metrics_dict
        }

        # Salva best_params per RF
        if name == 'Random Forest (Optimized)':
            results[name]['best_params'] = rf_results['best_params']

    # Confronto finale
    print(f"\n{'=' * 80}")
    print("CONFRONTO FINALE - MODELLI MIGLIORATI")
    print(f"{'=' * 80}")
    print(f"{'Algoritmo':<30} {'Test F1':<10} {'Test Acc':<10} {'Gap':<8} {'MCC':<8}")
    print("-" * 70)

    for name, res in results.items():
        print(f"{name:<30} {res['test_f1']:<10.4f} {res['test_accuracy']:<10.4f} "
              f"{res['overfitting_gap']:<8.4f} {res['test_mcc']:<8.4f}")

    best = max(results, key=lambda x: results[x]['test_f1'])
    print(f"\nMIGLIOR MODELLO: {best} (F1: {results[best]['test_f1']:.4f})")

    return results
