"""
Reti Neurali Migliorate - MLP
Basato su: Ingegneria della Conoscenza Capitolo 8 "Reti Neurali"

Implementa:
- MLP con architettura leggera (64, 32)
- Early stopping
- Regolarizzazione L2

"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.neural_network import MLPClassifier
from unified_visualization import UnifiedVisualizer


def create_mlp_improved(X, y):
    """
    MLP con architettura leggera (64, 32),
    con early stopping e regolarizzazione L2.
    """
    print("\n" + "=" * 80)
    print("RETI NEURALI - MLP ")
    print("=" * 80)
    print(f"\nDataset: {X.shape[0]} campioni, {X.shape[1]} feature")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.03,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.20,
        n_iter_no_change=15,
        tol=1e-4,
        verbose=False
    )

    # Training
    print("\n━━━ TRAINING ━━━")
    mlp.fit(X_train_scaled, y_train)
    print(f"Convergenza: {mlp.n_iter_} iterazioni")
    print(f"Loss finale: {mlp.loss_:.4f}")

    # Predizioni e metriche
    y_pred_train = mlp.predict(X_train_scaled)
    y_pred_test = mlp.predict(X_test_scaled)

    acc_train = accuracy_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train, average='weighted')

    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)

    print("\n━━━ RISULTATI ━━━")
    print(f"Train Accuracy: {acc_train:.4f} | Train F1: {f1_train:.4f}")
    print(f"Test  Accuracy: {acc_test:.4f} | Test  F1: {f1_test:.4f}")
    print(f"Test  Precision: {prec_test:.4f} | Test Recall: {rec_test:.4f}")
    print(f"Test  MCC: {mcc_test:.4f}")

    gap = f1_train - f1_test
    print(f"\nOverfitting gap: {gap:.4f}")

    # CV 5-fold su dati scalati completi
    print("\n━━━ CROSS-VALIDATION 5-FOLD ━━━")
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(mlp, X_scaled_full, y, cv=5, scoring='f1_weighted')
    print(f"F1-Score CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Plot (confusion + metriche) usando UnifiedVisualizer
    visualizer = UnifiedVisualizer()
    X_train_plot = scaler.transform(X_train)
    X_test_plot = X_test_scaled
    metrics = visualizer.plot_model_evaluation(
        model=mlp,
        X_train=X_train_plot,
        X_test=X_test_plot,
        y_train=y_train,
        y_test=y_test,
        model_name='MLP Improved',
        output_path='visualization/mlp_improved_evaluation.png'
    )

    return {
        'model': mlp,
        'scaler': scaler,
        'test_accuracy': acc_test,
        'test_f1': f1_test,
        'test_precision': prec_test,
        'test_recall': rec_test,
        'test_mcc': mcc_test,
        'train_accuracy': acc_train,
        'train_f1': f1_train,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'n_iterations': mlp.n_iter_,
        'final_loss': mlp.loss_,
        'overfitting_gap': gap,
        'plot_metrics': metrics
    }


def neural_networks_pipeline(X, y):
    """Pipeline completa reti neurali MIGLIORATE"""
    print("\n" + "=" * 80)
    print("PIPELINE RETI NEURALI")
    print("=" * 80)

    results = create_mlp_improved(X, y)

    print("\n" + "=" * 80)
    print("RIEPILOGO FINALE - MLP MIGLIORATO")
    print("=" * 80)
    print(f"Architettura: Input → 64 → 32 → Output")
    print(f"Regolarizzazione: L2 (alpha=0.03) + Early Stopping")
    print(f"\nPerformance:")
    print(f"  Test F1:  {results['test_f1']:.4f}")
    print(f"  Test Acc: {results['test_accuracy']:.4f}")
    print(f"  Test MCC: {results['test_mcc']:.4f}")
    print(f"\nTraining:")
    print(f"  Iterazioni: {results['n_iterations']}")
    print(f"  Loss: {results['final_loss']:.4f}")
    print(f"  Gap: {results['overfitting_gap']:.4f}")
    print("=" * 80)

    return results
