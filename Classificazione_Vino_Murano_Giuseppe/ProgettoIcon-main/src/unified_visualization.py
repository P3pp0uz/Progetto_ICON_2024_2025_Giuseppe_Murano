'''
Unified Visualizer for Supervised Learning and Knowledge Base Results
Crea visualizzazioni comparative per modelli supervisionati e risultati della Knowledge Base.
Utilizza matplotlib e seaborn per generare grafici informativi e salvali in PNG.
'''

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve


class UnifiedVisualizer:

    def __init__(self, dpi=300, style='seaborn-v0_8-darkgrid'):
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('seaborn-v0_8')
        os.makedirs('visualization', exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # PARSERS - Lettura file txt
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def parse_supervised_results(txt_file='supervised_summary.txt'):
        """
        Parser robusto di un riepilogo supervisionato in formato testuale.
        Atteso (per riga modello): "<Nome Modello> <F1> <Accuracy> <Gap> <MCC>"
        Ritorna un dict con liste parallele e calcola best_f1.
        """
        path = Path(txt_file)
        if not path.exists():
            print(f"[WARN] File non trovato: {txt_file}")
            return None

        algorithms, f1_scores, accuracies, gaps, mcc_scores = [], [], [], [], []

        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # euristica: riconosci alcune etichette comuni
                if any(tag in line for tag in ['KNN', 'Decision', 'Random', 'Gradient', 'SVM', 'Neural', 'MLP']):
                    parts = line.split()
                    # prova ad interpretare gli ultimi 4 token come numeri
                    if len(parts) >= 5:
                        try:
                            algo_name = ' '.join(parts[:-4]).strip()
                            f1v = float(parts[-4])
                            acc = float(parts[-3])
                            gap = float(parts[-2])
                            mcc = float(parts[-1])
                            algorithms.append(algo_name)
                            f1_scores.append(f1v)
                            accuracies.append(acc)
                            gaps.append(gap)
                            mcc_scores.append(mcc)
                        except (ValueError, IndexError):
                            # linea non nel formato atteso, ignora
                            continue

        if not algorithms:
            print(f"[WARN] Nessun modello parsato da {txt_file}")
            return None

        data = {
            'algorithms': algorithms,
            'f1_scores': f1_scores,
            'accuracies': accuracies,
            'gaps': gaps,
            'mcc_scores': mcc_scores
        }
        # valore utile per confronto ML vs KB
        try:
            data['best_f1'] = max(f1_scores) if f1_scores else None
        except Exception:
            data['best_f1'] = None
        return data

    @staticmethod
    def parse_kb_results(txt_file='kb_results.txt'):
        """
        Parser robusto per risultati della Knowledge Base in formato:
        Accuracy: <val>, Precision: <val>, Recall: <val>, F1-Score: <val>, Coverage: <val>
        Ritorna un dict con le chiavi normalizzate e default a 0.0 quando mancano.
        """
        path = Path(txt_file)
        if not path.exists():
            print(f"[WARN] File non trovato: {txt_file}")
            return None

        metrics = {}
        with path.open('r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or ':' not in line or '=' in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                try:
                    value = float(value.strip())
                    metrics[key] = value
                except ValueError:
                    continue

        return {
            'accuracy': metrics.get('Accuracy', 0.0),
            'precision': metrics.get('Precision', 0.0),
            'recall': metrics.get('Recall', 0.0),
            'f1_score': metrics.get('F1-Score', 0.0),
            'coverage': metrics.get('Coverage', 0.0),
        }

    # ════════════════════════════════════════════════════════════════════════
    # MODEL EVALUATION PLOT - Per singoli modelli
    # ════════════════════════════════════════════════════════════════════════

    def plot_model_evaluation(
            self,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            model_name='Model',
            output_path=None
    ):
        """
        Crea plot 2x2 per un singolo modello:
        - Confusion Matrix
        - ROC Curve (se binario e proba disponibile)
        - Learning Curve (cv=5, f1_weighted)
        - Riquadro con metriche di test
        Ritorna un dict con le metriche principali.
        """
        # Predizioni
        y_pred = model.predict(X_test)

        # Probabilità/score per ROC
        y_score = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                y_score = proba[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)

        # Metriche base
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC AUC (solo binario con score valido)
        test_auc = None
        if y_score is not None and len(np.unique(y_test)) == 2:
            try:
                test_auc = roc_auc_score(y_test, y_score)
            except Exception:
                test_auc = None

        # Figura 2x2
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1) Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        classes = sorted(np.unique(y_test))
        if len(classes) == 2:
            ticklabels = ['Class 0', 'Class 1']
        else:
            ticklabels = [f'Class {c}' for c in classes]

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=ticklabels, yticklabels=ticklabels,
            cbar_kws={'label': 'Count'}, ax=ax1
        )
        ax1.set_ylabel('True label', fontsize=11)
        ax1.set_xlabel('Predicted label', fontsize=11)
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

        # 2) ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        if test_auc is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate', fontsize=11)
            ax2.set_ylabel('True Positive Rate', fontsize=11)
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'ROC non disponibile\n(multi-classe o no proba)',
                     ha='center', va='center', fontsize=12)
        ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')

        # 3) Learning Curve
        ax3 = fig.add_subplot(gs[1, 0])
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=5, scoring='f1_weighted',
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)
        ax3.plot(train_sizes, tr_mean, 'o-', color='r', label='Training score', linewidth=2)
        ax3.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2, color='r')
        ax3.plot(train_sizes, va_mean, 'o-', color='g', label='Cross-validation score', linewidth=2)
        ax3.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.2, color='g')
        ax3.set_xlabel('Training examples', fontsize=11)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('Learning Curve', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)
        ax3.set_ylim([0.4, 1.0])

        # 4) Riquadro metriche
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        lines = [
            f"Model: {model_name}",
            f"Test Accuracy: {test_accuracy:.4f}",
            f"Test Precision: {test_precision:.4f}",
            f"Test Recall: {test_recall:.4f}",
            f"Test F1-score: {test_f1:.4f}",
            f"ROC AUC: {test_auc:.4f}" if test_auc is not None else "ROC AUC: N/A",
        ]
        ax4.text(
            0.5, 0.5, "\n".join(lines),
            ha='center', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace'
        )

        fig.suptitle(f'Model Evaluation - {model_name}', fontsize=15, fontweight='bold', y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[SAVE] ✓ {output_path}")
            plt.close()
        else:
            plt.show()

        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'roc_auc': test_auc
        }

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY VISUALIZATIONS
    # ════════════════════════════════════════════════════════════════════════

    def visualize_supervised_learning(self, data):
        """
        Crea PNG con 4 grafici comparativi (F1, Accuracy, Gap, MCC) per i modelli supervisionati.
        'data' è un dict con chiavi: algorithms, f1_scores, accuracies, gaps, mcc_scores.
        """
        if data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('SUPERVISED LEARNING - VALUTAZIONE MODELLI', fontsize=16, fontweight='bold', y=0.995)

        colors = plt.cm.Set3(np.linspace(0, 1, len(data['algorithms'])))

        # F1-Score
        ax = axes[0, 0]
        bars = ax.bar(data['algorithms'], data['f1_scores'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('F1-Score', fontweight='bold')
        ax.set_title('F1-Score Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=max(data['f1_scores']), color='red', linestyle='--', alpha=0.5, label='Best')
        for bar, score in zip(bars, data['f1_scores']):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Accuracy
        ax = axes[0, 1]
        bars = ax.bar(data['algorithms'], data['accuracies'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=max(data['accuracies']), color='red', linestyle='--', alpha=0.5, label='Best')
        for bar, acc in zip(bars, data['accuracies']):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{acc:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Overfitting Gap
        ax = axes[1, 0]
        bars = ax.bar(data['algorithms'], data['gaps'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Overfitting Gap', fontweight='bold')
        ax.set_title('Overfitting Gap Analysis', fontweight='bold')
        for bar, gap in zip(bars, data['gaps']):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{gap:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        ax.axhline(y=min(data['gaps']), color='green', linestyle='--', alpha=0.5, label='Better')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # MCC
        ax = axes[1, 1]
        bars = ax.bar(data['algorithms'], data['mcc_scores'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Matthews Correlation Coefficient', fontweight='bold')
        ax.set_title('MCC Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, mcc in zip(bars, data['mcc_scores']):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{mcc:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        ax.axhline(y=max(data['mcc_scores']), color='red', linestyle='--', alpha=0.5, label='Best')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        out = 'visualization/supervised_learning_visualization.png'
        plt.savefig(out, dpi=self.dpi, bbox_inches='tight')
        print(f"[SAVE] ✓ {out}")
        plt.close()

    def visualize_kb_results(self, data):
        """
        Crea PNG per Knowledge Base e (se disponibile) un confronto F1 con il miglior modello ML.
        'data' è un dict con chiavi: accuracy, precision, recall, f1_score, coverage
        e opzionalmente ml_f1.
        """
        if data is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('KNOWLEDGE BASE - RISULTATI VALUTAZIONE', fontsize=16, fontweight='bold')

        # KB Metrics
        ax = axes[0]
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            float(data.get('accuracy', 0.0)),
            float(data.get('precision', 0.0)),
            float(data.get('recall', 0.0)),
            float(data.get('f1_score', 0.0)),
        ]
        x = np.arange(len(categories))
        bars = ax.bar(x, values, color='steelblue', edgecolor='black', linewidth=2, alpha=0.7)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Knowledge Base Metrics', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Buono (0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Accettabile (0.6)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{val:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        # KB vs ML (robusto a chiavi mancanti)
        ax = axes[1]
        kb_f1 = data.get('f1_score', data.get('kb_f1'))
        ml_f1 = data.get('ml_f1')  # verrà fuso in generate_all_visualizations se disponibile
        labels, f1_vals = [], []
        if ml_f1 is not None:
            labels.append('ML Best Model')
            f1_vals.append(float(ml_f1))
        if kb_f1 is not None:
            labels.append('Knowledge Base')
            f1_vals.append(float(kb_f1))

        if f1_vals:
            colors_comp = ['#FF9999', '#66B2FF'][:len(f1_vals)]
            bars = ax.bar(labels, f1_vals, color=colors_comp, edgecolor='black', linewidth=2, alpha=0.7)
            ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
            ax.set_title('ML vs Knowledge Base', fontweight='bold', fontsize=12)
            ax.set_ylim([0, 1])
            ax.axhline(y=max(f1_vals), color='green', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{val:.4f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
            if ml_f1 is not None and kb_f1 is not None:
                delta = float(ml_f1) - float(kb_f1)
                ax.text(0.5, 0.95, f'ΔF1: {delta:+.4f}', transform=ax.transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=11, fontweight='bold')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'F1 non disponibile\n(per ML e/o KB)',
                    ha='center', va='center', fontsize=12)

        plt.tight_layout()
        out = 'visualization/kb_results_visualization.png'
        plt.savefig(out, dpi=self.dpi, bbox_inches='tight')
        print(f"[SAVE] ✓ {out}")
        plt.close()

    def plot_probabilistic_results(self, prob_results, output_path=None):
        """
        Visualizza i risultati di Naive Bayes e Bayesian Network su un PNG.

        Args:
            prob_results: dict con chiavi 'naive_bayes' e 'bayesian_network'
            output_path: Path dove salvare il PNG
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ========== NAIVE BAYES ==========
        nb_res = prob_results.get('naive_bayes', {})
        nb_metrics = {
            'Accuracy': nb_res.get('accuracy', 0),
            'F1-Score': nb_res.get('f1_score', 0),
        }

        colors_nb = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars_nb = ax1.bar(nb_metrics.keys(), nb_metrics.values(), color=colors_nb, alpha=0.8, edgecolor='black')
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f"Naive Bayes ({nb_res.get('type', 'Unknown')})",
                      fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Aggiungi valori sulle barre
        for bar in bars_nb:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # ========== BAYESIAN NETWORK ==========
        bn_res = prob_results.get('bayesian_network', {})
        bn_metrics = {
            'Accuracy': bn_res.get('accuracy', 0),
            'F1-Score': bn_res.get('f1_score', 0),
        }

        colors_bn = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars_bn = ax2.bar(bn_metrics.keys(), bn_metrics.values(), color=colors_bn, alpha=0.8, edgecolor='black')
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title("Bayesian Network (BDeu)",
                      fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Aggiungi valori sulle barre
        for bar in bars_bn:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Titolo principale
        fig.suptitle('Modelli Probabilistici - Risultati Comparativi',
                     fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[SAVE] ✓ {output_path}")

        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # WRAPPER GENERALE
    # ════════════════════════════════════════════════════════════════════════

    def generate_all_visualizations(
            self,
            supervised_txt='supervised_summary.txt',
            kb_txt='kb_results.txt'
    ):
        """
        Parserizza i file di riepilogo e genera le figure.
        Se trova il best F1 dei modelli supervisionati, lo fonde con i dati KB per il confronto.
        """
        sup = self.parse_supervised_results(supervised_txt)
        if sup is not None:
            self.visualize_supervised_learning(sup)

        kb = self.parse_kb_results(kb_txt)
        if kb is not None:
            # prova a derivare ml_f1 dal riepilogo supervisionato
            ml_candidate = None
            if isinstance(sup, dict):
                ml_candidate = (
                        sup.get('best_f1')
                        or sup.get('f1_weighted')
                        or sup.get('f1_score')
                        or sup.get('F1-Score')
                )
            if ml_candidate is not None:
                kb['ml_f1'] = ml_candidate
            self.visualize_kb_results(kb)


if __name__ == "__main__":
    viz = UnifiedVisualizer()
    viz.generate_all_visualizations()
