"""
Knowledge Base per inferenza qualità vini
Basata su regole di dominio derivate da esperienza enologica
"""

# Definizione della Knowledge Base
class WineQualityKB:

    def __init__(self):
        """Inizializza regole di dominio basate su esperienza enologica"""
        self.rules = [
            # Regola 1: Alto contenuto alcolico favorisce qualità
            {
                'name': 'High_Alcohol_Quality',
                'conditions': [
                    ('alcohol', '>', 11.5),
                    ('volatile acidity', '<', 0.6)
                ],
                'conclusion': 'high_quality',
                'confidence': 0.85
            },

            # Regola 2: Acidità volatile alta ↔ scarsa qualità
            {
                'name': 'High_Volatile_Acidity_Poor',
                'conditions': [
                    ('volatile acidity', '>', 0.8),
                    ('alcohol', '<', 10.0)
                ],
                'conclusion': 'low_quality',
                'confidence': 0.80
            },

            # Regola 3: SO2 equilibrato favorisce qualità
            {
                'name': 'Balanced_SO2_Quality',
                'conditions': [
                    ('free sulfur dioxide', '>=', 6),
                    ('free sulfur dioxide', '<=', 60),
                    ('total sulfur dioxide', '<=', 300),
                    ('alcohol', '>', 10.0)
                ],
                'conclusion': 'high_quality',
                'confidence': 0.75
            },

            # Regola 4: Densità bassa + alcol alto → qualità
            {
                'name': 'Low_Density_High_Alcohol',
                'conditions': [
                    ('density', '<', 0.9956),
                    ('alcohol', '>', 11.0)
                ],
                'conclusion': 'high_quality',
                'confidence': 0.70
            },

            # Regola 5: Eccesso sulfiti → scarsa qualità
            {
                'name': 'Excess_Sulfites_Poor',
                'conditions': [
                    ('total sulfur dioxide', '>', 300),
                    ('alcohol', '<', 10.5)
                ],
                'conclusion': 'low_quality',
                'confidence': 0.78
            },

            # Regola 6: Citric acid favorisce qualità
            {
                'name': 'Citric_Acid_Quality',
                'conditions': [
                    ('citric acid', '>', 0.3),
                    ('pH', '<', 3.5),
                    ('alcohol', '>', 10.0)
                ],
                'conclusion': 'high_quality',
                'confidence': 0.72
            },

            # Regola 7: pH equilibrato + solidi sospesi bassi
            {
                'name': 'Balanced_pH_Clear',
                'conditions': [
                    ('pH', '>=', 3.0),
                    ('pH', '<=', 3.4),
                    ('residual sugar', '<', 5.0)
                ],
                'conclusion': 'high_quality',
                'confidence': 0.68
            },

            # Regola 8: Solidi sospesi alti → scarsa qualità
            {
                'name': 'High_Residual_Poor',
                'conditions': [
                    ('residual sugar', '>', 10.0),
                    ('alcohol', '<', 9.5),
                    ('pH', '>', 3.6)
                ],
                'conclusion': 'low_quality',
                'confidence': 0.75
            }
        ]

    # Valutazione di un singolo campione di vino contro le regole della KB
    def evaluate_wine(self, wine_features: dict):
        predictions = []
        matched_rules = []

        for rule in self.rules:
            # Verifica se tutte le condizioni sono soddisfatte
            all_conditions_met = True

            for feature, operator, threshold in rule['conditions']:
                if feature not in wine_features:
                    all_conditions_met = False
                    break

                value = wine_features[feature]

                if operator == '>' and value <= threshold:
                    all_conditions_met = False
                    break
                elif operator == '<' and value >= threshold:
                    all_conditions_met = False
                    break
                elif operator == '>=' and value < threshold:
                    all_conditions_met = False
                    break
                elif operator == '<=' and value > threshold:
                    all_conditions_met = False
                    break

            # Se tutte le condizioni sono soddisfatte, aggiungi conclusione
            if all_conditions_met:
                predictions.append((rule['conclusion'], rule['confidence']))
                matched_rules.append(rule['name'])

        return predictions, matched_rules

    # Inferenza qualità vino basata sulle regole della KB
    def infer_quality(self, wine_features: dict):
        predictions, matched_rules = self.evaluate_wine(wine_features)

        if len(predictions) == 0:
            return 'unknown', 0.0, 0

        # Aggregazione confidenze per classe
        high_quality_conf = []
        low_quality_conf = []

        for conclusion, confidence in predictions:
            if conclusion == 'high_quality':
                high_quality_conf.append(confidence)
            else:
                low_quality_conf.append(confidence)

        # Media di confidenza per classe
        avg_high = sum(high_quality_conf) / len(high_quality_conf) if high_quality_conf else 0
        avg_low = sum(low_quality_conf) / len(low_quality_conf) if low_quality_conf else 0

        # Decisione finale
        if avg_high > avg_low:
            predicted_class = 'high_quality'
            confidence = avg_high
        else:
            predicted_class = 'low_quality'
            confidence = avg_low

        return predicted_class, confidence, len(matched_rules)


# valutazione della KB su dataset
def evaluate_kb_on_dataset(kb: WineQualityKB, feature_dict_list: list, true_labels: list):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    kb_predictions = []
    kb_confidences = []
    num_rules_per_sample = []

    for features in feature_dict_list:
        predicted, confidence, num_rules = kb.infer_quality(features)
        # Converti a 0/1: high_quality → 1, low_quality → 0
        pred_binary = 1 if predicted == 'high_quality' else 0
        kb_predictions.append(pred_binary)
        kb_confidences.append(confidence)
        num_rules_per_sample.append(num_rules)

    # Metriche
    accuracy = accuracy_score(true_labels, kb_predictions)
    precision = precision_score(true_labels, kb_predictions, zero_division=0)
    recall = recall_score(true_labels, kb_predictions, zero_division=0)
    f1 = f1_score(true_labels, kb_predictions, zero_division=0)

    # Coverage: % campioni per cui almeno 1 regola è stata applicata
    coverage = sum(1 for n in num_rules_per_sample if n > 0) / len(num_rules_per_sample)

    results = {
        'kb_accuracy': accuracy,
        'kb_precision': precision,
        'kb_recall': recall,
        'kb_f1': f1,
        'kb_coverage': coverage,  # % campioni con predizione (non 'unknown')
        'avg_confidence': sum(kb_confidences) / len(kb_confidences),
        'predictions': kb_predictions,
        'confidences': kb_confidences
    }

    return results
