# PREDIZIONE DELLA QUALITÀ DEL VINO
## Sistema integrato di Machine Learning e Knowledge Representation

**Università di Bari - A.A. 2024/2025**  
**Corso: Ingegneria della Conoscenza**  
**Autore:** Giuseppe Murano  
**Matricola**: 758407 
**Email**: g.murano2@studenti.uniba.it  
**Repository:** [GitHub Link]

---

## Sommario

1. [Introduzione](#introduzione)
2. [Elenco argomenti di interesse](#elenco-argomenti-di-interesse)
3. [Requisiti funzionali](#requisiti-funzionali)
4. [Dataset](#dataset)
5. [Rappresentazione della Conoscenza](#rappresentazione-della-conoscenza)
6. [Apprendimento supervisionato](#apprendimento-supervisionato)
7. [Apprendimento non supervisionato](#apprendimento-non-supervisionato)
8. [Ragionamento probabilistico](#ragionamento-probabilistico)
9. [Reti Neurali](#reti-neurali)
10. [Visualizzazione e Reportistica](#visualizzazione-e-reportistica)
11. [Conclusioni](#conclusioni)
12. [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

## Introduzione

La qualità del vino rappresenta un fattore critico nella gestione della produzione vitivinicola e nella soddisfazione del consumatore. Una valutazione accurata delle caratteristiche fisico-chimiche del vino è essenziale per garantire standard qualitativi coerenti e ottimizzare i processi di produzione.

Il presente progetto si concentra sullo sviluppo di un **sistema avanzato e integrato di predizione della qualità del vino**, che combina molteplici approcci dell'Ingegneria della Conoscenza:

- **Knowledge Representation**: Modellazione di regole esperte enologiche
- **Machine Learning**: Algoritmi di classificazione supervisionati
- **Unsupervised Learning**: Clustering per pattern discovery
- **Probabilistic Reasoning**: Reti Bayesiane per inferenza
- **Deep Learning**: Reti Neurali artificiali

L'obiettivo principale è fornire uno strumento **affidabile, preciso e interpretabile** per la classificazione della qualità del vino in categorie binarie (bassa/alta qualità), assistendo gli esperti enologi nella valutazione e nel controllo qualità.

### Dataset 

**Fonte:** Wine Quality Dataset di Cortez et al. (2009)  
**Link:** https://archive.ics.uci.edu/ml/datasets/wine+quality

Nel presente caso di studio è stata scelta l’unione dei dataset "Wine Quality Red" e "Wine Quality White", due insiemi di dati ampiamente utilizzati nel campo dell’analisi sensoriale e della valutazione della qualità dei vini. Questi dataset, una volta fusi, permettono di studiare e confrontare le caratteristiche chimiche e fisiche di entrambe le tipologie di vino, fornendo una panoramica completa e integrata sulle variabili che influenzano la qualità finale.​

Attributi del dataset unificato
Tipo di vino: rosso o bianco

Acidità fissa: concentrazione di acidi stabili nel vino

Acidità volatile: concentrazione di acidi facilmente volatili

Acido citrico: componente che contribuisce alla freschezza

Zuccheri residui: quantità di zuccheri rimasti dopo la fermentazione

Cloruri: concentrazione di sali

Anidride solforosa libera e totale: usata per la conservazione e l’inibizione di microrganismi

Densità: indice della concentrazione dei soluti

pH: acidità complessiva

Solfati: impatto sulla conservazione, possono influenzare il gusto

Alcol: gradazione alcolica finale

Qualità: punteggio sensoriale attribuito da esperti

Questi attributi forniti per ciascun campione permettono di analizzare la risposta del vino a differenti condizioni fisico-chimiche e di costruire modelli predittivi sulla qualità, riflettendo accuratamente le sfide della valutazione enologica. Grazie alla varietà e ricchezza dei dati uniti, il dataset consente analisi approfondite sui fattori che caratterizzano, differenziano e determinano la qualità dei vini rossi e bianchi.



---

## Elenco argomenti di interesse

### 2.1 Rappresentazione della Conoscenza (Capitolo 6 di [1])

L’integrazione della conoscenza nel sistema,"knowledge.py", si fonda sull’utilizzo di una Knowledge Base esperta progettata specificamente per il dominio della qualità del vino. Questo modulo implementa una base di conoscenza strutturata in regole di dominio, basate sull’esperienza enologica (ad esempio, quantità di alcol, acidità, solfiti), con l’obiettivo di arricchire i dati grezzi di partenza con informazioni semantiche rilevanti per i successivi processi di apprendimento automatico e analisi predittiva.​

**Modello e rappresentazione della conoscenza**
La rappresentazione della conoscenza nella Knowledge Base si basa su una struttura formale di regole, ciascuna definita da condizioni sulle diverse caratteristiche chimico-fisiche dei vini (come alcol, acidità volatile, anidride solforosa, pH, zuccheri residui, ecc.). Queste regole permettono di inferire in modo trasparente e interpretabile la qualità di un vino, favorendo sia la comprensione sia l’arricchimento del dataset con attributi derivati semantici.​

**Estrattori e arricchimento semantico**
Il processo di integrazione consiste nell’applicare le regole della Knowledge Base a ciascun campione del dataset: ogni vino viene valutato tramite forward chaining sulle condizioni definite dalle regole. L’esito di ogni regola aggiorna il set informativo tramite nuove colonne che rappresentano la conclusione raggiunta e il grado di confidenza della regola stessa. In questo modo, il dataset celebra sia arricchimento (nuove feature semantiche) sia una maggiore interpretabilità, grazie alla trasparenza delle regole utilizzate.​

**Strumenti e implementazione**
Per l’implementazione si utilizza Python puro; la classe principale, "WineQualityKB", contiene tutte le regole di dominio e le logiche inferenziali per valutare ogni campione e calcolare una predizione aggregata sulla qualità. La funzione di valutazione sul dataset permette anche la misurazione di metriche come accuratezza, precisione, recall, f1-score e coverage, sottolineando così l’efficacia dell’arricchimento semantico nella modellazione.​

**Decisioni di progetto**
Le regole implementate coprono sia fattori di qualità positiva (ad esempio, alto contenuto di alcol, livelli di solfiti equilibrati, presenza di acido citrico) che negativa (acidi volatili troppo elevati, eccesso di zuccheri residui, eccessiva anidride solforosa totale). La loro applicazione serve sia a migliorare le prestazioni predittive dei modelli sia a garantire che gli indici semantici creati siano rilevanti e interpretabili. L’arricchimento si realizza anche attraverso la rimozione automatica di colonne costanti, per garantire robustezza e prevenire ridondanze.

**Motivazione scelta regole**
Regola 1: High_Alcohol_Quality
Motivazione: Un contenuto alcolico elevato (>11.5%) combinato con bassa acidità volatile (<0.6) è indicativo di un vino ben fermentato e stabile. Vini ad alta gradazione senza difetti di acidità volatile rappresentano generalmente qualità superiore, poiché la fermentazione è stata completa e controllata.​

Regola 2: High_Volatile_Acidity_Poor
Motivazione: L'acidità volatile alta (>0.8%) associata a basso contenuto alcolico (<10%) è un marker di difetti enologici, spesso causato da contaminazione batterica o ossidazione incontrollata. Un vino giovane e poco alcolico con questi problemi risulta di qualità scadente.​

Regola 3: Balanced_SO2_Quality
Motivazione: L'anidride solforosa (SO2) ha ruoli fondamentali nella conservazione del vino. Un intervallo equilibrato (6-60 mg/L di SO2 libero, ≤300 mg/L totale) combattere l'ossidazione e le infezioni microbiche senza alterare i caratteri sensoriali. Questo equilibrio, associato a buon contenuto alcolico (>10%), garantisce stabilità e qualità.​

Regola 4: Low_Density_High_Alcohol
Motivazione: Una densità bassa (<0.9956) con alto contenuto alcolico (>11%) indica che gli zuccheri fermentabili sono stati completamente convertiti in alcol. Questo rappresenta un processo fermentativo ottimale e una buona struttura del vino.​

Regola 5: Excess_Sulfites_Poor
Motivazione: Un eccesso di solfiti totali (>300 mg/L) combinato con basso contenuto alcolico (<10.5%) suggerisce un tentativo di compensare la scarsa stabilità naturale del vino con conservanti. Questi vini presentano meno potenziale di qualità e gusto meno pulito.​

Regola 6: Citric_Acid_Quality
Motivazione: L'acido citrico (>0.3%), con pH acido (<3.5) e adeguato contenuto alcolico (>10%), contribuisce a un profilo sensoriale più fresco e complesso. Questa combinazione favorisce equilibrio gustativo e longevità del vino.​

Regola 7: Balanced_pH_Clear
Motivazione: Un pH equilibrato (3.0-3.4) con basso residuo zuccherino (<5.0 g/L) crea le condizioni ideali per un vino secco e stabile. Questo intervallo di pH è ottimale per inibire le contaminazioni microbiche e mantenere la chiarezza e la freschezza.​

Regola 8: High_Residual_Poor
Motivazione: Un residuo zuccherino elevato (>10.0 g/L) associato a basso contenuto alcolico (<9.5%) e pH alto (>3.6) indica fermentazione incompleta e scarsa stabilità. Queste caratteristiche insieme segnalano un vino di qualità inferiore, con rischi di instabilità microbiologica.

**Valutazione**
La valutazione dell’integrazione semantica si basa sulle metriche di performance ottenute applicando la Knowledge Base all’intero dataset, evidenziando come l’aggiunta di feature derivate aumenti la dimensionalità informativa e la capacità discriminante del dataset stesso. L’esito positivo di questo approccio è dimostrato dall’efficacia delle regole nel classificare differenti casi di qualità del vino e dall’incremento della copertura informativa nei confronti dei dati grezzi.

---

### 2.2 Apprendimento supervisionato (Capitolo 7 di [1])

**Sommario**
La rappresentazione della conoscenza nel codice si basa su modelli di apprendimento supervisionato applicati a dati tabellari con classi sbilanciate. La conoscenza è codificata attraverso modelli classici di machine learning (Random Forest, SVM, Gradient Boosting, KNN, Decision Tree) e reti neurali Multi-Layer Perceptron (MLP) ottimizzate tramite ricerca randomizzata di iperparametri (RandomizedSearchCV per i modelli classici, tuning architetturale ed early stopping per il MLP). La base di conoscenza è costituita da dati numerici preprocessati (selezione delle feature, bilanciamento con SMOTE, normalizzazione). Viene inoltre integrata la conoscenza derivante da tecniche di regolarizzazione per ridurre l'overfitting, sfruttando metriche di valutazione standard per modelli classificatori multilabel.​

**Strumenti utilizzati**
Scikit-learn: per implementazione modelli (Random Forest, SVM, KNN, Decision Tree, Gradient Boosting, MLPClassifier), metodi di preprocessing (StandardScaler, SelectKBest), validazione incrociata, metriche di valutazione (accuracy, f1-score, precision, recall, MCC).

imblearn (SMOTE): per il bilanciamento delle classi in fase di training.

SciPy: distribuzioni di parametri per la RandomizedSearchCV.

Intel Extension for Scikit-learn (patchsklearn): per ottimizzazione delle prestazioni (se disponibile).

UnifiedVisualizer: per la visualizzazione e il confronto dei risultati dei modelli.​

**Decisioni di Progetto**

Bilanciamento del dataset tramite SMOTE con parametri di default (random_state=42, k_neighbors=5).

Selezione delle migliori feature con SelectKBest.

Standardizzazione dei dati con StandardScaler.

Ottimizzazione dei modelli classici tramite RandomizedSearchCV; architettura e parametri MLP scelti per garantire alta generalizzazione (es. hidden layers 64-32, regolarizzazione L2, early stopping, learning rate adattivo).

Metriche di scoring focalizzate sull'F1-score ponderato e MCC, per gestire lo sbilanciamento delle classi.

Analisi del gap tra train/test, stampa degli iperparametri ottimali, e confronto cross-validation su tutti i modelli.​

**Valutazione**
Metriche adottate: Accuracy, Precision, Recall, F1-score ponderato (weighted), Matthew's Correlation Coefficient (MCC).

\[
MCC = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
\]

Valutazione effettuata sia sul training bilanciato (post-SMOTE) che sul test originale.
Comparison dei modelli con output tabellari e grafici generati da UnifiedVisualizer.
Discussione dei risultati focalizzata sul trade-off bias-variance e capacità di generalizzazione (gap tra F1-score train e test, early stopping per MLP).
Salvataggio e confronto finale delle performance con identificazione del modello migliore tramite punteggio F1 sul test set.​

---

### 2.3 Apprendimento non supervisionato (Capitolo 10 di [1])

**Sommario**
Il codice implementa una pipeline di apprendimento non supervisionato per dati tabellari, basata prevalentemente su metodi di clustering. La rappresentazione della conoscenza è affidata a modelli K-Means e Gaussian Mixture Models (GMM), con l’uso della PCA per la visualizzazione e riduzione della dimensionalità. La conoscenza pregressa (background knowledge) è limitata alla struttura intrinseca dei dati; i modelli apprendono pattern dai dati stessi, senza supervisioni o label esterne. Sono implementate varie strategie per la determinazione automatica del numero di cluster, sfruttando sia criteri statistici (BIC, AIC) che euristici (elbow method).​

**Strumenti utilizzati**
Scikit-learn: implementazione K-Means, GaussianMixture, PCA, MinMaxScaler, strumenti per la manipolazione e preprocessing dei dati.

Matplotlib: per la generazione e il salvataggio dei grafici (elbow plot, PCA plot).

NumPy: per l’elaborazione numerica e gestione degli array.


**Decisioni di Progetto**
I dati sono scalati tra 0 e 1 tramite MinMaxScaler per migliorare la qualità sia del clustering che della visualizzazione PCA.

Il numero di cluster viene determinato principalmente minimizzando il BIC calcolato da GaussianMixture, con il criterio dell’elbow su WSS di K-Means utilizzato come supporto.

Range di ricerca su k personalizzabile: default 2–10.

Se vi è disaccordo tra i criteri (BIC ed elbow), viene data priorità al BIC, ma il valore suggerito dall’elbow viene comunque riportato.

KMeans viene addestrato sempre con n_init=20 e max_iter=500 per consistenza e affidabilità dei risultati.

PCA è utilizzata per proiettare i dati in due dimensioni così da facilitare l’ispezione visiva dei cluster.​

**Valutazione**
Metriche adottate:

WSS (within-cluster sum of squares) per ogni valore di k (K-Means).

BIC e AIC (solo GMM).

Grafici:

Elbow plot per determinare la curvatura (“gomito”).

PCA scatter plot con colorazione secondo i cluster previsti.

Discussione dei risultati basata sull’analisi delle curve di WSS, BIC, AIC e sulla coerenza visiva dei cluster emersi in PCA.

Il codice indica sempre k scelto da BIC, eventuale differenza rispetto all’elbow, e salva i grafici generati per la successiva valutazione e reporting.

---

### 2.4 Ragionamento probabilistico (Capitolo 9 di [1])

**Sommario**
Questo codice realizza una pipeline di classificazione probabilistica con modelli Naive Bayes e Reti Bayesiane, applicata a dati tabellari. La conoscenza è formalizzata tramite modelli probabilistici: la rappresentazione si basa sulle distribuzioni condizionate tra variabili descritte dalla DAG di una rete bayesiana (struttura a grafo aciclico diretto) o tramite l’assunzione di indipendenza condizionale nel caso Naive Bayes. La base di conoscenza è composta dai dati stessi, discretizzati e utilizzati sia per stimare i parametri delle distribuzioni che la struttura della rete. La pipeline confronta i metodi in termini di performance predittiva selezionando automaticamente il migliore.​

**Strumenti utilizzati**
scikit-learn: funzioni per train-test split, discretizzazione (KBinsDiscretizer), metriche di valutazione (accuracy, f1-score), classificatori GaussianNB e MultinomialNB.

pgmpy: costruzione, stima parametri (MaximumLikelihoodEstimator, BayesianEstimator), e inferenza (VariableElimination) per le reti bayesiane.

pandas, numpy: gestione dati.


**Decisioni di Progetto**
Discretizzazione dei dati obbligatoria per Reti Bayesiane e Multinomial Naive Bayes (default: 3 bin per BN, 5 per NB).

Train-test split stratificato (15% di test).

Struttura del grafo della rete bayesiana definita in base alla presenza di feature note (alcol, volatile acidity, sulphates) o, in assenza, congiungendo le prime e ultime feature al target come fallback.

Stima dei parametri delle CPD tramite massimo di verosimiglianza (MLE) oppure, opzionalmente, con il prior BDeu.

Inferenza condotta con Variable Elimination.

Viene selezionato automaticamente il modello con miglior F1-score pesato sul test (tra Naive Bayes e Bayesian Network); tutti i risultati sono restituiti per confronto.​

**Valutazione**
Metriche adottate: Accuracy e F1-score (weighted) calcolati sul test set o su un sottocampione (per BN, default massimo 100 sample).

Output: confronto esplicito dei punteggi ottenuti, con stampa e ritorno della metrica migliore assieme alle predizioni.

Discussione sui risultati: confronto tra robustezza/semplicità di Naive Bayes e flessibilità/complessità delle Bayesian Network nella modellazione delle dipendenze tra feature.

Tutte le fasi (scelta struttura, fit, predizione, valutazione) sono tracciate nel log a console.

---

## Requisiti funzionali

### Ambiente

- **Linguaggio:** Python 3.10+
- **IDE consigliati:** Visual Studio Code, PyCharm
- **Sistema operativo:** Windows, macOS, Linux

### Librerie principali

**Librerie per Machine Learning e Data Science**
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

**Gestione dataset sbilanciati**
imbalanced-learn>=0.11.0

**Visualizzazione**
matplotlib>=3.7.0
seaborn>=0.12.0

**Modelli probabilistici (Bayesian Networks)**
pgmpy>=0.1.23

**Ottimizzazione Intel (opzionale)**
scikit-learn-intelex>=2023.2.0


---

## Conclusioni


## Riferimenti Bibliografici

[1] **Poole, D. L., & Mackworth, A. K.** (2023). *Artificial Intelligence: Foundations of Computational Agents* (3rd ed.). Cambridge University Press.

[2] **Cortez, P., Cerdeira, A., Alves, F., Matos, T., Reis, J., & Regueira, D.** (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.

---