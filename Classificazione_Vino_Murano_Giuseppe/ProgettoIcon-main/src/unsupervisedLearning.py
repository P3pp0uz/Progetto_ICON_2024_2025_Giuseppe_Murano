"""
Modulo per clustering non supervisionato con K-Means e
selezione del numero di cluster k tramite metodo del gomito (elbow) su WSS.
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Fit K-Means solo su training set
def fit_kmeans_train_only(X_train, k, random_state=42):
    km = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
    km.fit(X_train)
    return km


# Estrazione feature K-Means
def kmeans_features(km, X):
    labels = km.predict(X)
    onehot = np.eye(km.n_clusters)[labels]
    dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1).reshape(-1, 1)
    return labels, onehot, dists


# Calcolo WSS per range di k
def _compute_wss(X_scaled, k_range):
    wss = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        km.fit(X_scaled)
        wss.append(km.inertia_)
    return np.array(wss)


# Heuristica elbow per selezione k
def _elbow_k(k_range, wss):
    # Heuristica: punto di massima curvatura (metodo del "gomito")
    k_vals = np.array(list(k_range), dtype=float)
    x = (k_vals - k_vals.min()) / (k_vals.max() - k_vals.min() + 1e-12)
    y = (wss - wss.min()) / (wss.max() - wss.min() + 1e-12)
    # distanza dal segmento che unisce estremi (0, y0) e (1, y1)
    p1 = np.array([0.0, y[0]])
    p2 = np.array([1.0, y[-1]])
    num = np.abs((p2[1] - p1[1]) * x - (p2[0] - p1[0]) * y + p2[0] * p1[1] - p2[1] * p1[0])
    den = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) + 1e-12
    d = num / den
    elbow_idx = int(np.argmax(d))
    return int(k_range[elbow_idx])


# K-Means con selezione del numero di cluster tramite metodo del gomito su WSS
def kmeans_clustering(X, k_min=2, k_max=11):
    """
    Clustering K-Means con scelta di k tramite profilo SSE (WSS) ed heuristica del gomito.
    """
    print("\n" + "=" * 80)
    print("CLUSTERING NON SUPERVISIONATO - Selezione k via Elbow (WSS)")
    print("=" * 80)

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Calcolo WSS e selezione k_elbow
    k_range = range(k_min, k_max)
    wss = _compute_wss(X_scaled, k_range)
    k_final = _elbow_k(k_range, wss)
    print(f"\nSelezione modello (Elbow su WSS): k_elbow ≈ {k_final}")

    # Fit K-Means con k_final
    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)

    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wss, 'o-', lw=2)
    plt.axvline(k_final, color='tab:blue', ls='--', label=f'k_elbow={k_final}')
    plt.xlabel('k')
    plt.ylabel('WSS (inertia)')
    plt.title('Elbow plot (WSS)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualization/k_wss_elbow.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\n[OK] Grafico elbow salvato: k_wss_elbow.png")

    # PCA per visualizzazione
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=14, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, label='Cluster')
    plt.xlabel(f'PC1 ({var[0] * 100:.1f}% var)')
    plt.ylabel(f'PC2 ({var[1] * 100:.1f}% var)')
    plt.title(f'K-Means (k={k_final}) - PCA')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualization/clustering_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Visualizzazione PCA salvata: clustering_visualization.png")

    # PCA 3D per visualizzazione
    from mpl_toolkits.mplot3d import Axes3D

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    var_3d = pca_3d.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                         c=labels, cmap='tab10', s=20, alpha=0.6, edgecolors='none')

    ax.set_xlabel(f'PC1 ({var_3d[0] * 100:.1f}% var)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_3d[1] * 100:.1f}% var)', fontweight='bold')
    ax.set_zlabel(f'PC3 ({var_3d[2] * 100:.1f}% var)', fontweight='bold')
    ax.set_title(f'K-Means (k={k_final}) - PCA 3D', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, label='Cluster', pad=0.1, shrink=0.8)

    # Aggiungi angolo di visualizzazione migliore
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('visualization/clustering_visualization_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Visualizzazione PCA 3D salvata: clustering_visualization_3d.png")

    # Ritorna anche il modello PCA 3D
    return {
        'cluster_labels': labels,
        'n_clusters': k_final,
        'k_elbow': k_final,
        'scaler': scaler,
        'kmeans_model': kmeans,
        'pca_model': pca,
        'pca_model_3d': pca_3d,
        'wss': wss
    }


