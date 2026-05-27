"""
Script de skull stripping pour coupes axiales BASSES (niveau yeux / cervelet).

Méthode : K-means (vu en cours TNI) sur les intensités.

Raisonnement anatomique :
  - Fond (noir)         → classe intensité très basse
  - Tissu cérébral      → classe intensité intermédiaire  ← on veut ça
  - Os / crâne          → classe intensité haute (très brillant)
  - Orbites / sinus     → cavités sombres ENTOURÉES d'os brillant
                          → après K-means, les orbites tombent dans la classe
                            "fond" car leur intensité est similaire au fond noir
                          → naturellement exclus sans couper à la règle !

Étapes :
  1. Lissage gaussien
  2. K-means 4 classes sur les pixels de la tête (masque Otsu grossier)
  3. Identifier la classe "tissu cérébral" (intensité intermédiaire)
  4. Nettoyage morphologique + sélection du composant principal
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk, erosion
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766"

FICHIERS_TEST = [
    "1.mat",
]


# ─── UTILITAIRES ─────────────────────────────────────────────────────────────

def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _masque_tete_grossier(img_lisse):
    """Masque binaire grossier de la tête par Otsu — exclut le fond noir."""
    thresh  = threshold_otsu(img_lisse)
    binaire = img_lisse > thresh * 0.4
    binaire = closing(binaire, disk(8))
    binaire = binary_fill_holes(binaire)
    etiq    = label(binaire)
    props   = regionprops(etiq)
    if not props:
        return binaire
    masque = etiq == max(props, key=lambda p: p.area).label
    return binary_fill_holes(masque)


# ─── SKULL STRIPPING COUPE BASSE — K-MEANS ───────────────────────────────────

def skull_strip_coupe_basse(img_n, sigma=2, n_clusters=4):
    """
    Skull stripping pour coupes basses par classification K-means.

    K-means segmente les pixels en N classes d'intensité.
    Les classes sont ensuite triées par intensité croissante :
      Classe 0 : fond noir + orbites sombres  → exclue
      Classe 1 : tissu cérébral gris foncé    → gardée
      Classe 2 : tissu cérébral gris clair /
                 substance blanche            → gardée
      Classe 3 : os cortical très brillant    → exclue
    """
    H, W = img_n.shape

    # 1. Lissage gaussien léger
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    # 2. Masque grossier de la tête (exclut le fond noir de l'IRM)
    masque_tete = _masque_tete_grossier(img_lisse)

    # 3. Extraire uniquement les pixels de la tête pour K-means
    pixels_tete = img_lisse[masque_tete].reshape(-1, 1)

    # 4. K-means avec n_clusters classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_tete)

    # 5. Trier les classes par intensité croissante
    centres        = kmeans.cluster_centers_.flatten()
    ordre          = np.argsort(centres)          # indices triés par intensité
    centres_tries  = centres[ordre]

    # 6. Reconstruire l'image des labels sur toute l'image
    labels_image = np.zeros((H, W), dtype=int)
    labels_tete  = kmeans.labels_
    # Remettre les labels dans l'ordre d'intensité (0=sombre … n-1=brillant)
    remapping = {old: new for new, old in enumerate(ordre)}
    labels_image[masque_tete] = np.array([remapping[l] for l in labels_tete])

    # 7. Identifier les classes "tissu cérébral"
    #    → exclure la classe la plus sombre (fond/orbites) et
    #      la classe la plus brillante (os)
    #    → garder les classes intermédiaires (substance grise + blanche)
    classes_cerveau = list(range(1, n_clusters - 1))   # ex: [1, 2] pour 4 classes

    masque_kmeans = np.zeros((H, W), dtype=bool)
    for c in classes_cerveau:
        masque_kmeans |= (labels_image == c) & masque_tete

    # 8. Fermeture + remplissage des trous (ventricules, sillons)
    masque_kmeans = closing(masque_kmeans, disk(6))
    masque_kmeans = binary_fill_holes(masque_kmeans)

    # 9. Garder uniquement le plus grand composant connexe = cerveau principal
    #    Les orbites, même si elles restent, sont petites et déconnectées
    etiq  = label(masque_kmeans)
    props = regionprops(etiq, intensity_image=img_n)
    if not props:
        return img_n.astype(np.float32), masque_kmeans, labels_image, centres_tries

    # Garder uniquement le PLUS GRAND composant = cerveau principal
    # Les oreilles et artefacts latéraux sont toujours plus petits
    # et physiquement déconnectés après érosion + K-means
    meilleur     = max(props, key=lambda p: p.area)
    masque_final = binary_fill_holes(etiq == meilleur.label)

    # 10. Nettoyage morphologique final
    masque_final = closing(masque_final, disk(8))
    masque_final = binary_fill_holes(masque_final)
    masque_final = opening(masque_final, disk(3))

    return (img_n * masque_final).astype(np.float32), masque_final, labels_image, centres_tries


# ─── CHARGEMENT + TRAITEMENT ─────────────────────────────────────────────────

donnees = []
for chemin_rel in FICHIERS_TEST:
    chemin = os.path.join(DOSSIER_BASE, chemin_rel)
    nom    = os.path.basename(chemin)
    print(f"Traitement : {nom}...")

    try:
        with h5py.File(chemin, 'r') as f:
            img     = np.array(f['cjdata']['image']).T.astype(np.float32)
            mask_gt = (np.array(f['cjdata']['tumorMask']).T > 0)

        img_n = normaliser(img)
        img_stripped, masque, labels_image, centres = skull_strip_coupe_basse(img_n)

        rgb = np.stack([img_n] * 3, axis=-1).copy()
        rgb[masque,  1] = np.clip(rgb[masque,  1] + 0.4, 0, 1)
        rgb[mask_gt, 0] = np.clip(rgb[mask_gt, 0] + 0.5, 0, 1)

        pct = 100 * (mask_gt & masque).sum() / (mask_gt.sum() + 1e-8)
        donnees.append((nom, img_n, labels_image, img_stripped, rgb, pct, centres))
        print(f"  → tumeur conservée : {pct:.0f}%")
        print(f"  → centres K-means (triés) : {np.round(centres, 3)}")

    except FileNotFoundError:
        print(f"  Fichier non trouvé : {chemin}")
        donnees.append((nom, None, None, None, None, 0, None))
    except Exception as e:
        print(f"  Erreur : {e}")
        donnees.append((nom, None, None, None, None, 0, None))


# ─── AFFICHAGE ───────────────────────────────────────────────────────────────

def afficher(donnees):
    n = len(donnees)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Skull Stripping — Coupe Basse — K-means",
                 fontsize=12, fontweight="bold")

    for i, (nom, img_n, labels_image, img_s, rgb, pct, centres) in enumerate(donnees):
        if img_n is None:
            for j in range(4): axes[i, j].axis('off')
            axes[i, 0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
            continue

        axes[i, 0].imshow(img_n, cmap='gray')
        axes[i, 0].set_title(f"{nom} — Original", fontsize=9)
        axes[i, 0].axis('off')

        # K-means coloré : chaque classe = une couleur
        axes[i, 1].imshow(labels_image, cmap='tab10', vmin=0, vmax=9)
        titre_kmeans = "K-means (4 classes)"
        if centres is not None:
            titre_kmeans += f"\n{np.round(centres, 2)}"
        axes[i, 1].set_title(titre_kmeans, fontsize=8)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_s, cmap='gray')
        axes[i, 2].set_title("Cerveau isolé", fontsize=9)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(rgb)
        axes[i, 3].set_title(f"Overlay — {pct:.0f}% conservée", fontsize=9)
        axes[i, 3].axis('off')

    plt.tight_layout()


afficher(donnees)
plt.show()