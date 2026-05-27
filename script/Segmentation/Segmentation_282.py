"""
Script de skull stripping pour coupes CORONALES FRONTALES.
Méthode : Active Contour / Snake (vu en cours TNI ESEO).

Stratégie :
  1. Otsu + érosion → masque tête pour localiser le cerveau
  2. Initialisation ellipse dans la moitié SUPÉRIEURE de la tête
     → le snake ne peut pas descendre sur le visage
  3. Snake converge vers les bords du crâne
  4. Remplissage du contour final → masque cerveau plein
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.segmentation import active_contour
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing, opening, disk, erosion
from skimage.measure import label, regionprops
from skimage.draw import polygon

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766"
# brainTumorDataPublic_767-1532
# brainTumorDataPublic_1-766
FICHIERS_TEST = [
    "282.mat",
]

# 1475

SIGMA = 5


# ─── UTILITAIRES ─────────────────────────────────────────────────────────────

def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _masque_tete(img_lisse):
    thresh  = threshold_otsu(img_lisse)
    binaire = img_lisse > thresh * 0.5
    binaire = closing(binaire, disk(8))
    binaire = binary_fill_holes(binaire)
    etiq    = label(binaire)
    props   = regionprops(etiq)
    if not props:
        return binaire
    masque = etiq == max(props, key=lambda p: p.area).label
    return binary_fill_holes(masque)


def contour_vers_masque(snake, H, W):
    """Convertit le contour snake (N×2) en masque binaire rempli."""
    rr, cc  = polygon(snake[:, 0], snake[:, 1], shape=(H, W))
    masque  = np.zeros((H, W), dtype=bool)
    masque[rr, cc] = True
    return binary_fill_holes(masque)


# ─── SKULL STRIPPING CORONALE — SNAKE ────────────────────────────────────────

def skull_strip_coronale(img_n, sigma=SIGMA):
    """
    Skull stripping coupe coronale par Active Contour (Snake).

    Initialisation : ellipse dans la moitié supérieure de la tête
    → le contour actif converge vers le crâne osseux
    → ne descend pas sur le visage/cou car initialisé trop haut

    Paramètres Snake (selon cours TNI) :
    - alpha : élasticité (force interne) — maintient le contour lisse
    - beta  : rigidité (force interne)  — évite les oscillations
    - gamma : vitesse de convergence
    """
    H, W = img_n.shape

    # 1. Lissage pour réduire le bruit avant le snake
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    # 2. Masque tête entière → boîte englobante du cerveau
    masque_tete  = _masque_tete(img_lisse)
    prop         = max(regionprops(label(masque_tete)), key=lambda p: p.area)
    min_row, min_col, max_row, max_col = prop.bbox
    hauteur_tete = max_row - min_row
    largeur_tete = max_col - min_col
    cx_tete      = (min_col + max_col) / 2

    # 3. Centre de l'ellipse = dans la moitié supérieure de la tête
    #    Assez bas pour couvrir tout le cerveau incluant les lobes temporaux
    cy_ellipse = min_row + hauteur_tete * 0.42
    cx_ellipse = cx_tete

    # Demi-axes : couvre ~85% de la largeur et ~65% de la hauteur cérébrale
    a_axe = largeur_tete * 0.42   # demi-axe horizontal — plus large
    b_axe = hauteur_tete * 0.38   # demi-axe vertical — plus grand pour couvrir les lobes

    # 4. Initialisation du contour en ellipse (400 points comme dans le cours)
    t            = np.linspace(0, 2 * np.pi, 400)
    ellipse_init = np.array([
        cy_ellipse + b_axe * np.sin(t),   # coordonnées lignes (y)
        cx_ellipse + a_axe * np.cos(t)    # coordonnées colonnes (x)
    ]).T

    # 5. Lissage de l'image pour le snake (améliore les gradients)
    img_snake = gaussian(img_n, sigma=2.0)

    # 6. Active Contour (Snake) — paramètres du cours TNI
    # np.errstate supprime les warnings numériques connus de skimage/active_contour
    import warnings
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'),          warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        snake = active_contour(
            img_snake,
            ellipse_init,
            alpha=0.05,      # élasticité : faible → suit les bords irréguliers
            beta=10.0,       # rigidité : élevée → contour lisse
            gamma=0.001,     # vitesse de convergence
            max_num_iter=500,
            convergence=0.001
        )

    # 7. Convertir le contour en masque binaire rempli
    masque_snake = contour_vers_masque(snake, H, W)

    # 8. Intersection avec le masque tête (sécurité : évite de déborder)
    masque_cerveau = masque_snake & masque_tete

    # 9. ✨ Retirer l'os cortical brillant (blanc sur les bords)
    #    L'os a une intensité > p88 de la tête → on l'exclut du masque
    p88 = np.percentile(img_n[masque_tete], 96)
    masque_cerveau = masque_cerveau & (img_n < p88)

    # 10. Érosion légère → retire la fine couche de crâne résiduelle
    rayon = max(3, int(min(hauteur_tete, largeur_tete) * 0.03))
    masque_cerveau = erosion(masque_cerveau, disk(rayon))

    # 11. Garder uniquement le plus grand composant = cerveau
    etiq  = label(masque_cerveau)
    props = regionprops(etiq)
    if props:
        masque_cerveau = binary_fill_holes(
            etiq == max(props, key=lambda p: p.area).label)

    # 12. Remplissage complet → bloc plein homogène
    masque_cerveau = closing(masque_cerveau, disk(10))
    masque_cerveau = binary_fill_holes(masque_cerveau)
    masque_cerveau = opening(masque_cerveau, disk(2))

    return (img_n * masque_cerveau).astype(np.float32), masque_cerveau, ellipse_init, snake


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
        img_stripped, masque, ellipse_init, snake = skull_strip_coronale(img_n)

        rgb = np.stack([img_n] * 3, axis=-1).copy()
        rgb[masque,  1] = np.clip(rgb[masque,  1] + 0.4, 0, 1)
        rgb[mask_gt, 0] = np.clip(rgb[mask_gt, 0] + 0.5, 0, 1)

        pct = 100 * (mask_gt & masque).sum() / (mask_gt.sum() + 1e-8)
        donnees.append((nom, img_n, img_stripped, rgb, pct, ellipse_init, snake))
        print(f"  → tumeur conservée : {pct:.0f}%")

    except FileNotFoundError:
        print(f"  Fichier non trouvé : {chemin}")
        donnees.append((nom, None, None, None, 0, None, None))
    except Exception as e:
        print(f"  Erreur : {e}")
        donnees.append((nom, None, None, None, 0, None, None))


# ─── AFFICHAGE ───────────────────────────────────────────────────────────────

def afficher(donnees):
    n = len(donnees)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Skull Stripping — Coupe Coronale — Active Contour (Snake)",
                 fontsize=12, fontweight="bold")

    for i, (nom, img_n, img_s, rgb, pct, ellipse_init, snake) in enumerate(donnees):
        if img_n is None:
            for j in range(4): axes[i, j].axis('off')
            axes[i, 0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
            continue

        # Col 0 : original
        axes[i, 0].imshow(img_n, cmap='gray')
        axes[i, 0].set_title(f"{nom} — Original", fontsize=9)
        axes[i, 0].axis('off')

        # Col 1 : contour initial (ellipse rouge) + contour final (snake bleu)
        axes[i, 1].imshow(img_n, cmap='gray')
        if ellipse_init is not None:
            axes[i, 1].plot(ellipse_init[:, 1], ellipse_init[:, 0],
                           '--r', lw=1.5, label='Init ellipse')
        if snake is not None:
            axes[i, 1].plot(snake[:, 1], snake[:, 0],
                           '-b', lw=2, label='Snake final')
        axes[i, 1].set_title("Contour initial (rouge)\nSnake final (bleu)", fontsize=8)
        axes[i, 1].axis('off')

        # Col 2 : cerveau isolé
        axes[i, 2].imshow(img_s, cmap='gray')
        axes[i, 2].set_title("Cerveau isolé", fontsize=9)
        axes[i, 2].axis('off')

        # Col 3 : overlay
        axes[i, 3].imshow(rgb)
        axes[i, 3].set_title(f"Overlay — {pct:.0f}% conservée", fontsize=9)
        axes[i, 3].axis('off')

    plt.tight_layout()


afficher(donnees)
plt.show()