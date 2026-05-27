"""
Script de skull stripping universel — Otsu + morphologie + Watershed.
- Détecte automatiquement la coupe (axiale vs sagittale)
- Érosion adaptative basée sur la taille réelle de la tête
- Sélection du composant connexe du seed → élimine artefacts et colonne
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.segmentation import watershed
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import closing, opening, disk, erosion
from skimage.measure import label, regionprops

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766"

FICHIERS_TEST = [
    "1.mat",
    "9.mat",
    "157.mat",
]

SIGMA       = 10   # lissage gaussien (sagittal)
SEED_RADIUS = 30   # taille du seed central en pixels


# ─── UTILITAIRES ─────────────────────────────────────────────────────────────

def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _masque_tete(img_lisse):
    """Retourne le masque de la tête entière (plus grand composant Otsu)."""
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


def _rayon_erosion(masque, facteur=0.09):
    """
    Rayon d'érosion adaptatif.
    facteur : fraction de la plus petite dimension.
              0.07 = léger  |  0.09 = standard  |  0.11 = agressif
    """
    prop = max(regionprops(label(masque)), key=lambda p: p.area)
    min_row, min_col, max_row, max_col = prop.bbox
    hauteur = max_row - min_row
    largeur = max_col - min_col
    return max(5, int(min(hauteur, largeur) * facteur))


def _composant_du_seed(masque, cy, cx):
    """Garde uniquement le composant connexe contenant (cy, cx)."""
    etiq = label(masque)
    lbl  = etiq[cy, cx]
    if lbl != 0:
        return binary_fill_holes(etiq == lbl)
    props = regionprops(etiq)
    if props:
        return binary_fill_holes(etiq == max(props, key=lambda p: p.area).label)
    return masque


def _retirer_petits_artefacts(masque, seuil=0.05):
    """Supprime les composants isolés < seuil × aire du plus grand."""
    etiq  = label(masque)
    props = regionprops(etiq)
    if len(props) <= 1:
        return masque
    masque = masque.copy()
    aire_max = max(p.area for p in props)
    for p in props:
        if p.area < aire_max * seuil:
            masque[etiq == p.label] = False
    return masque


# ─── SKULL STRIPPING AXIAL ───────────────────────────────────────────────────

def skull_strip_axial(img_n, sigma=2):
    """
    Coupe axiale (vue du dessus) : cerveau centré et ovale.
    Otsu → tête entière → érosion adaptative → composant du centre de masse.
    """
    H, W = img_n.shape
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    masque_tete    = _masque_tete(img_lisse)
    rayon          = _rayon_erosion(masque_tete, facteur=0.09)
    masque_cerveau = erosion(masque_tete, disk(rayon))

    # ✨ Centre de masse du tissu cérébral (intensité intermédiaire)
    # plutôt que le centre géométrique fixe H//2, W//2
    # → robuste même quand la coupe est basse (yeux, cervelet)
    p10 = np.percentile(img_n, 10)
    p85 = np.percentile(img_n, 85)  # exclut le crâne très brillant
    masque_tissu = (img_n > p10) & (img_n < p85) & masque_cerveau
    ys, xs = np.where(masque_tissu)
    if len(ys) > 0:
        cy = int(ys.mean())
        cx = int(xs.mean())
    else:
        cy, cx = H // 2, W // 2

    masque_cerveau = _composant_du_seed(masque_cerveau, cy, cx)
    masque_cerveau = closing(masque_cerveau, disk(5))
    masque_cerveau = binary_fill_holes(masque_cerveau)
    masque_cerveau = _retirer_petits_artefacts(masque_cerveau)

    return (img_n * masque_cerveau).astype(np.float32), masque_cerveau


# ─── SKULL STRIPPING SAGITTAL ────────────────────────────────────────────────

def skull_strip_sagittal(img_n, sigma=SIGMA, seed_radius=SEED_RADIUS):
    """
    Coupe sagittale : cerveau décalé, colonne vertébrale visible.
    Otsu → tête → érosion → watershed → composant du seed.
    """
    H, W = img_n.shape
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    masque_tete = _masque_tete(img_lisse)
    rayon       = _rayon_erosion(masque_tete, facteur=0.07)
    masque_erod = erosion(masque_tete, disk(rayon))

    ys, xs = np.where(masque_erod)
    cy = int(ys.mean()) if len(ys) > 0 else H // 2
    cx = int(xs.mean()) if len(xs) > 0 else W // 2

    gradient = sobel(img_lisse)
    p10      = np.percentile(img_n, 10)

    markers = np.zeros((H, W), dtype=np.int32)
    r = seed_radius
    markers[max(0, cy - r):min(H, cy + r),
            max(0, cx - r):min(W, cx + r)] = 1
    markers[img_n < p10 * 0.5] = 2
    markers[~masque_tete]       = 2

    labels_ws  = watershed(gradient, markers, mask=masque_tete)
    masque_ws  = (labels_ws == 1)
    masque_ws  = closing(masque_ws, disk(5))
    masque_ws  = binary_fill_holes(masque_ws)

    masque_final = masque_ws & masque_erod
    masque_final = binary_fill_holes(masque_final)
    masque_final = _composant_du_seed(masque_final, cy, cx)
    masque_final = closing(masque_final, disk(3))
    masque_final = binary_fill_holes(masque_final)
    masque_final = _retirer_petits_artefacts(masque_final)

    return (img_n * masque_final).astype(np.float32), masque_final


# ─── DÉTECTION AUTOMATIQUE DE LA COUPE ──────────────────────────────────────

def skull_strip(img_n, sigma=SIGMA, seed_radius=SEED_RADIUS):
    """
    Détecte automatiquement axiale vs sagittale selon la position du centre
    de masse du tissu par rapport au centre géométrique de l'image.
    """
    H, W      = img_n.shape
    img_lisse = gaussian_filter(img_n, sigma=2)
    thresh    = threshold_otsu(img_lisse)
    masque    = img_lisse > thresh * 0.5

    ys, xs = np.where(masque)
    if len(ys) == 0:
        return img_n.astype(np.float32), np.ones((H, W), dtype=bool)

    cy_rel = ys.mean() / H
    cx_rel = xs.mean() / W

    if (0.30 < cy_rel < 0.70) and (0.30 < cx_rel < 0.70):
        print("    → coupe détectée : AXIALE")
        return skull_strip_axial(img_n, sigma=2)
    else:
        print("    → coupe détectée : SAGITTALE")
        return skull_strip_sagittal(img_n, sigma=sigma, seed_radius=seed_radius)


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
        img_stripped, masque = skull_strip(img_n)

        rgb = np.stack([img_n] * 3, axis=-1).copy()
        rgb[masque,  1] = np.clip(rgb[masque,  1] + 0.4, 0, 1)
        rgb[mask_gt, 0] = np.clip(rgb[mask_gt, 0] + 0.5, 0, 1)

        pct = 100 * (mask_gt & masque).sum() / (mask_gt.sum() + 1e-8)
        donnees.append((nom, img_n, img_stripped, rgb, pct))
        print(f"  → tumeur conservée : {pct:.0f}%")

    except FileNotFoundError:
        print(f"  Fichier non trouvé : {chemin}")
        donnees.append((nom, None, None, None, 0))
    except Exception as e:
        print(f"  Erreur : {e}")
        donnees.append((nom, None, None, None, 0))


# ─── AFFICHAGE (figures adaptées au nombre exact d'images) ───────────────────

def afficher_figure(tranche, titre):
    """Crée une figure avec exactement autant de lignes que d'images."""
    n = len(tranche)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(titre, fontsize=11, fontweight="bold")

    for i, (nom, img_n, img_s, rgb, pct) in enumerate(tranche):
        if img_n is None:
            for j in range(3):
                axes[i, j].axis('off')
            axes[i, 0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
            continue
        axes[i, 0].imshow(img_n, cmap='gray')
        axes[i, 0].set_title(f"{nom} — Original", fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_s, cmap='gray')
        axes[i, 1].set_title("Cerveau isolé", fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rgb)
        axes[i, 2].set_title(f"Overlay — {pct:.0f}% conservée", fontsize=9)
        axes[i, 2].axis('off')

    plt.tight_layout()


PAR_FIG    = 4
N          = len(donnees)
nb_figs    = max(1, (N + PAR_FIG - 1) // PAR_FIG)
titre_base = f"Skull Stripping  sigma={SIGMA}  seed={SEED_RADIUS}px"

for k in range(nb_figs):
    tranche = donnees[k * PAR_FIG:(k + 1) * PAR_FIG]
    afficher_figure(tranche, f"Figure {k+1}/{nb_figs} — {titre_base}")

plt.show()