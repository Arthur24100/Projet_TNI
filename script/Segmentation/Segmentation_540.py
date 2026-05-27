"""
Script de skull stripping pour coupes CORONALES LATÉRALES (540.mat).
Méthode : Watershed avec seeds cerveau / tronc / fond.

Le cerveau = tout l'intérieur du crâne (tissu + tumeur + LCR).
Seeds :
  - Seed 1 (cerveau) = centre de masse du tissu dans la moitié supérieure
  - Seed 2 (tronc)   = bas de la tête
  - Seed 3 (fond)    = pixels noirs hors tête
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import closing, opening, disk, erosion, dilation
from skimage.measure import label, regionprops

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766"

FICHIERS_TEST = ["540.mat"]

SIGMA = 3


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


# ─── SKULL STRIPPING — WATERSHED ─────────────────────────────────────────────

def skull_strip_coronale(img_n, sigma=SIGMA):
    H, W = img_n.shape
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    # 1. Masque tête + boîte englobante
    masque_tete  = _masque_tete(img_lisse)
    prop         = max(regionprops(label(masque_tete)), key=lambda p: p.area)
    min_row, min_col, max_row, max_col = prop.bbox
    hauteur_tete = max_row - min_row
    largeur_tete = max_col - min_col
    cx_tete      = (min_col + max_col) // 2

    # 2. Gradient pour le watershed
    gradient = sobel(gaussian(img_n, sigma=2.0))

    # 3. ✨ Placement des seeds
    markers = np.zeros((H, W), dtype=np.int32)

    # Seed 1 — CERVEAU
    # On érode le masque tête pour être sûr d'être dans le cerveau
    # et on prend le centre de masse dans le tiers supérieur
    masque_erod_seed = erosion(masque_tete, disk(int(min(hauteur_tete, largeur_tete)*0.08)))
    limite_bas_seed  = int(min_row + hauteur_tete * 0.45)
    zone_cerveau     = np.zeros((H, W), dtype=bool)
    zone_cerveau[min_row:limite_bas_seed, min_col:max_col] = True
    zone_cerveau    &= masque_erod_seed

    ys_c, xs_c = np.where(zone_cerveau)
    cy_seed = int(ys_c.mean()) if len(ys_c) > 0 else int(min_row + hauteur_tete * 0.25)
    cx_seed = int(xs_c.mean()) if len(xs_c) > 0 else cx_tete
    r = 15
    markers[max(0,cy_seed-r):min(H,cy_seed+r),
            max(0,cx_seed-r):min(W,cx_seed+r)] = 1

    # Seed 2 — TRONC/COU
    # Bas de la tête érodée (80% vers le bas)
    limite_tronc = int(min_row + hauteur_tete * 0.80)
    zone_tronc   = np.zeros((H, W), dtype=bool)
    zone_tronc[limite_tronc:max_row, min_col:max_col] = True
    zone_tronc  &= masque_tete

    cy_tronc, cx_tronc = 0, 0
    ys_t, xs_t = np.where(zone_tronc)
    if len(ys_t) > 0:
        cy_tronc = int(ys_t.mean())
        cx_tronc = int(xs_t.mean())
        markers[max(0,cy_tronc-r):min(H,cy_tronc+r),
                max(0,cx_tronc-r):min(W,cx_tronc+r)] = 2

    # Seed 3 — FOND
    p3 = np.percentile(img_n, 3)
    markers[(img_n <= p3) & ~masque_tete] = 3

    print(f"    → seed cerveau : ({cy_seed}, {cx_seed})")
    print(f"    → seed tronc   : ({cy_tronc if len(ys_t)>0 else 'N/A'}, {cx_tronc if len(ys_t)>0 else 'N/A'})")

    # 4. Watershed limité au masque tête
    labels_ws    = watershed(gradient, markers, mask=masque_tete)
    masque_ws    = (labels_ws == 1)

    # 5. Nettoyage
    masque_ws = closing(masque_ws, disk(6))
    masque_ws = binary_fill_holes(masque_ws)

    # 6. Garder le composant du seed cerveau
    etiq_ws  = label(masque_ws)
    lbl_seed = etiq_ws[cy_seed, cx_seed]
    if lbl_seed != 0:
        masque_cerveau = binary_fill_holes(etiq_ws == lbl_seed)
    else:
        props_ws = regionprops(etiq_ws)
        masque_cerveau = binary_fill_holes(
            etiq_ws == max(props_ws, key=lambda p: p.area).label) if props_ws else masque_ws

    # 7. Remplissage complet → bloc plein
    masque_cerveau = dilation(masque_cerveau, disk(8))
    masque_cerveau = binary_fill_holes(masque_cerveau)
    masque_cerveau = closing(masque_cerveau, disk(10))
    masque_cerveau = binary_fill_holes(masque_cerveau)
    masque_cerveau = opening(masque_cerveau, disk(2))

    return (img_n * masque_cerveau).astype(np.float32), masque_cerveau, markers, cy_seed, cx_seed


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
        img_stripped, masque, markers, cy_s, cx_s = skull_strip_coronale(img_n)

        rgb = np.stack([img_n] * 3, axis=-1).copy()
        rgb[masque,  1] = np.clip(rgb[masque,  1] + 0.4, 0, 1)
        rgb[mask_gt, 0] = np.clip(rgb[mask_gt, 0] + 0.5, 0, 1)

        pct = 100 * (mask_gt & masque).sum() / (mask_gt.sum() + 1e-8)
        donnees.append((nom, img_n, img_stripped, rgb, pct, markers, cy_s, cx_s))
        print(f"  → tumeur conservée : {pct:.0f}%")

    except FileNotFoundError:
        print(f"  Fichier non trouvé : {chemin}")
        donnees.append((nom, None, None, None, 0, None, 0, 0))
    except Exception as e:
        print(f"  Erreur : {e}")
        donnees.append((nom, None, None, None, 0, None, 0, 0))


# ─── AFFICHAGE ───────────────────────────────────────────────────────────────

def afficher(donnees):
    n = len(donnees)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Skull Stripping — Coupe Coronale — Watershed",
                 fontsize=12, fontweight="bold")

    for i, (nom, img_n, img_s, rgb, pct, markers, cy_s, cx_s) in enumerate(donnees):
        if img_n is None:
            for j in range(4): axes[i, j].axis('off')
            axes[i, 0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
            continue

        # Col 0 : original + seeds
        axes[i, 0].imshow(img_n, cmap='gray')
        if markers is not None:
            axes[i, 0].imshow(np.ma.masked_where(markers == 0, markers),
                             cmap='Set1', alpha=0.6, vmin=1, vmax=3)
        axes[i, 0].set_title(f"{nom} — Seeds (vert=cerveau, rouge=tronc)", fontsize=8)
        axes[i, 0].axis('off')

        # Col 1 : watershed brut
        axes[i, 1].imshow(img_n, cmap='gray')
        axes[i, 1].set_title("Watershed", fontsize=9)
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