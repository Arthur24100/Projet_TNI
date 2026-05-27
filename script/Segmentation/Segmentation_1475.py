"""
Script de skull stripping pour coupes CORONALES FRONTALES.
Méthode : Otsu + érosion forte + sélection par compacité et position.
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk, erosion
from skimage.measure import label, regionprops

# ─── PARAMÈTRES ──────────────────────────────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_767-1532"

FICHIERS_TEST = ["1475.mat"]

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


# ─── SKULL STRIPPING CORONALE ────────────────────────────────────────────────

def skull_strip_coronale(img_n, sigma=SIGMA):
    H, W = img_n.shape
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    # 1. Masque tête + boîte englobante
    masque_tete  = _masque_tete(img_lisse)
    prop         = max(regionprops(label(masque_tete)), key=lambda p: p.area)
    min_row, min_col, max_row, max_col = prop.bbox
    hauteur_tete = max_row - min_row
    largeur_tete = max_col - min_col

    # 2. Érosion progressive : on cherche le rayon qui déconnecte
    #    le cou du cerveau sans détruire le cerveau lui-même
    #    On teste de 8% à 18% et on prend le rayon où le cerveau
    #    et le cou deviennent des composants séparés
    masque_cerveau_final = None

    for facteur in [0.10, 0.12, 0.14, 0.16, 0.18]:
        rayon       = max(6, int(min(hauteur_tete, largeur_tete) * facteur))
        masque_erod = erosion(masque_tete, disk(rayon))
        masque_erod = binary_fill_holes(masque_erod)

        etiq  = label(masque_erod)
        props = regionprops(etiq)

        if len(props) < 2:
            # Pas encore déconnecté → érosion plus forte
            continue

        # ✨ On a plusieurs composants → identifier le cerveau
        # Critère : le composant le plus COMPACT et le plus HAUT
        # Compacité = aire / (hauteur × largeur de sa bbox)
        # → le cerveau est dense et ovale, le cou est allongé et creux
        meilleur       = None
        meilleur_score = -1

        for p in props:
            if p.area < masque_tete.sum() * 0.03:
                continue  # trop petit

            bbox_h = p.bbox[2] - p.bbox[0]
            bbox_w = p.bbox[3] - p.bbox[1]
            compacite = p.area / (bbox_h * bbox_w + 1e-8)

            # Position verticale normalisée (0 = tout en haut)
            pos_verticale = (p.centroid[0] - min_row) / hauteur_tete

            # Score : favorise compact ET haut
            score = compacite * np.exp(-pos_verticale * 1.5)

            print(f"      facteur={facteur:.2f} | label={p.label} | "
                  f"aire={p.area} | compacité={compacite:.2f} | "
                  f"pos_v={pos_verticale:.2f} | score={score:.3f}")

            if score > meilleur_score:
                meilleur_score = score
                meilleur       = p

        if meilleur is not None:
            masque_cerveau_final = binary_fill_holes(etiq == meilleur.label)
            print(f"    → cerveau trouvé au facteur {facteur:.2f} "
                  f"(score={meilleur_score:.3f})")
            break

    # Fallback si aucune déconnexion trouvée
    if masque_cerveau_final is None:
        rayon = max(6, int(min(hauteur_tete, largeur_tete) * 0.10))
        masque_cerveau_final = erosion(masque_tete, disk(rayon))
        masque_cerveau_final = binary_fill_holes(masque_cerveau_final)
        print("    → fallback : érosion simple")

    # 3. Remplissage complet → bloc plein homogène
    masque_cerveau_final = closing(masque_cerveau_final, disk(12))
    masque_cerveau_final = binary_fill_holes(masque_cerveau_final)
    masque_cerveau_final = opening(masque_cerveau_final, disk(2))

    return (img_n * masque_cerveau_final).astype(np.float32), masque_cerveau_final


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
        img_stripped, masque = skull_strip_coronale(img_n)

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


# ─── AFFICHAGE ───────────────────────────────────────────────────────────────

def afficher(donnees):
    n = len(donnees)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Skull Stripping — Coupe Coronale  sigma={SIGMA}",
                 fontsize=12, fontweight="bold")

    for i, (nom, img_n, img_s, rgb, pct) in enumerate(donnees):
        if img_n is None:
            for j in range(3): axes[i, j].axis('off')
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


afficher(donnees)
plt.show()