"""
Étape 3 — Correction du biais d'intensité (N4ITK)
Algorithme officiel recommandé par le prof.

Installation :
    pip install SimpleITK

Pourquoi N4ITK ?
    Certaines images du dataset ont un gradient d'intensité dû à l'antenne IRM.
    Le même tissu peut apparaître clair d'un côté et sombre de l'autre.
    N4ITK estime et corrige ce gradient automatiquement.
"""

import glob, zlib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import median_filter

# ─── Loader ──────────────────────────────────────────────────

def lire_mat(chemin):
    with open(chemin, "rb") as f:
        f.seek(512)
        raw = f.read()
    chunks, i = [], 0
    while i < len(raw) - 2:
        if raw[i] == 0x78 and raw[i+1] in (0x01, 0x5E, 0x9C, 0xDA):
            for ml in [5000, 10000, 20000, 40000, 80000]:
                try:
                    d = zlib.decompress(raw[i:i+ml])
                    if len(d) == 65536:
                        chunks.append(d); break
                except: pass
            i += 100
        else:
            i += 1
    image = np.hstack([np.frombuffer(c, dtype="<u2").reshape(512, 64)  for c in chunks[:8]])
    mask  = np.hstack([np.frombuffer(c, dtype=np.uint8).reshape(512, 128) for c in chunks[8:12]])
    return image, (mask > 0).astype(np.uint8)

def n4itk(image_numpy):
    """
    Applique la correction N4ITK sur une image numpy 2D.

    Entrée  : image float32 normalisée [0, 1]
    Sortie  : image corrigée float32 [0, 1]
    """
    # Convertir numpy → SimpleITK
    image_sitk = sitk.GetImageFromArray(image_numpy.astype(np.float32))

    # Créer un masque : on ne corrige que les pixels non-noirs (le cerveau)
    # Otsu sépare automatiquement fond / tissu
    masque_sitk = sitk.OtsuThreshold(image_sitk, 0, 1, 200)

    # Convertir en float32 (N4ITK le demande)
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

    # Appliquer N4ITK
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])  # 4 niveaux, 50 iter chacun
    image_corrigee_sitk = corrector.Execute(image_sitk, masque_sitk)

    # Récupérer le champ de biais estimé
    biais_log = corrector.GetLogBiasFieldAsImage(image_sitk)
    biais = sitk.GetArrayFromImage(sitk.Exp(biais_log))

    # Convertir SimpleITK → numpy
    image_corrigee = sitk.GetArrayFromImage(image_corrigee_sitk).astype(np.float32)

    # Renormaliser entre 0 et 1
    image_corrigee = (image_corrigee - image_corrigee.min()) / (image_corrigee.max() - image_corrigee.min() + 1e-8)

    return image_corrigee, biais


# ─── Étapes précédentes ───────────────────────────────────────

chemin = glob.glob("5-Projet/Dataset/1512427/brainTumorDataPublic_1-766/1.mat")[0]
image, mask = lire_mat(chemin)

# Étape 1 : normalisation
image_norm = (image.astype(np.float32) - image.min()) / (image.max() - image.min())

# Étape 2 : filtrage médian
image_filtre = median_filter(image_norm, size=3)


# ─── Étape 3 : N4ITK ─────────────────────────────────────────

print("Application de N4ITK... (peut prendre quelques secondes)")
image_corrigee, biais = n4itk(image_filtre)
print("N4ITK terminé.")


# ─── Affichage ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Étape 3 — Correction du biais N4ITK", fontsize=13, fontweight="bold")

# Ligne 1 : images complètes
axes[0, 0].imshow(image_filtre,   cmap="gray")
axes[0, 0].set_title("Avant correction")
axes[0, 0].axis("off")

axes[0, 1].imshow(biais, cmap="hot")
axes[0, 1].set_title("Champ de biais estimé\n(zones claires = biais fort)")
axes[0, 1].axis("off")

axes[0, 2].imshow(image_corrigee, cmap="gray")
axes[0, 2].set_title("Après correction N4ITK")
axes[0, 2].axis("off")

# Ligne 2 : zoom centre pour voir la différence
cx, cy, t = 256, 256, 120
for ax, img, titre in zip(
    axes[1],
    [image_filtre, biais, image_corrigee],
    ["Zoom — Avant", "Zoom — Biais", "Zoom — Après"]
):
    ax.imshow(img[cy-t:cy+t, cx-t:cx+t], cmap="gray")
    ax.set_title(titre, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("etape3_n4itk.png", dpi=120, bbox_inches="tight")
plt.show()
print("Sauvegardé : etape3_n4itk.png")

# ─── Stats ───────────────────────────────────────────────────
print(f"\nAvant  : min={image_filtre.min():.3f}  max={image_filtre.max():.3f}  moy={image_filtre.mean():.3f}")
print(f"Après  : min={image_corrigee.min():.3f}  max={image_corrigee.max():.3f}  moy={image_corrigee.mean():.3f}")