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
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import sobel, laplace
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

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

    return image_corrigee

def normaliser(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())

def variance_locale(img, sigma=3):
    moy  = gaussian_filter(img,    sigma=sigma)
    moy2 = gaussian_filter(img**2, sigma=sigma)
    return np.sqrt(np.maximum(moy2 - moy**2, 0))

# ─── Étapes précédentes ───────────────────────────────────────

chemin = glob.glob("5-Projet/Dataset/1512427/brainTumorDataPublic_1-766/1.mat")[0]
image, mask = lire_mat(chemin)

# ══════════════════════════════════════════════════════════════
#  PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════

# Étape 1 — Normalisation
image_norm = normaliser(image)
 
# Étape 2 — Filtrage médian
image_filtre = median_filter(image_norm, size=3)
 
# Étape 3 — N4ITK
image_n4 = n4itk(image_filtre)

# ══════════════════════════════════════════════════════════════
#  EXTRACTION — CARACTÉRISTIQUES MORPHOLOGIQUES
# ══════════════════════════════════════════════════════════════
 
print("Extraction morphologique...")
 
f1  = normaliser(image_n4.copy())
f2  = normaliser(sobel(image_n4))
f3  = normaliser(np.abs(laplace(image_n4)))
f4  = normaliser(gaussian_filter(image_n4, sigma=2))
f5  = normaliser(variance_locale(image_n4, sigma=3))
 
print("  → 5 caractéristiques morphologiques OK")
 
 
# ══════════════════════════════════════════════════════════════
#  EXTRACTION — TEXTURE (LBP)
# ══════════════════════════════════════════════════════════════
 
print("Extraction LBP...")
 
image_uint8 = (image_n4 * 255).astype(np.uint8)
lbp = local_binary_pattern(image_uint8, P=8, R=1, method="uniform")
f6  = normaliser(lbp)
 
print("  → LBP OK")
 
 
# ══════════════════════════════════════════════════════════════
#  EXTRACTION — TEXTURE (GLCM)
# ══════════════════════════════════════════════════════════════
 
print("Extraction GLCM (peut prendre 1-2 minutes)...")
 
N_NIVEAUX      = 32
TAILLE_FENETRE = 8        # réduit de 16 à 8
pas            = 4        # pas de 4 au lieu de 8
H, W           = image_n4.shape
image_reduite  = (image_n4 * (N_NIVEAUX - 1)).astype(np.uint8)
 
f7  = np.zeros((H, W), dtype=np.float32)
f8  = np.zeros((H, W), dtype=np.float32)
f9  = np.zeros((H, W), dtype=np.float32)
 
for r in range(0, H - TAILLE_FENETRE + 1, pas):
    for c in range(0, W - TAILLE_FENETRE + 1, pas):
        fen  = image_reduite[r:r+TAILLE_FENETRE, c:c+TAILLE_FENETRE]
        glcm = graycomatrix(fen, distances=[1], angles=[0],
                            levels=N_NIVEAUX, symmetric=True, normed=True)
        f7[r:r+TAILLE_FENETRE, c:c+TAILLE_FENETRE] = graycoprops(glcm, "contrast")[0, 0]
        f8[r:r+TAILLE_FENETRE, c:c+TAILLE_FENETRE] = graycoprops(glcm, "homogeneity")[0, 0]
        f9[r:r+TAILLE_FENETRE, c:c+TAILLE_FENETRE] = graycoprops(glcm, "energy")[0, 0]
        # Corrélation supprimée car trop bruitée
 
f7 = normaliser(f7)
f8 = normaliser(f8)
f9 = normaliser(f9)
print("GLCM OK\n")
 
caracteristiques = {
    "1. Intensité":        f1,
    "2. Gradient":         f2,
    "3. Laplacien":        f3,
    "4. Moyenne locale":   f4,
    "5. Variance locale":  f5,
    "6. LBP":              f6,
    "7. GLCM Contraste":   f7,
    "8. GLCM Homogénéité": f8,
    "9. GLCM Énergie":     f9,
}
 
 
# ══════════════════════════════════════════════════════════════
#  VALIDATION : tumeur vs tissu sain
#  C'est la SEULE fois où on utilise le masque
# ══════════════════════════════════════════════════════════════
 
pixels_tumeur = mask.ravel() == 1
pixels_sain   = mask.ravel() == 0
 
print("─" * 55)
print(f"  {'Caractéristique':<22} {'Sain':>8} {'Tumeur':>8}  {'Séparation':>10}")
print("─" * 55)
 
scores = {}
for nom, carte in caracteristiques.items():
    vals = carte.ravel()
    moy_sain   = vals[pixels_sain].mean()
    moy_tumeur = vals[pixels_tumeur].mean()
    std_sain   = vals[pixels_sain].std()
    std_tumeur = vals[pixels_tumeur].std()
 
    # Score de séparation : différence des moyennes / moyenne des écarts-types
    # Plus c'est grand, mieux la caractéristique sépare les deux classes
    separation = abs(moy_tumeur - moy_sain) / ((std_sain + std_tumeur) / 2 + 1e-8)
    scores[nom] = separation
 
    barre = "★" * int(separation * 5)
    print(f"  {nom:<22} {moy_sain:>8.3f} {moy_tumeur:>8.3f}  {barre}")
 
print("─" * 55)
meilleure = max(scores, key=scores.get)
print(f"\n  Meilleure caractéristique : {meilleure}")
 
 
# ─── Affichage : histogrammes tumeur vs sain ──────────────────
 
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Séparation tumeur (rouge) vs tissu sain (bleu)\npour chaque caractéristique",
             fontsize=13, fontweight="bold")
 
for ax, (nom, carte) in zip(axes.flat, caracteristiques.items()):
    vals = carte.ravel()
    ax.hist(vals[pixels_sain],   bins=60, alpha=0.6, color="steelblue",
            label="Sain",   density=True)
    ax.hist(vals[pixels_tumeur], bins=60, alpha=0.7, color="red",
            label="Tumeur", density=True)
    ax.set_title(f"{nom}\nséparation={scores[nom]:.3f}", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_yticks([])
 
plt.tight_layout()
plt.show()
print("\nSauvegardé : validation_caracteristiques.png")

# On observe que plusieurs parameètre ne servent pas car ils sont condondu entre le sain et la tumeur grace a la validation du masque