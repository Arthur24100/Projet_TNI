"""
Script de test du skull stripping avec Watershed.
- Seed au centre de l'image = cerveau
- S'arrête aux gradients forts = os du crâne
"""

import h5py, numpy as np, os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.morphology import closing, opening, disk
from skimage.measure import label, regionprops

# ─── PARAMÈTRES À AJUSTER ────────────────────────────────────
DOSSIER_BASE = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766"
 
FICHIERS_TEST = [
    "1.mat",
    "9.mat",
    "157.mat",
    "511.mat",
    "540.mat",
    "552.mat",
    "559.mat",
]

SIGMA        = 5    # lissage gaussien avant watershed — augmente pour moins de bruit
SEED_RADIUS  = 10   # taille du seed central en pixels
# ─────────────────────────────────────────────────────────────

def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def skull_strip_watershed(img_n, sigma=SIGMA, seed_radius=SEED_RADIUS):
    H, W = img_n.shape

    # 1. Lissage
    img_lisse = gaussian_filter(img_n, sigma=sigma)

    # 2. Gradient
    gradient = sobel(img_lisse)

    # 3. Détection automatique du centre du cerveau
    # Le cerveau = région avec intensité intermédiaire (ni trop noir, ni trop blanc)
    p10 = np.percentile(img_n, 10)  # fond noir
    p90 = np.percentile(img_n, 90)  # os très brillant
    masque_tissu = (img_n > p10) & (img_n < p90)

    # Centre de masse du tissu cérébral
    ys, xs  = np.where(masque_tissu)
    if len(ys) > 0:
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = H // 2, W // 2

    # 4. Seeds
    markers = np.zeros((H, W), dtype=np.int32)

    # Seed cerveau = zone autour du centre de masse
    r = seed_radius
    markers[max(0,cy-r):min(H,cy+r),
            max(0,cx-r):min(W,cx+r)] = 1

    # Seed fond = pixels très sombres (fond noir garanti)
    markers[img_n < p10 * 0.5] = 2

    # 5. Watershed
    labels = watershed(gradient, markers)

    # 6. Masque cerveau
    masque = (labels == 1)

    # 7. Nettoyage
    masque = closing(masque, disk(5))
    masque = binary_fill_holes(masque)
    masque = opening(masque, disk(3))

    # 8. Plus grand composant
    etiq  = label(masque)
    props = regionprops(etiq)
    if props:
        masque = binary_fill_holes(
            etiq == max(props, key=lambda p: p.area).label)

    return (img_n * masque).astype(np.float32), masque


# ─── Chargement + skull stripping ────────────────────────────
donnees = []
for chemin_rel in FICHIERS_TEST:
    chemin = os.path.join(DOSSIER_BASE, chemin_rel)
    nom    = os.path.basename(chemin)
    print(f"Traitement : {nom}...")

    try:
        with h5py.File(chemin, 'r') as f:
            img     = np.array(f['cjdata']['image']).T.astype(np.float32)
            mask_gt = (np.array(f['cjdata']['tumorMask']).T > 0)

        img_n                = normaliser(img)
        img_stripped, masque = skull_strip_watershed(img_n)

        rgb = np.stack([img_n] * 3, axis=-1).copy()
        rgb[masque,  1] = np.clip(rgb[masque,  1] + 0.4, 0, 1)  # vert = cerveau
        rgb[mask_gt, 0] = np.clip(rgb[mask_gt, 0] + 0.5, 0, 1)  # rouge = tumeur GT

        pct = 100 * (mask_gt & masque).sum() / (mask_gt.sum() + 1e-8)
        donnees.append((nom, img_n, img_stripped, rgb, pct))
        print(f"  → tumeur conservée : {pct:.0f}%")

    except FileNotFoundError:
        print(f"  Fichier non trouvé : {chemin}")
        donnees.append((nom, None, None, None, 0))
    except Exception as e:
        print(f"  Erreur : {e}")
        donnees.append((nom, None, None, None, 0))

# ─── Figure 1 : images 1 à 4 ─────────────────────────────────
fig1, axes1 = plt.subplots(4, 3, figsize=(12, 16))
fig1.suptitle(f"Figure 1/2 — Skull Stripping Watershed  sigma={SIGMA}  seed={SEED_RADIUS}px",
              fontsize=11, fontweight="bold")

for i, (nom, img_n, img_s, rgb, pct) in enumerate(donnees[:4]):
    if img_n is None:
        for j in range(3): axes1[i,j].axis('off')
        axes1[i,0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
        continue
    axes1[i,0].imshow(img_n, cmap='gray'); axes1[i,0].set_title(f"{nom} — Original",              fontsize=9); axes1[i,0].axis('off')
    axes1[i,1].imshow(img_s, cmap='gray'); axes1[i,1].set_title("Cerveau isolé (Watershed)",       fontsize=9); axes1[i,1].axis('off')
    axes1[i,2].imshow(rgb);                axes1[i,2].set_title(f"Overlay — {pct:.0f}% conservée", fontsize=9); axes1[i,2].axis('off')
plt.tight_layout()

# ─── Figure 2 : images 5 à 7 ─────────────────────────────────
fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))
fig2.suptitle(f"Figure 2/2 — Skull Stripping Watershed  sigma={SIGMA}  seed={SEED_RADIUS}px",
              fontsize=11, fontweight="bold")

for i, (nom, img_n, img_s, rgb, pct) in enumerate(donnees[4:]):
    if img_n is None:
        for j in range(3): axes2[i,j].axis('off')
        axes2[i,0].set_title(f"{nom} — NON TROUVÉ", fontsize=9)
        continue
    axes2[i,0].imshow(img_n, cmap='gray'); axes2[i,0].set_title(f"{nom} — Original",              fontsize=9); axes2[i,0].axis('off')
    axes2[i,1].imshow(img_s, cmap='gray'); axes2[i,1].set_title("Cerveau isolé (Watershed)",       fontsize=9); axes2[i,1].axis('off')
    axes2[i,2].imshow(rgb);                axes2[i,2].set_title(f"Overlay — {pct:.0f}% conservée", fontsize=9); axes2[i,2].axis('off')
plt.tight_layout()
plt.show()