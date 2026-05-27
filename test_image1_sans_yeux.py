import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian, threshold_otsu
from skimage.draw import polygon
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes

# ============================================================
# 1. CHARGEMENT
# ============================================================
def charger_mat_v73(path):
    with h5py.File(path, 'r') as f:
        img    = np.array(f['cjdata']['image']).astype(np.float32).T
        mask_t = np.array(f['cjdata']['tumorMask']).astype(np.uint8).T
        return img, mask_t


# ============================================================
# 2. NORMALISATION
# ============================================================
def normaliser(img_raw):
    return cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ============================================================
# 3. SUPPRESSION DU CRÂNE : Otsu → ellipse → snake → soustraction
# ============================================================
def supprimer_crane(img_8u,
                    sigma_snake=3,
                    alpha=0.01, beta=5, gamma=0.001,
                    rayon_dilation=15,
                    seuil_crane=170):
    h, w = img_8u.shape

    # Otsu → plus grande région → fill holes
    img_blur = cv2.GaussianBlur(img_8u, (5, 5), 0)
    seuil    = threshold_otsu(img_blur)
    mask_bin = (img_blur > seuil).astype(np.uint8)
    labels   = measure.label(mask_bin)
    regions  = measure.regionprops(labels)
    plus_grande = max(regions, key=lambda r: r.area)
    mask_brain  = (labels == plus_grande.label).astype(np.uint8)
    mask_brain  = binary_fill_holes(mask_brain).astype(np.uint8)

    # Ellipse ajustée sur le masque (init snake)
    contours, _ = cv2.findContours(mask_brain, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy_e), (ax, ay), angle = cv2.fitEllipse(cnt)
    s      = np.linspace(0, 2 * np.pi, 400)
    r_init = cy_e + (ay / 2 * 0.92) * np.sin(s)
    c_init = cx   + (ax / 2 * 0.92) * np.cos(s)
    init_contour = np.array([r_init, c_init]).T

    # Snake
    img_smooth = gaussian(img_8u, sigma=sigma_snake)
    snake = active_contour(img_smooth, init_contour,
                           alpha=alpha, beta=beta, gamma=gamma)

    # Masque intérieur snake = cerveau
    mask_interieur = np.zeros((h, w), dtype=np.uint8)
    rr, cc = polygon(snake[:, 0], snake[:, 1], (h, w))
    mask_interieur[rr, cc] = 1

    # Anneau crâne = entre Otsu et snake, dilaté
    masque_crane = ((mask_brain == 1) & (mask_interieur == 0)).astype(np.uint8)
    masque_crane = morphology.dilation(masque_crane, morphology.disk(rayon_dilation))
    masque_crane = (masque_crane * mask_brain).astype(np.uint8)

    img_sans_crane = img_8u.copy()
    img_sans_crane[masque_crane == 1] = 0

    print(f"Crâne supprimé : {masque_crane.sum()} pixels")
    return img_sans_crane, masque_crane, init_contour, snake


# ============================================================
# 4. SUPPRESSION DES ORBITES sur IMAGE ORIGINALE
# ============================================================
def supprimer_orbites(img_original_8u, img_a_nettoyer,
                      seuil_gris=150,
                      seuil_taille_min=100,
                      seuil_taille_max=10000,
                      seuil_position=0.28,
                      debug=True):
    """
    Détecte sur img_original_8u (image brute), supprime dans img_a_nettoyer.
    4 critères :
      1. Position haute    : cy < h * 0.28
      2. Taille cohérente  : 100 < aire < 10000
      3. Pas trop latérale : 15% < cx < 85%
      4. Forme compacte    : ratio L/H < 4
    """
    h, w = img_original_8u.shape

    _, img_thresh     = cv2.threshold(img_original_8u, seuil_gris, 255, cv2.THRESH_BINARY)
    img_thresh_dilate = morphology.dilation(img_thresh, morphology.disk(10))

    labels  = measure.label(img_thresh_dilate)
    regions = measure.regionprops(labels)
    masque_orbites = np.zeros_like(img_original_8u, dtype=np.uint8)

    if debug:
        print(f"\n{'Label':>6} | {'Aire':>7} | {'cy':>6} | {'cx':>6} | {'Ratio':>6} | {'Décision'}")
        print("-" * 70)

    for region in regions:
        if region.area < seuil_taille_min:
            continue

        cy_r, cx_r  = region.centroid
        en_avant     = cy_r < h * seuil_position
        bonne_taille = seuil_taille_min < region.area < seuil_taille_max
        pas_lateral  = 0.15 * w < cx_r < 0.85 * w

        minr, minc, maxr, maxc = region.bbox
        ratio       = (maxc - minc) / (maxr - minr + 1e-5)
        pas_allonge = ratio < 4.0

        est_orbite = en_avant and bonne_taille and pas_lateral and pas_allonge

        if debug:
            tag = "✓ ORBITE" if est_orbite else "✗ gardée"
            print(f"{region.label:>6} | {region.area:>7.0f} | {cy_r:>6.1f} | {cx_r:>6.1f} | {ratio:>6.1f} | {tag}")

        if est_orbite:
            masque_orbites[labels == region.label] = 1

    masque_orbites = morphology.erosion(masque_orbites, morphology.disk(3))
    img_finale = img_a_nettoyer.copy()
    img_finale[masque_orbites == 1] = 0

    return img_finale, masque_orbites


# ============================================================
# 5. AFFICHAGE
# ============================================================
def afficher_resultats(img_raw, init_contour, snake,
                       img_sans_crane, masque_crane,
                       img_8u, masque_orbites,
                       img_final, tumor_truth):
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    axes[0].imshow(img_raw, cmap='gray')
    axes[0].plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=2, label="Ellipse init")
    axes[0].plot(snake[:, 1], snake[:, 0], '-b', lw=2, label="Snake final")
    axes[0].set_title("1. Ellipse → Snake")
    axes[0].legend(loc='upper right', fontsize=7)

    axes[1].imshow(img_sans_crane, cmap='gray')
    axes[1].contour(masque_crane, colors='cyan', linewidths=1.2)
    axes[1].contour(tumor_truth, colors='red', linewidths=1.5)
    axes[1].set_title("2. Crâne supprimé (cyan)")

    axes[2].imshow(img_8u, cmap='gray')
    if masque_orbites.any():
        axes[2].contour(masque_orbites, colors='yellow', linewidths=1.5)
    axes[2].contour(tumor_truth, colors='red', linewidths=1.5)
    axes[2].set_title("3. Orbites détectées\n(sur image originale)")

    axes[3].imshow(img_final, cmap='gray')
    axes[3].contour(tumor_truth, colors='red', linewidths=1.5)
    axes[3].set_title("4. Résultat final\n(crâne + orbites supprimés)")

    axes[4].imshow(img_raw, cmap='gray')
    axes[4].contour(tumor_truth, colors='red', linewidths=1.5)
    axes[4].set_title("5. Image originale (référence)")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. PIPELINE PRINCIPAL  —  paramètres figés
# ============================================================
if __name__ == "__main__":

    PATH = './brainTumorDataPublic_1-766/1.mat'
    img_raw, tumor_truth = charger_mat_v73(PATH)
    img_8u = normaliser(img_raw)

    # ── Étape 1 : suppression du crâne ──────────────────────
    # rayon_dilation=15 → run donnant 13031 pixels supprimés (meilleur résultat)
    img_sans_crane, masque_crane, init_contour, snake = supprimer_crane(
        img_8u,
        sigma_snake=3,
        alpha=0.01,
        beta=5,
        gamma=0.001,
        rayon_dilation=15,
        seuil_crane=170
    )

    # ── Étape 2 : suppression des orbites ───────────────────
    # Détection sur image originale (img_8u), disk(10) pour fusionner fragments
    img_final, masque_orbites = supprimer_orbites(
        img_original_8u=img_8u,
        img_a_nettoyer=img_sans_crane,
        seuil_gris=150,
        seuil_taille_min=100,
        seuil_taille_max=10000,
        seuil_position=0.28,
        debug=True
    )

    afficher_resultats(img_raw, init_contour, snake,
                       img_sans_crane, masque_crane,
                       img_8u, masque_orbites,
                       img_final, tumor_truth)