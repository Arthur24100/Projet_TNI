"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        PIPELINE DÉTECTION TUMEUR CÉRÉBRALE — RF + SVM                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. PHASE ENTRAÎNEMENT (une seule fois)                                    ║
║     → 200 images aléatoires des deux dossiers dataset                      ║
║     → Extraction features par pixel (intensité, texture, gradient)         ║
║     → Entraînement Random Forest + SVM                                     ║
║     → Sauvegarde modèles (.joblib)                                         ║
║                                                                              ║
║  2. PHASE TEST / UTILISATION                                                ║
║     → L'utilisateur choisit une image de test                              ║
║     → Choisit le type de coupe (1-4)                                       ║
║     → Skull stripping → extraction features → prédiction RF + SVM         ║
║     → Affichage : original | tumeur RF | tumeur SVM | Dice comparatif      ║
╚══════════════════════════════════════════════════════════════════════════════╝

UTILISATION :
    # Phase 1 — entraîner les modèles (à faire une seule fois)
    python tumor_pipeline.py --train

    # Phase 2 — tester sur une image
    python tumor_pipeline.py --test

    # Tout faire d'un coup
    python tumor_pipeline.py --train --test
"""

import h5py
import numpy as np
import os
import sys
import warnings
import argparse
import joblib
import csv
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.segmentation import watershed, active_contour
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.morphology import (closing, opening, disk, erosion,
                                 remove_small_objects, remove_small_holes)
from skimage.measure import label, regionprops
from skimage.draw import polygon
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶▶  PARAMÈTRES À MODIFIER ICI  ◀◀◀
# ══════════════════════════════════════════════════════════════════════════════

# Dossiers contenant les fichiers .mat
DOSSIERS_DATASET = [
    "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766",
    "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_767-1532",
]

# Dossier où sauvegarder/charger les modèles entraînés
DOSSIER_MODELES = "./modeles_tumeur"

# Fichier CSV des images labellisées (produit par labellisation_coupes.py)
FICHIER_CSV = "./coupes.csv"

# Nombre d'images pour entraînement et test
NB_TRAIN = 51
NB_TEST  = 5

# Graine aléatoire pour reproductibilité
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def charger_mat(chemin):
    """Charge image + masque GT depuis un .mat HDF5."""
    with h5py.File(chemin, 'r') as f:
        img = np.array(f['cjdata']['image']).T.astype(np.float32)
        try:
            mask_gt = (np.array(f['cjdata']['tumorMask']).T > 0)
        except KeyError:
            mask_gt = None
    return normaliser(img), mask_gt


def lister_fichiers(dossiers):
    """Liste tous les .mat disponibles dans les dossiers donnés."""
    fichiers = []
    for d in dossiers:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.mat'):
                    fichiers.append(os.path.join(d, f))
    return fichiers


def lire_csv(fichier_csv):
    """
    Lit coupes.csv et retourne un dict {chemin_fichier: methode_int}.
    Ignore les lignes avec methode=0.
    """
    donnees = {}
    if not os.path.isfile(fichier_csv):
        print(f"  ✗ CSV non trouvé : {fichier_csv}")
        return donnees
    with open(fichier_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['methode'].strip()
            if m in ('1', '2', '3', '4'):
                donnees[row['fichier'].strip()] = int(m)
    return donnees


def split_train_test(fichiers, nb_train, nb_test, seed=SEED):
    """Tire aléatoirement nb_train + nb_test fichiers sans remise."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(fichiers), size=nb_train + nb_test, replace=False)
    train_idx = indices[:nb_train]
    test_idx  = indices[nb_train:]
    return [fichiers[i] for i in train_idx], [fichiers[i] for i in test_idx]


# ──────────────────────────────────────────────────────────────────────────────
#  SKULL STRIPPING
# ──────────────────────────────────────────────────────────────────────────────

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

def _rayon_erosion(masque, facteur=0.09):
    prop = max(regionprops(label(masque)), key=lambda p: p.area)
    h = prop.bbox[2] - prop.bbox[0]
    w = prop.bbox[3] - prop.bbox[1]
    return max(5, int(min(h, w) * facteur))

def _composant_du_seed(masque, cy, cx):
    etiq = label(masque)
    lbl  = etiq[cy, cx]
    if lbl != 0:
        return binary_fill_holes(etiq == lbl)
    props = regionprops(etiq)
    if props:
        return binary_fill_holes(etiq == max(props, key=lambda p: p.area).label)
    return masque

def _retirer_petits_artefacts(masque, seuil=0.05):
    etiq  = label(masque)
    props = regionprops(etiq)
    if len(props) <= 1:
        return masque
    masque   = masque.copy()
    aire_max = max(p.area for p in props)
    for p in props:
        if p.area < aire_max * seuil:
            masque[etiq == p.label] = False
    return masque

def skull_strip_axiale(img_n, sigma=2):
    H, W       = img_n.shape
    img_lisse  = gaussian_filter(img_n, sigma=sigma)
    mt         = _masque_tete(img_lisse)
    rayon      = _rayon_erosion(mt, 0.09)
    mc         = erosion(mt, disk(rayon))
    p10, p85   = np.percentile(img_n, 10), np.percentile(img_n, 85)
    tissu      = (img_n > p10) & (img_n < p85) & mc
    ys, xs     = np.where(tissu)
    cy = int(ys.mean()) if len(ys) > 0 else H // 2
    cx = int(xs.mean()) if len(xs) > 0 else W // 2
    mc = _composant_du_seed(mc, cy, cx)
    mc = closing(mc, disk(5))
    mc = binary_fill_holes(mc)
    mc = _retirer_petits_artefacts(mc)
    return (img_n * mc).astype(np.float32), mc

def skull_strip_sagittale(img_n, sigma=10, seed_radius=30):
    H, W       = img_n.shape
    img_lisse  = gaussian_filter(img_n, sigma=sigma)
    mt         = _masque_tete(img_lisse)
    rayon      = _rayon_erosion(mt, 0.07)
    me         = erosion(mt, disk(rayon))
    ys, xs     = np.where(me)
    cy = int(ys.mean()) if len(ys) > 0 else H // 2
    cx = int(xs.mean()) if len(xs) > 0 else W // 2
    gradient   = sobel(img_lisse)
    p10        = np.percentile(img_n, 10)
    markers    = np.zeros((H, W), dtype=np.int32)
    r          = seed_radius
    markers[max(0,cy-r):min(H,cy+r), max(0,cx-r):min(W,cx+r)] = 1
    markers[img_n < p10 * 0.5] = 2
    markers[~mt] = 2
    lws        = watershed(gradient, markers, mask=mt)
    mws        = closing(binary_fill_holes(lws == 1), disk(5))
    mf         = binary_fill_holes(mws & me)
    mf         = _composant_du_seed(mf, cy, cx)
    mf         = binary_fill_holes(closing(mf, disk(3)))
    mf         = _retirer_petits_artefacts(mf)
    return (img_n * mf).astype(np.float32), mf

def skull_strip_coronale(img_n, sigma=5):
    H, W       = img_n.shape
    img_lisse  = gaussian_filter(img_n, sigma=sigma)
    mt         = _masque_tete(img_lisse)
    prop       = max(regionprops(label(mt)), key=lambda p: p.area)
    r0,c0,r1,c1 = prop.bbox
    ht, wt     = r1-r0, c1-c0
    cy_e       = r0 + ht * 0.42
    cx_e       = (c0 + c1) / 2
    t          = np.linspace(0, 2*np.pi, 400)
    ellipse    = np.array([cy_e + ht*0.38*np.sin(t),
                           cx_e + wt*0.42*np.cos(t)]).T
    img_s      = gaussian(img_n, sigma=2.0)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snake = active_contour(img_s, ellipse, alpha=0.05, beta=10.0,
                               gamma=0.001, max_num_iter=500, convergence=0.001)
    rr, cc = polygon(snake[:,0], snake[:,1], shape=(H,W))
    ms     = np.zeros((H,W), dtype=bool)
    ms[rr,cc] = True
    ms     = binary_fill_holes(ms) & mt
    p96    = np.percentile(img_n[mt], 96)
    ms     = ms & (img_n < p96)
    rayon  = max(3, int(min(ht,wt)*0.03))
    ms     = erosion(ms, disk(rayon))
    etiq   = label(ms); props = regionprops(etiq)
    if props:
        ms = binary_fill_holes(etiq == max(props, key=lambda p: p.area).label)
    ms = closing(ms, disk(10))
    ms = binary_fill_holes(ms)
    ms = opening(ms, disk(2))
    return (img_n * ms).astype(np.float32), ms, ellipse, snake

def skull_strip_watershed(img_n, sigma=10, seed_radius=30):
    H, W       = img_n.shape
    img_lisse  = gaussian_filter(img_n, sigma=sigma)
    thresh     = threshold_otsu(img_lisse)
    mb         = closing(binary_fill_holes(img_lisse > thresh*0.5), disk(10))
    etiq       = label(mb); props = regionprops(etiq)
    if not props:
        return img_n.astype(np.float32), np.ones((H,W), dtype=bool)
    mt         = binary_fill_holes(etiq == max(props, key=lambda p: p.area).label)
    prop       = max(regionprops(label(mt)), key=lambda p: p.area)
    h = prop.bbox[2]-prop.bbox[0]; w = prop.bbox[3]-prop.bbox[1]
    mc         = erosion(mt, disk(max(4, int(min(h,w)*0.06))))
    gradient   = sobel(img_lisse)
    p10        = np.percentile(img_n, 10)
    ys, xs     = np.where(mc)
    cy = int(ys.mean()) if len(ys) > 0 else H//2
    cx = int(xs.mean()) if len(xs) > 0 else W//2
    markers    = np.zeros((H,W), dtype=np.int32)
    r          = seed_radius
    markers[max(0,cy-r):min(H,cy+r), max(0,cx-r):min(W,cx+r)] = 1
    markers[img_n < p10*0.5] = 2
    markers[~mt] = 2
    lws        = watershed(gradient, markers, mask=mt)
    mws        = binary_fill_holes(closing(lws==1, disk(5)))
    mf         = binary_fill_holes(mws & mc)
    etiq2      = label(mf)
    ls         = etiq2[cy, cx]
    mf         = (etiq2 == ls) if ls != 0 else \
                 (etiq2 == max(regionprops(etiq2), key=lambda p: p.area).label
                  if regionprops(etiq2) else mf)
    mf         = binary_fill_holes(closing(mf, disk(3)))
    return (img_n * mf).astype(np.float32), mf

def skull_strip(img_n, methode):
    """Applique la méthode de skull stripping choisie."""
    if methode == 1:
        return skull_strip_axiale(img_n) + (None, None)
    elif methode == 2:
        return skull_strip_sagittale(img_n) + (None, None)
    elif methode == 3:
        return skull_strip_coronale(img_n)   # retourne 4 valeurs
    elif methode == 4:
        return skull_strip_watershed(img_n) + (None, None)
    else:
        raise ValueError(f"Méthode {methode} invalide (1-4)")


# ──────────────────────────────────────────────────────────────────────────────
#  EXTRACTION DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def extraire_features(img_n, masque_cerveau):
    """
    Extrait un vecteur de features pour chaque pixel du cerveau.

    Features par pixel (9 au total) :
      [0]   Intensité normalisée
      [1]   Gradient Sobel (bords)
      [2]   Intensité lissée (gaussien σ=2)
      [3]   Contraste local (différence avec moyenne locale, fenêtre 5×5)
      [4]   LBP (Local Binary Pattern) — texture locale
      [5]   GLCM contraste    (calculé sur fenêtre 9×9)
      [6]   GLCM homogénéité
      [7]   Intensité lissée σ=5  (contexte plus large)
      [8]   Écart-type local (fenêtre 5×5)
    """
    from scipy.ndimage import uniform_filter, generic_filter

    H, W = img_n.shape

    # ── Features de base ─────────────────────────────────────────────────────
    f0_intensite = img_n.copy()
    f1_gradient  = sobel(img_n)
    f2_lisse2    = gaussian_filter(img_n, sigma=2)
    f7_lisse5    = gaussian_filter(img_n, sigma=5)

    # ── Contraste local (I - moyenne locale 5×5) ─────────────────────────────
    moyenne_loc  = uniform_filter(img_n, size=5)
    f3_contraste = img_n - moyenne_loc

    # ── Écart-type local (fenêtre 5×5) ───────────────────────────────────────
    moyenne2_loc = uniform_filter(img_n**2, size=5)
    f8_std_loc   = np.sqrt(np.maximum(moyenne2_loc - moyenne_loc**2, 0))

    # ── LBP (texture) ────────────────────────────────────────────────────────
    img_uint8 = (img_n * 255).astype(np.uint8)
    f4_lbp    = local_binary_pattern(img_uint8, P=8, R=1,
                                      method='uniform').astype(np.float32)
    f4_lbp    = f4_lbp / (f4_lbp.max() + 1e-8)

    # ── GLCM sur fenêtre glissante 9×9 (contraste + homogénéité) ─────────────
    # Pour accélérer : on calcule pixel par pixel sur fenêtre réduite
    f5_glcm_contrast = np.zeros((H, W), dtype=np.float32)
    f6_glcm_homo     = np.zeros((H, W), dtype=np.float32)
    pad = 4
    img_p = np.pad(img_uint8, pad, mode='reflect')

    # Sous-échantillonnage pour accélérer (calcul tous les 3 pixels)
    step = 3
    for y in range(0, H, step):
        for x in range(0, W, step):
            if not masque_cerveau[y, x]:
                continue
            patch = img_p[y:y+2*pad+1, x:x+2*pad+1]
            try:
                glcm = graycomatrix(patch, distances=[1],
                                    angles=[0, np.pi/4, np.pi/2],
                                    levels=64, symmetric=True, normed=True)
                c = float(graycoprops(glcm, 'contrast').mean())
                h = float(graycoprops(glcm, 'homogeneity').mean())
            except Exception:
                c, h = 0.0, 1.0
            # Remplir le voisinage step×step
            y2 = min(y + step, H)
            x2 = min(x + step, W)
            f5_glcm_contrast[y:y2, x:x2] = c
            f6_glcm_homo[y:y2, x:x2]     = h

    # Normaliser GLCM
    f5_glcm_contrast = f5_glcm_contrast / (f5_glcm_contrast.max() + 1e-8)

    # ── Empiler en matrice (N_pixels × 9) ────────────────────────────────────
    idx = np.where(masque_cerveau)
    features = np.stack([
        f0_intensite[idx],
        f1_gradient [idx],
        f2_lisse2   [idx],
        f3_contraste[idx],
        f4_lbp      [idx],
        f5_glcm_contrast[idx],
        f6_glcm_homo    [idx],
        f7_lisse5   [idx],
        f8_std_loc  [idx],
    ], axis=1).astype(np.float32)

    return features, idx


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────────────────────

def phase_entrainement(donnees_csv):
    """
    Charge les images depuis coupes.csv, applique le bon skull stripping
    pour chaque image et entraîne RF + SVM.
    donnees_csv : dict {chemin: methode_int} lu depuis coupes.csv
    """
    os.makedirs(DOSSIER_MODELES, exist_ok=True)

    fichiers_train = list(donnees_csv.keys())
    print(f"\n{'═'*60}")
    print(f"  PHASE ENTRAÎNEMENT — {len(fichiers_train)} images (depuis coupes.csv)")
    print(f"{'═'*60}")

    X_all, y_all = [], []
    n_ok = 0

    for i, chemin in enumerate(fichiers_train):
        nom     = os.path.basename(chemin)
        methode = donnees_csv[chemin]
        print(f"  [{i+1:3d}/{len(fichiers_train)}] {nom} (méthode {methode})", end=' ', flush=True)

        try:
            img_n, mask_gt = charger_mat(chemin)
            if mask_gt is None or mask_gt.sum() == 0:
                print("→ pas de GT, ignoré")
                continue

            # Skull stripping avec la méthode assignée dans le CSV
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = skull_strip(img_n, methode)
            img_stripped, masque_cerveau = result[0], result[1]

            # Extraction features
            features, idx = extraire_features(img_n, masque_cerveau)
            labels        = mask_gt[idx].astype(int)

            # Équilibrage : max 500 pixels tumeur + 500 pixels sains par image
            # (évite le déséquilibre de classes)
            idx_tumeur = np.where(labels == 1)[0]
            idx_sain   = np.where(labels == 0)[0]
            n_samp     = min(500, len(idx_tumeur), len(idx_sain))

            if n_samp < 10:
                print("→ tumeur trop petite, ignoré")
                continue

            rng = np.random.default_rng(SEED + i)
            sel_t = rng.choice(idx_tumeur, size=n_samp, replace=False)
            sel_s = rng.choice(idx_sain,   size=n_samp, replace=False)
            sel   = np.concatenate([sel_t, sel_s])

            X_all.append(features[sel])
            y_all.append(labels[sel])
            n_ok += 1
            print(f"→ ✓ ({n_samp*2} pixels)")

        except Exception as e:
            print(f"→ erreur : {e}")

    if n_ok == 0:
        print("  ✗ Aucune image valide pour l'entraînement !")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\n  Dataset : {X.shape[0]} pixels  "
          f"({y.sum()} tumeur / {(y==0).sum()} sain)")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n  Entraînement Random Forest...")
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(DOSSIER_MODELES, 'random_forest.joblib'))
    print(f"  ✓ RF sauvegardé")

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("  Entraînement SVM (peut prendre quelques minutes)...")
    # On sous-échantillonne pour SVM (lent sur gros datasets)
    max_svm = 20000
    if len(X) > max_svm:
        idx_sub = np.random.default_rng(SEED).choice(len(X), max_svm, replace=False)
        X_svm, y_svm = X[idx_sub], y[idx_sub]
        print(f"    (sous-échantillonnage SVM : {max_svm} pixels)")
    else:
        X_svm, y_svm = X, y

    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=False,
            random_state=SEED
        ))
    ])
    svm.fit(X_svm, y_svm)
    joblib.dump(svm, os.path.join(DOSSIER_MODELES, 'svm.joblib'))
    print(f"  ✓ SVM sauvegardé")

    # Sauvegarder la liste des fichiers de test
    print(f"\n  ✓ Entraînement terminé — modèles dans '{DOSSIER_MODELES}/'")


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — PRÉDICTION SUR UNE IMAGE
# ──────────────────────────────────────────────────────────────────────────────

def predire(img_n, masque_cerveau, modele):
    """Prédit le masque tumeur pixel par pixel avec le modèle donné."""
    H, W = img_n.shape
    features, idx = extraire_features(img_n, masque_cerveau)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = modele.predict(features)

    masque_pred = np.zeros((H, W), dtype=bool)
    masque_pred[idx] = y_pred.astype(bool)

    # Affinage morphologique
    masque_pred = closing(masque_pred, disk(3))
    masque_pred = remove_small_objects(masque_pred, min_size=50)
    masque_pred = binary_fill_holes(masque_pred)

    return masque_pred


def calculer_metriques(masque_pred, mask_gt):
    """Dice, Précision, Rappel."""
    inter     = (masque_pred & mask_gt).sum()
    dice      = 2 * inter / (masque_pred.sum() + mask_gt.sum() + 1e-8)
    precision = inter / (masque_pred.sum() + 1e-8)
    rappel    = inter / (mask_gt.sum() + 1e-8)
    return {
        "Dice": float(dice),
        "Précision": float(precision),
        "Rappel": float(rappel),
        "TP": int(inter),
        "FP": int(masque_pred.sum() - inter),
        "FN": int(mask_gt.sum() - inter),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  AFFICHAGE
# ──────────────────────────────────────────────────────────────────────────────

def afficher_resultats(nom, methode, img_n, img_stripped,
                       masque_rf, masque_svm,
                       mask_gt, met_rf, met_svm):

    noms_methode = {
        1: "Axiale", 2: "Sagittale",
        3: "Coronale (Snake)", 4: "Watershed"
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"Détection tumeur — Méthode skull stripping : {noms_methode[methode]} | {nom}",
        fontsize=13, fontweight="bold"
    )

    def overlay(img, masque, couleur, alpha=0.6):
        rgb = np.stack([img]*3, axis=-1).copy()
        if couleur == 'rouge':
            rgb[masque, 0] = np.clip(rgb[masque, 0] + alpha, 0, 1)
        elif couleur == 'bleu':
            rgb[masque, 2] = np.clip(rgb[masque, 2] + alpha, 0, 1)
        elif couleur == 'vert':
            rgb[masque, 1] = np.clip(rgb[masque, 1] + alpha, 0, 1)
        return rgb

    # 1. Image originale
    axes[0].imshow(img_n, cmap='gray')
    axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis('off')

    # 2. Cerveau isolé
    axes[1].imshow(img_stripped, cmap='gray')
    axes[1].set_title("Cerveau isolé\n(skull stripping)", fontsize=11)
    axes[1].axis('off')

    # 3. Random Forest
    rgb_rf = overlay(img_n, masque_rf, 'rouge')
    if mask_gt is not None:
        rgb_rf = np.stack([img_n]*3, axis=-1).copy()
        rgb_rf[mask_gt,  1] = np.clip(rgb_rf[mask_gt,  1] + 0.5, 0, 1)
        rgb_rf[masque_rf, 0] = np.clip(rgb_rf[masque_rf, 0] + 0.5, 0, 1)
    axes[2].imshow(rgb_rf)
    titre_rf = "Random Forest\n(rouge=prédit | vert=GT)"
    if met_rf:
        titre_rf += (f"\nDice={met_rf['Dice']:.3f}  "
                     f"Préc={met_rf['Précision']:.3f}  "
                     f"Rappel={met_rf['Rappel']:.3f}")
    axes[2].set_title(titre_rf, fontsize=10)
    axes[2].axis('off')

    # 4. SVM
    rgb_svm = np.stack([img_n]*3, axis=-1).copy()
    if mask_gt is not None:
        rgb_svm[mask_gt,   1] = np.clip(rgb_svm[mask_gt,   1] + 0.5, 0, 1)
    rgb_svm[masque_svm, 0] = np.clip(rgb_svm[masque_svm, 0] + 0.5, 0, 1)
    axes[3].imshow(rgb_svm)
    titre_svm = "SVM\n(rouge=prédit | vert=GT)"
    if met_svm:
        titre_svm += (f"\nDice={met_svm['Dice']:.3f}  "
                      f"Préc={met_svm['Précision']:.3f}  "
                      f"Rappel={met_svm['Rappel']:.3f}")
    axes[3].set_title(titre_svm, fontsize=10)
    axes[3].axis('off')

    # 5. Comparaison RF vs SVM côte à côte
    rgb_comp = np.stack([img_n]*3, axis=-1).copy()
    rgb_comp[masque_rf,  0] = np.clip(rgb_comp[masque_rf,  0] + 0.5, 0, 1)  # rouge = RF
    rgb_comp[masque_svm, 2] = np.clip(rgb_comp[masque_svm, 2] + 0.5, 0, 1)  # bleu  = SVM
    if mask_gt is not None:
        rgb_comp[mask_gt, 1] = np.clip(rgb_comp[mask_gt, 1] + 0.4, 0, 1)    # vert  = GT
    axes[4].imshow(rgb_comp)
    axes[4].set_title("Comparaison\nRouge=RF | Bleu=SVM | Vert=GT", fontsize=10)
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()


def afficher_bilan(nom, met_rf, met_svm):
    print(f"\n{'═'*55}")
    print(f"  RÉSULTATS — {nom}")
    print(f"{'═'*55}")
    print(f"  {'Métrique':<12} {'Random Forest':>15} {'SVM':>15}")
    print(f"  {'-'*42}")
    for k in ('Dice', 'Précision', 'Rappel'):
        rf_v  = f"{met_rf[k]:.4f}"  if met_rf  else "N/A"
        svm_v = f"{met_svm[k]:.4f}" if met_svm else "N/A"
        emoji = ''
        if met_rf and met_svm:
            emoji = ' ✓' if max(met_rf[k], met_svm[k]) > 0.6 else ' ⚠'
        print(f"  {k:<12} {rf_v:>15} {svm_v:>15}{emoji}")
    print(f"{'═'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  PHASE TEST
# ──────────────────────────────────────────────────────────────────────────────

def afficher_image_originale(img_n, nom, numero, total):
    """
    Affiche l'image originale en BLOQUANT jusqu'à fermeture de la fenêtre.
    L'utilisateur ferme la fenêtre, PUIS choisit la coupe dans le terminal.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(
        f"Image {numero}/{total} — {nom}\n"
        f"→ Fermez cette fenêtre puis choisissez la coupe dans le terminal",
        fontsize=11, fontweight="bold", color="darkred"
    )
    ax.imshow(img_n, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)   # bloque ici jusqu'à fermeture manuelle
    plt.close('all')


def phase_test_batch(fichiers_test):
    """
    Teste RF + SVM sur les 30 images de test.
    Pour chaque image : affiche l'original, l'utilisateur choisit la coupe,
    puis affiche les résultats RF + SVM + Dice.
    À la fin : Dice moyen comparatif RF vs SVM.
    """
    chemin_rf  = os.path.join(DOSSIER_MODELES, 'random_forest.joblib')
    chemin_svm = os.path.join(DOSSIER_MODELES, 'svm.joblib')

    if not os.path.isfile(chemin_rf) or not os.path.isfile(chemin_svm):
        print("  ✗ Modèles non trouvés. Lance d'abord : python tumor_pipeline.py --train")
        return

    print(f"\n{'═'*60}")
    print(f"  PHASE TEST — {len(fichiers_test)} images")
    print(f"  Pour chaque image, choisissez la coupe dans le terminal.")
    print(f"{'═'*60}")

    rf  = joblib.load(chemin_rf)
    svm = joblib.load(chemin_svm)

    dices_rf, dices_svm = [], []
    total = len(fichiers_test)

    for i, chemin in enumerate(fichiers_test):
        nom = os.path.basename(chemin)
        print(f"\n  ── Image {i+1}/{total} : {nom} ──")

        try:
            img_n, mask_gt = charger_mat(chemin)
        except Exception as e:
            print(f"  ✗ Erreur chargement : {e} — ignorée")
            continue

        # Afficher l'image pour que l'utilisateur voie ce qu'il traite
        afficher_image_originale(img_n, nom, i+1, total)

        # L'utilisateur choisit la coupe pour CETTE image
        print("  Méthode de skull stripping :")
        print("    1 → Coupe AXIALE       (Otsu + morphologie)")
        print("    2 → Coupe SAGITTALE    (Otsu + Watershed)")
        print("    3 → Coupe CORONALE     (Active Contour / Snake)")
        print("    4 → Watershed AMÉLIORÉ")
        while True:
            try:
                methode = int(input("  Votre choix (1-4) : ").strip())
                if methode in (1, 2, 3, 4):
                    break
                print("  ⚠  Entrez 1, 2, 3 ou 4.")
            except ValueError:
                print("  ⚠  Entrez un nombre entier.")

        # Skull stripping
        print("  → Skull stripping...", end=' ', flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = skull_strip(img_n, methode)
            img_stripped, masque_cerveau = result[0], result[1]
            print("✓")
        except Exception as e:
            print(f"✗ : {e} — ignorée")
            continue

        # Prédiction
        print("  → Prédiction RF...",  end=' ', flush=True)
        masque_rf  = predire(img_n, masque_cerveau, rf)
        print("✓")
        print("  → Prédiction SVM...", end=' ', flush=True)
        masque_svm = predire(img_n, masque_cerveau, svm)
        print("✓")

        # Métriques
        met_rf = met_svm = None
        if mask_gt is not None:
            met_rf  = calculer_metriques(masque_rf,  mask_gt)
            met_svm = calculer_metriques(masque_svm, mask_gt)
            dices_rf.append(met_rf['Dice'])
            dices_svm.append(met_svm['Dice'])
            afficher_bilan(nom, met_rf, met_svm)

        # Affichage résultats
        afficher_resultats(
            nom, methode, img_n, img_stripped,
            masque_rf, masque_svm,
            mask_gt, met_rf, met_svm
        )

    # ── Bilan final ───────────────────────────────────────────────────────────
    if dices_rf:
        print(f"\n{'═'*55}")
        print(f"  BILAN FINAL — {len(dices_rf)} images évaluées")
        print(f"{'═'*55}")
        print(f"  {'':<12} {'Random Forest':>15} {'SVM':>15}")
        print(f"  {'-'*42}")
        print(f"  {'Dice moyen':<12} {np.mean(dices_rf):>15.4f} {np.mean(dices_svm):>15.4f}")
        print(f"  {'Dice std':<12} {np.std(dices_rf):>15.4f}  {np.std(dices_svm):>15.4f}")
        print(f"  {'Dice min':<12} {np.min(dices_rf):>15.4f}  {np.min(dices_svm):>15.4f}")
        print(f"  {'Dice max':<12} {np.max(dices_rf):>15.4f}  {np.max(dices_svm):>15.4f}")
        winner = "Random Forest" if np.mean(dices_rf) > np.mean(dices_svm) else "SVM"
        print(f"\n  Meilleur modèle : {winner} ✓")
        print(f"{'═'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  ÉVALUATION SUR LES 30 IMAGES DE TEST
# ──────────────────────────────────────────────────────────────────────────────

def evaluer_sur_test(fichiers_test):
    """Évalue RF et SVM sur les 30 images de test et affiche les Dice moyens."""
    chemin_rf  = os.path.join(DOSSIER_MODELES, 'random_forest.joblib')
    chemin_svm = os.path.join(DOSSIER_MODELES, 'svm.joblib')

    if not os.path.isfile(chemin_rf) or not os.path.isfile(chemin_svm):
        print("  ✗ Modèles non trouvés.")
        return

    rf  = joblib.load(chemin_rf)
    svm = joblib.load(chemin_svm)

    dices_rf, dices_svm = [], []

    print(f"\n{'═'*65}")
    print(f"  ÉVALUATION SUR {len(fichiers_test)} IMAGES DE TEST")
    print(f"{'═'*65}")
    print(f"  {'Fichier':<15} {'Dice RF':>10} {'Dice SVM':>10}")
    print(f"  {'-'*35}")

    for chemin in fichiers_test:
        nom = os.path.basename(chemin)
        try:
            img_n, mask_gt = charger_mat(chemin)
            if mask_gt is None or mask_gt.sum() == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img_stripped, masque_cerveau = skull_strip_axiale(img_n)
            mrf  = predire(img_n, masque_cerveau, rf)
            msvm = predire(img_n, masque_cerveau, svm)
            d_rf  = calculer_metriques(mrf,  mask_gt)['Dice']
            d_svm = calculer_metriques(msvm, mask_gt)['Dice']
            dices_rf.append(d_rf)
            dices_svm.append(d_svm)
            print(f"  {nom:<15} {d_rf:>10.3f} {d_svm:>10.3f}")
        except Exception as e:
            print(f"  {nom:<15} {'erreur':>10}")

    if dices_rf:
        print(f"  {'-'*35}")
        print(f"  {'MOYENNE':<15} {np.mean(dices_rf):>10.3f} {np.mean(dices_svm):>10.3f}")
        print(f"  {'STD':<15} {np.std(dices_rf):>10.3f} {np.std(dices_svm):>10.3f}")
    print(f"{'═'*65}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════════════════════════════╗
║      PIPELINE DÉTECTION TUMEUR — RF + SVM                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Skull stripping :                                              ║
║    1 → Coupe AXIALE       (Otsu + morphologie)                  ║
║    2 → Coupe SAGITTALE    (Otsu + Watershed)                    ║
║    3 → Coupe CORONALE     (Active Contour / Snake)              ║
║    4 → Watershed AMÉLIORÉ                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline détection tumeur cérébrale RF + SVM")
    parser.add_argument('--train',    action='store_true',
                        help="Entraîner les modèles sur 200 images")
    parser.add_argument('--test',     action='store_true',
                        help="Tester sur l'image définie dans CHEMIN_IMAGE_TEST")
    parser.add_argument('--evaluate', action='store_true',
                        help="Évaluer sur les 30 images de test (Dice moyen)")
    args = parser.parse_args()

    # Par défaut : si aucun argument → tout faire
    if not args.train and not args.test and not args.evaluate:
        args.train = args.test = True

    print(MENU)

    # ── Lecture du CSV de labellisation ─────────────────────────────────────
    donnees_csv = lire_csv(FICHIER_CSV)
    if not donnees_csv:
        print(f"  ✗ Aucune image dans {FICHIER_CSV}")
        print("     → Lance d'abord labellisation_coupes.py")
        return
    print(f"  {len(donnees_csv)} images labellisées dans coupes.csv")

    # ── Images de test : tirées aléatoirement parmi celles PAS dans le CSV ───
    fichiers_tous = lister_fichiers(DOSSIERS_DATASET)
    fichiers_hors_csv = [f for f in fichiers_tous if f not in donnees_csv]
    rng = np.random.default_rng(SEED)
    nb_test = min(NB_TEST, len(fichiers_hors_csv))
    fichiers_test = list(rng.choice(fichiers_hors_csv, nb_test, replace=False))
    print(f"  {len(fichiers_test)} images de test tirées aléatoirement (hors CSV)")

    # Sauvegarde de la liste de test pour référence
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    with open(os.path.join(DOSSIER_MODELES, 'fichiers_test.txt'), 'w') as f:
        f.write('\n'.join(fichiers_test))

    # ── Entraînement ─────────────────────────────────────────────────────────
    if args.train:
        phase_entrainement(donnees_csv)

    # ── Évaluation sur les 30 images de test ─────────────────────────────────
    if args.evaluate:
        evaluer_sur_test(fichiers_test)

    # ── Test sur les 30 images aléatoires ──────────────────────────────────
    if args.test:
        # L'utilisateur choisit la méthode de skull stripping une seule fois
        print("\n  Choisissez la méthode de skull stripping :")
        print("    1 → Coupe AXIALE       (Otsu + morphologie)")
        print("    2 → Coupe SAGITTALE    (Otsu + Watershed)")
        print("    3 → Coupe CORONALE     (Active Contour / Snake)")
        print("    4 → Watershed AMÉLIORÉ")
        while True:
            try:
                methode = int(input("\n  Votre choix (1-4) : ").strip())
                if methode in (1, 2, 3, 4):
                    break
                print("  ⚠  Entrez 1, 2, 3 ou 4.")
            except ValueError:
                print("  ⚠  Entrez un nombre entier.")

        phase_test_batch(fichiers_test)


if __name__ == "__main__":
    main()