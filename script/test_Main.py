import h5py
import numpy as np
import os
import warnings
import argparse
import joblib
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from scipy.ndimage import binary_fill_holes, gaussian_filter, distance_transform_edt
from skimage.segmentation import watershed, active_contour
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.morphology import (closing, opening, disk, erosion,
                                remove_small_objects)
from skimage.measure import label, regionprops
from skimage.draw import polygon
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# paramètre

DOSSIERS_DATASET = [
    "/Users/paloma/Desktop/25-26/traitement_image/projet/matlab_projet/brainTumorDataPublic_1-766",
    "/Users/paloma/Desktop/25-26/traitement_image/projet/matlab_projet/brainTumorDataPublic_767-1532",
    "/Users/paloma/Desktop/25-26/traitement_image/projet/matlab_projet/brainTumorDataPublic_1533-2298",
    "/Users/paloma/Desktop/25-26/traitement_image/projet/matlab_projet/brainTumorDataPublic_2299-3064"
]

DOSSIER_MODELES = "./modeles_tumeur"
FICHIER_CSV     = "./coupes.csv"
NB_TEST         = 5
SEED            = 42
MAX_PIX_TUMEUR  = 500
SEUIL_RF        = 0.50  

# chargement 

def normaliser(img):
    """
    Normalise une image entre 0 et 1.
    Entree  : img (numpy array float)
    Sortie  : image normalisee entre 0 et 1
    """
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def charger_mat(chemin):
    """
    Charge un fichier .mat et retourne l'image,
    le masque ground truth et le label de type de tumeur.
    Entree  : chemin vers le fichier .mat
    Sortie  : (img_n, mask_gt, label_tumor)
              img_n : image normalisee (float32)
              mask_gt : masque binaire de la tumeur (bool) ou None
              label_tumor : type de tumeur : 1=meningiome, 2=gliome, 3=pituitaire
    """
    with h5py.File(chemin, 'r') as f:
        img = np.array(f['cjdata']['image']).T.astype(np.float32)
        try:
            mask_gt = (np.array(f['cjdata']['tumorMask']).T > 0)
        except KeyError:
            mask_gt = None
        try:
            label_tumor = int(np.array(f['cjdata']['label']).flat[0])
        except KeyError:
            label_tumor = 1
    return normaliser(img), mask_gt, label_tumor

def lister_fichiers(dossiers):
    """
    Liste tous les fichiers .mat dans une liste de dossiers.
    Entree  : dossiers (list of str)
    Sortie  : liste de chemins complets vers les .mat
    """
    fichiers = []
    for d in dossiers:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.mat'):
                    fichiers.append(os.path.join(d, f))
    return fichiers

def lire_csv(fichier_csv):
    """
    Lit le fichier CSV contenant les paires (fichier, methode de skull stripping).
    Entree  : fichier_csv (str), chemin vers coupes.csv
    Sortie  : dict {chemin_fichier,  methode (int 1-5)}
    """
    donnees = {}
    if not os.path.isfile(fichier_csv):
        return donnees
    with open(fichier_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['methode'].strip()
            if m in ('1','2','3','4','5'):
                donnees[row['fichier'].strip()] = int(m)
    return donnees

# pretraitement clache

def preprocess(img_n):
    """
    Applique un egalisation adaptative d'histogramme (CLAHE) pour ameliorer
    le contraste local de l'image IRM.
    Entree  : img_n (float32 normalise 0-1)
    Sortie  : image pretraitee (float32 normalise 0-1)
    """
    img_u8 = (img_n * 255).astype(np.uint8)
    clahe  = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    out    = clahe.apply(img_u8)
    return out.astype(np.float32) / 255.0

# délimitation du crâne 5 méthodes :

def _masque_tete(img_lisse):
    """
    Cree un masque binaire de la tete entiere par seuillage Otsu.
    Entree  : img_lisse = image gaussienne lissee
    Sortie  : masque binaire de la tete (bool)
    """
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
    Calcule un rayon d'erosion adaptatif en fonction de la taille du masque.
    Entree  : masque  = masque binaire
              facteur = fraction de la plus petite dimension
    Sortie  : rayon d'erosion (int, minimum 5)
    """
    props = regionprops(label(masque))
    if not props:
        return 5
    prop = max(props, key=lambda p: p.area)
    h = prop.bbox[2] - prop.bbox[0]
    w = prop.bbox[3] - prop.bbox[1]
    return max(5, int(min(h, w) * facteur))

def _composant_du_seed(masque, cy, cx):
    """
    Extrait la composante connexe contenant le point (cy, cx).
    Si le point est hors masque, retourne la plus grande composante.
    Entree  : masque binaire
              cy, cx : coordonnees du point seed
    Sortie  : masque de la composante selectionnee (bool)
    """
    etiq = label(masque)
    lbl  = etiq[cy, cx]
    if lbl != 0:
        return binary_fill_holes(etiq == lbl)
    props = regionprops(etiq)
    if props:
        return binary_fill_holes(etiq == max(props, key=lambda p: p.area).label)
    return masque

def _retirer_petits(masque, seuil=0.05):
    """
    Supprime les petites composantes connexes (moins de 5% de la plus grande).
    Entree  : masque = masque binaire
              seuil  = fraction minimale de la plus grande composante
    Sortie  : masque nettoye (bool)
    """
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


def skull_strip_axiale(img_n):
    """
    Skull stripping pour coupe axiale (vue du dessus).
    Methode : erosion Otsu + seed au barycentre du tissu cerebral.
    Entree  : img_n (float32 normalise)
    Sortie  : (img_strippee, masque_cerveau)
    """
    H, W     = img_n.shape
    img_l    = gaussian_filter(img_n, sigma=2)
    mt       = _masque_tete(img_l)
    rayon    = _rayon_erosion(mt, 0.09)
    mc       = erosion(mt, disk(rayon))
    p10, p85 = np.percentile(img_n, 10), np.percentile(img_n, 85)
    tissu    = (img_n > p10) & (img_n < p85) & mc
    ys, xs   = np.where(tissu)
    cy = int(ys.mean()) if len(ys) > 0 else H // 2
    cx = int(xs.mean()) if len(xs) > 0 else W // 2
    mc = _composant_du_seed(mc, cy, cx)
    mc = closing(mc, disk(5))
    mc = binary_fill_holes(mc)
    mc = _retirer_petits(mc)
    return (img_n * mc).astype(np.float32), mc

def skull_strip_sagittale(img_n):
    """
    Skull stripping pour coupe sagittale (vue de profil).
    Methode : Watershed avec marqueurs interieur/exterieur.
    Garde bien le bas du cerveau (cervelet).
    Entree  : img_n (float32 normalise)
    Sortie  : (img_strippee, masque_cerveau)
    """
    H, W     = img_n.shape
    img_l    = gaussian_filter(img_n, sigma=10)
    mt       = _masque_tete(img_l)
    rayon    = _rayon_erosion(mt, 0.07)
    me       = erosion(mt, disk(rayon))
    ys, xs   = np.where(me)
    cy = int(ys.mean()) if len(ys) > 0 else H // 2
    cx = int(xs.mean()) if len(xs) > 0 else W // 2
    gradient = sobel(img_l)
    p10      = np.percentile(img_n, 10)
    markers  = np.zeros((H, W), dtype=np.int32)
    r        = 30
    markers[max(0,cy-r):min(H,cy+r), max(0,cx-r):min(W,cx+r)] = 1
    markers[img_n < p10 * 0.5] = 2
    markers[~mt] = 2
    lws      = watershed(gradient, markers, mask=mt)
    mws      = closing(binary_fill_holes(lws == 1), disk(5))
    mf       = binary_fill_holes(mws & me)
    mf       = _composant_du_seed(mf, cy, cx)
    mf       = binary_fill_holes(closing(mf, disk(3)))
    mf       = _retirer_petits(mf)
    return (img_n * mf).astype(np.float32), mf

def skull_strip_coronale(img_n):
    """
    Skull stripping pour coupe coronale (vue de face).
    Methode : ellipse initiale + active contour (snake).
    Attention : peut couper les tumeurs basses -> preferer m4 dans ce cas.
    Entree  : img_n (float32 normalise)
    Sortie  : (img_strippee, masque_cerveau)
    """
    H, W         = img_n.shape
    img_l        = gaussian_filter(img_n, sigma=5)
    mt           = _masque_tete(img_l)
    props        = regionprops(label(mt))
    if not props:
        return img_n.astype(np.float32), np.ones((H,W), dtype=bool)
    prop         = max(props, key=lambda p: p.area)
    r0,c0,r1,c1  = prop.bbox
    ht, wt       = r1-r0, c1-c0
    cy_e         = r0 + ht * 0.55
    cx_e         = (c0 + c1) / 2
    t            = np.linspace(0, 2*np.pi, 400)
    ellipse = np.array([cy_e + ht*0.46*np.sin(t),
                    cx_e + wt*0.50*np.cos(t)]).T
    img_s        = gaussian(img_n, sigma=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snake = active_contour(img_s, ellipse, alpha=0.01, beta=20.0,
                       gamma=0.001, max_num_iter=300)
    rr, cc = polygon(snake[:,0], snake[:,1], shape=(H,W))
    ms     = np.zeros((H,W), dtype=bool)
    ms[rr,cc] = True
    ms     = binary_fill_holes(ms) & mt
    p96    = np.percentile(img_n[mt], 96)
    ms     = ms & (img_n < p96)
    rayon  = max(3, int(min(ht,wt)*0.02))
    ms     = erosion(ms, disk(rayon))
    props2 = regionprops(label(ms))
    if props2:
        etiq2 = label(ms)
        ms    = binary_fill_holes(
            etiq2 == max(props2, key=lambda p: p.area).label)
    ms = closing(ms, disk(10))
    ms = binary_fill_holes(ms)
    ms = opening(ms, disk(2))
    return (img_n * ms).astype(np.float32), ms

def skull_strip_isolation(img_n):
    """
    Skull stripping par isolation du cerveau via detection du scalp.
    Methode robuste, recommandee pour les grosses tumeurs ou tumeurs basses.
    Entree  : img_n (float32 normalise)
    Sortie  : (img_strippee, masque_cerveau)
    """
    mn, mx  = img_n.min(), img_n.max()
    img_u8  = ((img_n - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    img_fil = cv2.medianBlur(img_u8, 5)
    seuil   = threshold_otsu(img_fil)
    binaire = (img_fil > seuil).astype(np.uint8)
    rempli  = binary_fill_holes(binaire).astype(np.uint8)
    labs    = label(rempli)
    tailles = np.bincount(labs.ravel())
    tailles[0] = 0
    tete    = (labs == np.argmax(tailles)).astype(np.uint8)
    ep      = max(5, int(0.04 * min(img_n.shape)))
    dist    = distance_transform_edt(tete)
    bande   = ((dist > 0) & (dist <= ep)).astype(np.uint8)
    pix_t   = img_fil[tete > 0]
    s_sc    = np.percentile(pix_t, 70) if len(pix_t) > 0 else 255
    scalp   = ((img_fil >= s_sc) & (bande > 0)).astype(np.uint8)
    rc      = max(4, int(0.012 * min(img_n.shape)))
    elt     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rc+1, 2*rc+1))
    couronne = cv2.dilate(scalp, elt, iterations=1)
    bande2   = (dist <= ep + rc).astype(np.uint8)
    couronne = cv2.bitwise_and(couronne, bande2)
    cerv_b   = cv2.bitwise_and(tete, 1 - couronne)
    re2      = max(2, int(0.006 * min(img_n.shape)))
    elt2     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*re2+1, 2*re2+1))
    erode    = cv2.erode(cerv_b, elt2, iterations=1)
    labs2    = label(erode)
    n2       = labs2.max()
    if n2 == 0:
        cerv_s = cerv_b
    else:
        ys, xs = np.where(tete > 0)
        cy, cx = ys.mean(), xs.mean()
        diag   = np.sqrt(img_n.shape[0]**2 + img_n.shape[1]**2)
        sizes  = np.bincount(labs2.ravel())
        sizes[0] = 0
        scores = np.zeros(len(sizes))
        for li in range(1, len(sizes)):
            if sizes[li] < 200:
                continue
            yl, xl = np.where(labs2 == li)
            d = np.sqrt((yl.mean()-cy)**2 + (xl.mean()-cx)**2)
            scores[li] = sizes[li] * (1 - d/diag)
        cerv_s = (labs2 == int(np.argmax(scores))).astype(np.uint8)
    rd       = re2 + rc + 2
    elt3     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rd+1, 2*rd+1))
    cerv_d   = cv2.dilate(cerv_s, elt3, iterations=1)
    zone     = cv2.bitwise_and(tete, 1 - scalp)
    masque_c = cv2.bitwise_and(cerv_d, zone)
    elt4     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    masque_c = cv2.morphologyEx(masque_c, cv2.MORPH_CLOSE, elt4, iterations=2)
    masque_c = binary_fill_holes(masque_c).astype(np.uint8)
    return (img_n * masque_c.astype(bool)).astype(np.float32), masque_c.astype(bool)

def supprimer_orbites_masque(img_n, masque):
    """
    Supprime les orbites (yeux) du masque cerveau pour les coupes axiales.
    Utilise des criteres de position (haut de l'image), taille et forme.
    Entree  : img_n  = image normalisee
              masque = masque cerveau binaire
    Sortie  : masque nettoye sans les orbites (bool)
    """
    img_u8 = (img_n * 255).astype(np.uint8)
    _, img_thr = cv2.threshold(img_u8, 150, 255, cv2.THRESH_BINARY)
    from skimage.morphology import dilation as sk_dil, erosion as sk_eros
    img_thr_d = sk_dil(img_thr, disk(10))
    labs2     = label(img_thr_d)
    regions2  = regionprops(labs2)
    H, W      = img_n.shape
    masque_orb = np.zeros((H, W), dtype=np.uint8)
    for region in regions2:
        cy_r, cx_r = region.centroid
        minr, minc, maxr, maxc = region.bbox
        ratio = (maxc - minc) / (maxr - minr + 1e-5)
        if (100 < region.area < 10000 and
                cy_r < H * 0.28 and
                0.15 * W < cx_r < 0.85 * W and
                ratio < 4.0):
            masque_orb[labs2 == region.label] = 1
    masque_orb = sk_eros(masque_orb, disk(3))
    masque_propre = masque & ~masque_orb.astype(bool)
    return masque_propre

def skull_strip_snake_orbites(img_n):
    """
    Skull stripping avec active contour (snake) + suppression des orbites.
    Specifiquement concu pour les coupes axiales avec orbites visibles.
    Entree  : img_n (float32 normalise)
    Sortie  : (img_strippee, masque_cerveau)
    """
    h, w    = img_n.shape
    mn, mx  = img_n.min(), img_n.max()
    img_8u  = ((img_n - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_8u, (5,5), 0)
    seuil_o  = threshold_otsu(img_blur)
    mask_bin = (img_blur > seuil_o).astype(np.uint8)
    labs     = label(mask_bin)
    regions  = regionprops(labs)
    if not regions:
        return img_n.astype(np.float32), np.ones((h,w), dtype=bool)
    plus_grande = max(regions, key=lambda r: r.area)
    mask_brain  = (labs == plus_grande.label).astype(np.uint8)
    mask_brain  = binary_fill_holes(mask_brain).astype(np.uint8)
    contours, _ = cv2.findContours(mask_brain, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (cx_e, cy_e), (ax_e, ay_e), _ = cv2.fitEllipse(cnt)
    s      = np.linspace(0, 2*np.pi, 400)
    r_init = cy_e + (ay_e/2*0.92)*np.sin(s)
    c_init = cx_e + (ax_e/2*0.92)*np.cos(s)
    init_c = np.array([r_init, c_init]).T
    img_sm = gaussian(img_8u, sigma=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snake = active_contour(img_sm, init_c, alpha=0.01, beta=5, gamma=0.001)
    mask_int = np.zeros((h,w), dtype=np.uint8)
    rr, cc   = polygon(snake[:,0], snake[:,1], (h,w))
    mask_int[rr,cc] = 1
    from skimage.morphology import dilation as sk_dil, erosion as sk_eros
    masque_crane = ((mask_brain == 1) & (mask_int == 0)).astype(np.uint8)
    masque_crane = sk_dil(masque_crane, disk(15))
    masque_crane = (masque_crane * mask_brain).astype(np.uint8)
    _, img_thr = cv2.threshold(img_8u, 150, 255, cv2.THRESH_BINARY)
    img_thr_d  = sk_dil(img_thr, disk(10))
    labs2      = label(img_thr_d)
    regions2   = regionprops(labs2)
    masque_orb = np.zeros((h,w), dtype=np.uint8)
    for region in regions2:
        cy_r, cx_r = region.centroid
        if (100 < region.area < 10000 and cy_r < h*0.28
                and 0.15*w < cx_r < 0.85*w):
            masque_orb[labs2 == region.label] = 1
    masque_orb = sk_eros(masque_orb, disk(3))
    masque_fin = (mask_int.astype(bool)) & (~masque_orb.astype(bool))
    return (img_n * masque_fin).astype(np.float32), masque_fin

def skull_strip(img_n, methode, pour_train=False):
    """
    Applique la methode de skull stripping choisie.
    En entrainement, remplace la methode 5 par la methode 4 (plus rapide).
    Entree  : img_n  = image normalisee
              methode = int 1 a 5
              pour_train = bool, True pendant l'entrainement
    Sortie  : (img_strippee, masque_cerveau)
    """
    if pour_train and methode == 5:
        methode = 4
    if methode == 1:
        img_s, masque = skull_strip_axiale(img_n)
    elif methode == 2:
        img_s, masque = skull_strip_sagittale(img_n)
    elif methode == 3:
        img_s, masque = skull_strip_coronale(img_n)
    elif methode == 4:
        img_s, masque = skull_strip_isolation(img_n)
    elif methode == 5:
        img_s, masque = skull_strip_snake_orbites(img_n)
        masque = supprimer_orbites_masque(img_n, masque)
        img_s  = (img_n * masque).astype(np.float32)
    
    else:
        raise ValueError(f"Méthode {methode} invalide")
    
    return img_s, masque

# features : 24

def extraire_features(img_n, masque_cerveau, label_tumor=1):
    """
    Extrait 24 features radiomiques par pixel dans le masque cerveau :
      - 6  : intensite multi-echelle (I, moyennes 3/7/15, variances 3/7)
      - 2  : gradient Sobel + Laplacien
      - 4  : filtres Gabor (2 frequences x 2 orientations)
      - 2  : LBP rayon 1 et 2
      - 3  : texture locale (energie, contraste, entropie)
      - 1  : asymetrie gauche/droite
      - 1  : distance au centre du cerveau
      - 2  : K-means tissus (3 clusters)
      - 2  : position normalisee (row, col)
      - 1  : type de tumeur encode
    Entree  : img_n = image normalisee
              masque_cerveau = masque binaire du cerveau
              label_tumor = type de tumeur (1/2/3)
    Sortie  : (X, idx) avec X de forme (N, 24) et idx les indices des pixels
    """
    H, W = img_n.shape
    idx  = np.where(masque_cerveau)

    I   = img_n.astype(np.float32)
    m3  = cv2.blur(I, (3,3))
    m7  = cv2.blur(I, (7,7))
    m15 = cv2.blur(I, (15,15))
    v3  = cv2.blur(I**2, (3,3)) - m3**2
    v7  = cv2.blur(I**2, (7,7)) - m7**2
    f_int = np.stack([I[idx], m3[idx], m7[idx],
                      m15[idx], v3[idx], v7[idx]], axis=1)

    Id  = img_n.astype(np.float64)
    sx  = cv2.Sobel(Id, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(Id, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    lap = cv2.Laplacian(Id, cv2.CV_64F)
    f_grad = np.stack([mag[idx], lap[idx]], axis=1)

    gab_feats = []
    for freq in [0.1, 0.3]:
        for theta in [0, np.pi/4]:
            r, _ = gabor(img_n, frequency=freq, theta=theta)
            gab_feats.append(r[idx])
    f_gabor = np.stack(gab_feats, axis=1)

    img_u8 = (img_n * 255).astype(np.uint8)
    lbp1   = local_binary_pattern(img_u8, 8,  1, method='var')
    lbp2   = local_binary_pattern(img_u8, 16, 2, method='var')
    lbp1   = lbp1 / (lbp1.max() + 1e-8)
    lbp2   = lbp2 / (lbp2.max() + 1e-8)
    f_lbp  = np.stack([lbp1[idx], lbp2[idx]], axis=1)

    In      = I / 255.0
    energy  = cv2.blur(In**2, (5,5))
    shifted = np.roll(In, 1, axis=1)
    contr   = cv2.blur(np.abs(In - shifted), (5,5))
    entr    = -cv2.blur(In * np.log(In + 1e-7), (5,5))
    f_tex   = np.stack([energy[idx], contr[idx], entr[idx]], axis=1)

    asym   = np.abs(I - np.fliplr(I))
    f_sym  = asym[idx].reshape(-1, 1)

    dist   = distance_transform_edt(masque_cerveau)
    dist   = dist / (dist.max() + 1e-6)
    f_dist = dist[idx].reshape(-1, 1)

    pixels   = np.float32(img_n[masque_cerveau].reshape(-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_km, centers = cv2.kmeans(
        pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    order     = np.argsort(centers.flatten())
    label_map = np.zeros(3, dtype=int)
    for new_l, old_l in enumerate(order):
        label_map[old_l] = new_l
    labels_r  = label_map[labels_km.flatten()]
    tmap      = np.zeros(img_n.shape, dtype=np.float32)
    tmap[masque_cerveau] = labels_r / 2.0
    tcand     = (tmap == 1.0).astype(np.float32)
    f_km      = np.stack([tmap[idx], tcand[idx]], axis=1)

    rows, cols = idx
    f_pos = np.stack([rows / H, cols / W], axis=1)

    val_type = (label_tumor - 1) / 2.0
    f_type   = np.full((len(rows), 1), val_type, dtype=np.float32)

    X = np.concatenate([
        f_int, f_grad, f_gabor, f_lbp, f_tex,
        f_sym, f_dist, f_km, f_pos, f_type
    ], axis=1)

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), idx

# post traitement 

def postprocess_mask(pred_mask, label_tumor=1):
    """
    Applique un post-traitement morphologique adapte au type de tumeur :
      - meningiome (1) : fermeture forte + dilatation
      - gliome (2) : fermeture legere, pas de dilatation (contours flous)
      - pituitaire (3) : fermeture + dilatation
    Entree  : pred_mask = masque de prediction binaire
              label_tumor = type de tumeur (1/2/3)
    Sortie  : masque post-traite (numpy uint8)
    """
    params = {
    1: {"ck": 9,  "ci": 1, "dk": 5,  "di": 1},
    2: {"ck": 3,  "ci": 1, "dk": 0,  "di": 0},  # gliome
    3: {"ck": 7,  "ci": 1, "dk": 5,  "di": 1},
    }   
    p      = params.get(label_tumor, params[1])
    binary = (pred_mask > 0).astype(np.uint8)
    kc     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["ck"],)*2)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc, iterations=p["ci"])
    labs   = label(closed)
    if labs.max() == 0:
        return pred_mask
    sizes    = np.bincount(labs.ravel())
    sizes[0] = 0
    biggest  = (labs == np.argmax(sizes)).astype(np.uint8)
    if p["di"] > 0:
        kd      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["dk"],)*2)
        biggest = cv2.dilate(biggest, kd, iterations=p["di"])
    return biggest

# entrainement 

def phase_entrainement(donnees_csv):
    """
    Entraine les modeles Random Forest et SVM sur les images du CSV.
    Pour chaque image : skull stripping, extraction features, echantillonnage,
    equilibre tumeur/sain et entrainement.
    Les modeles sont sauvegardes dans DOSSIER_MODELES.
    Entree  : donnees_csv = dict {chemin a methode}
    """
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    fichiers_train = list(donnees_csv.keys())

    print(f"\n{'═'*60}")
    print(f"  PHASE ENTRAÎNEMENT — {len(fichiers_train)} images | 24 features")
    print(f"  Max {MAX_PIX_TUMEUR} px tumeur/image | méthode 5→4 pendant train")
    print(f"{'═'*60}")

    X_all, y_all = [], []
    n_ok = 0

    for i, chemin in enumerate(fichiers_train):
        nom     = os.path.basename(chemin)
        methode = donnees_csv[chemin]
        print(f"  [{i+1:3d}/{len(fichiers_train)}] {nom} (m{methode})",
              end=' ', flush=True)
        try:
            img_n, mask_gt, label_tumor = charger_mat(chemin)
            if mask_gt is None or mask_gt.sum() == 0:
                print("→ pas de GT")
                continue

            img_p = preprocess(img_n)
            img_s, masque = skull_strip(img_p, methode, pour_train=True)

            # Vérifie que le masque cerveau est suffisant
            if masque.sum() < 500:
                print("→ masque trop petit, skip")
                continue

            features, idx = extraire_features(img_s, masque, label_tumor)
            labels        = mask_gt[idx].astype(int)

            idx_t = np.where(labels == 1)[0]
            idx_s = np.where(labels == 0)[0]
            n_t   = len(idx_t)

            if n_t < 10:
                print("→ tumeur trop petite")
                continue

            n_t_samp = min(n_t, MAX_PIX_TUMEUR)
            n_s_samp = min(len(idx_s), n_t_samp * 3)

            rng   = np.random.default_rng(SEED + i)
            sel_t = rng.choice(idx_t, size=n_t_samp, replace=False)
            sel_s = rng.choice(idx_s, size=n_s_samp, replace=False)
            sel   = np.concatenate([sel_t, sel_s])

            X_all.append(features[sel])
            y_all.append(labels[sel])
            n_ok += 1
            print(f"→ ✓ ({len(sel)} px)")

        except Exception as e:
            print(f"→ erreur : {e}")

    if n_ok == 0:
        print("  ✗ Aucune image valide !")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\n  Dataset : {X.shape[0]} px "
          f"({y.sum()} tumeur / {(y==0).sum()} sain)\n")

    print("  Entraînement Random Forest (100 arbres)...")
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(DOSSIER_MODELES, 'random_forest.joblib'))
    print("  ✓ RF sauvegardé")

    print("  Entraînement SVM...")
    max_svm = 30000
    if len(X) > max_svm:
        idx_sub = np.random.default_rng(SEED).choice(len(X), max_svm, replace=False)
        X_svm, y_svm = X[idx_sub], y[idx_sub]
    else:
        X_svm, y_svm = X, y

    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                    class_weight='balanced', probability=True,
                    random_state=SEED))
    ])
    svm.fit(X_svm, y_svm)
    joblib.dump(svm, os.path.join(DOSSIER_MODELES, 'svm.joblib'))
    print(f"  ✓ SVM sauvegardé → '{DOSSIER_MODELES}/'")

# prédiction

def predire(img_s, masque, label_tumor, modele, seuil=SEUIL_RF):
    """
    Predit le masque de tumeur avec un modele donne.
    Utilise predict_proba avec seuil calibre + post-traitement morphologique.
    Entree  : img_s = image apres skull stripping
              masque = masque cerveau
              label_tumor = type de tumeur
              modele = RF ou SVM entraine
              seuil = seuil de probabilite (defaut SEUIL_RF)
    Sortie  : masque de prediction (bool)
    """
    H, W          = img_s.shape
    features, idx = extraire_features(img_s, masque, label_tumor)

    # predict_proba pour avoir un score de confiance
    try:
        proba  = modele.predict_proba(features)[:, 1]
        y_pred = (proba >= seuil).astype(int)
    except Exception:
        y_pred = modele.predict(features)

    masque_pred = np.zeros((H, W), dtype=bool)
    masque_pred[idx] = y_pred.astype(bool)
    masque_pred = closing(masque_pred, disk(3))
    masque_pred = remove_small_objects(masque_pred, min_size=50)
    masque_pred = binary_fill_holes(masque_pred)
    masque_pred = postprocess_mask(masque_pred, label_tumor)
    return masque_pred.astype(bool)

def calculer_metriques(masque_pred, mask_gt):
    """
    Calcule les metriques de segmentation : Dice, Precision, Rappel.
    Entree  : masque_pred = masque predit (bool)
              mask_gt= masque ground truth (bool)
    Sortie  : dict avec cles 'Dice', 'Precision', 'Rappel'
    """
    inter     = (masque_pred & mask_gt).sum()
    dice      = 2 * inter / (masque_pred.sum() + mask_gt.sum() + 1e-8)
    precision = inter / (masque_pred.sum() + 1e-8)
    rappel    = inter / (mask_gt.sum() + 1e-8)
    return {"Dice": float(dice),
            "Précision": float(precision),
            "Rappel": float(rappel)}

# affichage 

def dessiner_contour(img_rgb, masque, couleur, epaisseur=2):
    """
    Dessine le contour d'un masque binaire sur une image RGB.
    Entree  : img_rgb = image RGB (numpy uint8)
              masque = masque binaire
              couleur = tuple BGR (ex: (255, 50, 50))
              epaisseur = epaisseur du contour en pixels
    Sortie  : image avec contour dessine (numpy uint8)
    """
    if masque is None or masque.sum() == 0:
        return img_rgb
    m      = masque.astype(np.uint8) * 255
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img_rgb.copy()
    cv2.drawContours(img_out, cnts, -1, couleur, epaisseur)
    return img_out

def afficher_resultats(nom, methode, img_n, img_s,
                       masque_rf, masque_svm, mask_gt,
                       met_rf, met_svm):
    """
    Affiche les 5 colonnes de resultats :
      1. Image originale
      2. Image apres skull stripping
      3. Prediction RF  + GT 
      4. Prediction SVM  + GT 
      5. Comparaison RF + SVM + GT
    Entree  : nom, methode, img_n, img_s = infos image
              masque_rf, masque_svm = masques predits
              mask_gt = ground truth
              met_rf, met_svm = dict metriques
    """
    noms_m = {1:"Axiale", 2:"Sagittale", 3:"Coronale",
              4:"Isolation", 5:"Snake+orbites"}

    # Convertit en RGB 8 bits pour OpenCV
    def to_rgb8(img):
        g = (img * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    img_orig_rgb = to_rgb8(img_n)
    img_ss_rgb   = to_rgb8(img_s)

    # Random Forest avec remplissage semi-transparent + contour 
    img_rf = img_ss_rgb.copy().astype(np.float32)
    if masque_rf is not None and masque_rf.sum() > 0:
        img_rf[masque_rf] = img_rf[masque_rf] * 0.5 + np.array([255, 50, 50]) * 0.5
    if mask_gt is not None and mask_gt.sum() > 0:
        img_rf[mask_gt] = img_rf[mask_gt] * 0.5 + np.array([50, 255, 50]) * 0.5
    img_rf = img_rf.clip(0, 255).astype(np.uint8)
    # contours par-dessus
    img_rf = dessiner_contour(img_rf, masque_rf, (255, 50, 50), 2)
    img_rf = dessiner_contour(img_rf, mask_gt,   (50, 255, 50), 2)

    # SVM 
    img_svm = img_ss_rgb.copy().astype(np.float32)
    if masque_svm is not None and masque_svm.sum() > 0:
        img_svm[masque_svm] = img_svm[masque_svm] * 0.5 + np.array([50, 100, 255]) * 0.5
    if mask_gt is not None and mask_gt.sum() > 0:
        img_svm[mask_gt] = img_svm[mask_gt] * 0.5 + np.array([50, 255, 50]) * 0.5
    img_svm = img_svm.clip(0, 255).astype(np.uint8)
    img_svm = dessiner_contour(img_svm, masque_svm, (50, 100, 255), 2)
    img_svm = dessiner_contour(img_svm, mask_gt,    (50, 255, 50),  2)

    # Comparaison RF + SVM + GT 
    img_comp = img_ss_rgb.copy().astype(np.float32)
    if masque_rf is not None and masque_rf.sum() > 0:
        img_comp[masque_rf] = img_comp[masque_rf] * 0.6 + np.array([255, 50, 50]) * 0.4
    if masque_svm is not None and masque_svm.sum() > 0:
        img_comp[masque_svm] = img_comp[masque_svm] * 0.6 + np.array([50, 100, 255]) * 0.4
    if mask_gt is not None and mask_gt.sum() > 0:
        img_comp[mask_gt] = img_comp[mask_gt] * 0.6 + np.array([50, 255, 50]) * 0.4
    img_comp = img_comp.clip(0, 255).astype(np.uint8)
    img_comp = dessiner_contour(img_comp, masque_rf,  (255, 50, 50),  2)
    img_comp = dessiner_contour(img_comp, masque_svm, (50, 100, 255), 2)
    img_comp = dessiner_contour(img_comp, mask_gt,    (50, 255, 50),  2)

    #Plot
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f"Détection tumeur — {noms_m.get(methode,'?')} | {nom}",
        fontsize=13, fontweight="bold", color='white')

    titres = [
        "Image originale",
        "Skull stripping",
        f"Random Forest\nDice={met_rf['Dice']:.3f}  Préc={met_rf['Précision']:.3f}  Rappel={met_rf['Rappel']:.3f}" if met_rf else "Random Forest",
        f"SVM\nDice={met_svm['Dice']:.3f}  Préc={met_svm['Précision']:.3f}  Rappel={met_svm['Rappel']:.3f}" if met_svm else "SVM",
        "Comparaison\nRouge=RF | Bleu=SVM | Vert=GT"
    ]
    images = [img_orig_rgb, img_ss_rgb, img_rf, img_svm, img_comp]

    for ax, img, titre in zip(axes, images, titres):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape)==3 else img,
                  cmap='gray')
        ax.set_title(titre, fontsize=8.5, color='white', pad=4)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Légende
    legend_elems = [
        mpatches.Patch(color=(1.0, 0.2, 0.2), label='Prédiction RF'),
        mpatches.Patch(color=(0.2, 0.4, 1.0), label='Prédiction SVM'),
        mpatches.Patch(color=(0.2, 1.0, 0.2), label='Ground Truth'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               fontsize=9, facecolor='#1a1a2e', labelcolor='white',
               framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def afficher_bilan(nom, met_rf, met_svm):
    """
    Affiche le tableau de metriques RF vs SVM pour une image.
    Avertit si le Dice est inferieur a 0.3 sur un des modeles.
    Entree  : nom = nom du fichier
              met_rf = dict metriques RF
              met_svm = dict metriques SVM
    """

    # Warning si Dice faible
    if met_rf is not None:
        dice_rf  = met_rf['Dice']
        dice_svm = met_svm['Dice']
        if dice_rf < 0.3 and dice_svm < 0.3:
            print("   Dice faible sur les deux modèles — région détectée mais à vérifier visuellement !")
        elif dice_rf < 0.3:
            print("   RF : Dice faible — préférer le résultat SVM pour cette image")
        elif dice_svm < 0.3:
            print("  SVM : Dice faible — préférer le résultat RF pour cette image")
        for k in ('Dice', 'Précision', 'Rappel'):
            rf_v  = f"{met_rf[k]:.4f}"  if met_rf  else "N/A"
            svm_v = f"{met_svm[k]:.4f}" if met_svm else "N/A"
            v_max = max(met_rf[k] if met_rf else 0,
                        met_svm[k] if met_svm else 0)
            emoji = ' ✓' if v_max > 0.5 else ' ⚠'
            print(f"  {k:<12} {rf_v:>15} {svm_v:>15}{emoji}")
        print(f"{'═'*55}\n")

# test

def phase_test(fichiers_test):
    """
    Lance le test interactif sur les images hors CSV.
    Pour chaque image :
      1. Affiche l'image pour identifier le type de coupe
      2. Demande la methode de skull stripping a l'utilisateur
      3. Predit avec RF et SVM
      4. Affiche les metriques et les visualisations
    Affiche un bilan global a la fin.
    Entree  : fichiers_test = liste de chemins vers les .mat de test
    """
    chemin_rf  = os.path.join(DOSSIER_MODELES, 'random_forest.joblib')
    chemin_svm = os.path.join(DOSSIER_MODELES, 'svm.joblib')
    if not os.path.isfile(chemin_rf):
        print("   Modèles non trouvés. Lance d'abord --train")
        return

    rf  = joblib.load(chemin_rf)
    svm = joblib.load(chemin_svm)

    print(f"\n{'═'*60}")
    print(f"  PHASE TEST — {len(fichiers_test)} images")
    print(f"{'═'*60}")

    dices_rf, dices_svm = [], []

    for i, chemin in enumerate(fichiers_test):
        nom = os.path.basename(chemin)

        try:
            img_n, mask_gt, label_tumor = charger_mat(chemin)
        except Exception as e:
            print(f"\n  ✗ Chargement {nom} : {e}")
            continue

        # Affiche l'image originale AVANT de demander la méthode
        fig_prev, ax_prev = plt.subplots(1, 1, figsize=(5, 5))
        fig_prev.suptitle(f"Image {i+1}/{len(fichiers_test)} — {nom}\nFermez pour choisir la méthode",
                        fontsize=11, fontweight='bold', color='white')
        fig_prev.patch.set_facecolor('#1a1a2e')
        ax_prev.imshow(img_n, cmap='gray')
        ax_prev.axis('off')
        plt.tight_layout()
        plt.show(block=True)
        plt.close('all')

        print(f"\n{'─'*55}")
        print(f"  Image {i+1}/{len(fichiers_test)} : {nom}")
        print(f"{'─'*55}")
        print("    1: Axiale  2: Sagittale  3: Coronale")
        print("    4: Isolation cerveau  5: Snake+orbites")
        print("    Axiale    : vue du dessus                           ")
        print("               (m1 standard et m5 si orbites visibles)  ")
        print("    Sagittale : vue de profil                          ")
        print("               (m2 si tumeur normale ou basse)         ")
        print("                (m4 si grosse tumeur )                  ")
        print("   Coronale  : vue de face                             ")
        print("              (m3 si tumeur haute)                    ")
        print("              (m4 si tumeur basse )                   ")
        
        while True:
            try:
                methode = int(input("  Votre choix (1-5) : ").strip())
                if methode in (1,2,3,4,5):
                    break
            except ValueError:
                pass

        print("  → Skull stripping...", end=' ', flush=True)
        try:
            img_p = preprocess(img_n)
            img_s, masque = skull_strip(img_p, methode)
            if masque.sum() < 100:
                print("✗ masque trop petit, essaie méthode 4")
                img_s, masque = skull_strip(img_p, 4)
            print("✓")
        except Exception as e:
            print(f"✗ : {e}")
            continue

        if masque.sum() < 1000:
            print("  ⚠ Masque cerveau trop petit — essaie méthode 4 (Isolation)")

        print("  → Prédiction RF...",  end=' ', flush=True)
        masque_rf = predire(img_s, masque, label_tumor, rf, seuil=SEUIL_RF)
        print("✓")
        print("  → Prédiction SVM...", end=' ', flush=True)
        masque_svm = predire(img_s, masque, label_tumor, svm, seuil=SEUIL_RF)
        print("✓")

        met_rf = met_svm = None
        if mask_gt is not None:
            met_rf  = calculer_metriques(masque_rf,  mask_gt)
            met_svm = calculer_metriques(masque_svm, mask_gt)
            dices_rf.append(met_rf['Dice'])
            dices_svm.append(met_svm['Dice'])
            afficher_bilan(nom, met_rf, met_svm)

        afficher_resultats(nom, methode, img_n, img_s,
                           masque_rf, masque_svm, mask_gt, met_rf, met_svm)

    if dices_rf:
        print(f"\n{'═'*55}")
        print(f"  BILAN FINAL — {len(dices_rf)} images")
        print(f"{'═'*55}")
        print(f"  {'':12} {'Random Forest':>15} {'SVM':>15}")
        print(f"  {'-'*42}")
        print(f"  {'Dice moyen':<12} {np.mean(dices_rf):>15.4f} "
              f"{np.mean(dices_svm):>15.4f}")
        print(f"  {'Dice std':<12} {np.std(dices_rf):>15.4f}  "
              f"{np.std(dices_svm):>15.4f}")
        print(f"  {'Dice max':<12} {np.max(dices_rf):>15.4f}  "
              f"{np.max(dices_svm):>15.4f}")
        winner = ("Random Forest"
                  if np.mean(dices_rf) >= np.mean(dices_svm) else "SVM")
        print(f"\n  Meilleur modèle : {winner} ✓")
        print(f"{'═'*55}\n")


MENU = """
        détection tumeur
"""

def main():
    """
    Point d'entree.
    Arguments :
      --train : lance la phase d'entrainement
      --test  : lance la phase de test interactif
    Si aucun argument : lance les deux.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test',  action='store_true')
    args = parser.parse_args()

    if not args.train and not args.test:
        args.train = args.test = True

    print(MENU)

    donnees_csv = lire_csv(FICHIER_CSV)
    if not donnees_csv:
        print(f"  ✗ CSV non trouvé : {FICHIER_CSV}")
        return
    print(f"  {len(donnees_csv)} images dans coupes.csv")

    fichiers_tous     = lister_fichiers(DOSSIERS_DATASET)
    fichiers_hors_csv = [f for f in fichiers_tous if f not in donnees_csv]
    rng               = np.random.default_rng(SEED)
    nb_test           = min(NB_TEST, len(fichiers_hors_csv))
    fichiers_test     = list(rng.choice(fichiers_hors_csv, nb_test, replace=False))
    print(f"  {len(fichiers_test)} images de test (hors CSV)\n")

    if args.train:
        phase_entrainement(donnees_csv)

    if args.test:
        phase_test(fichiers_test)

if __name__ == "__main__":
    main()