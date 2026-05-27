"""
============================================================
    ISOLATION DU CERVEAU (SKULL STRIPPING) — ROBUSTE AUX VUES
============================================================

Objectif :
    Enlever TOUT ce qui entoure le cerveau (crâne, os, yeux, scalp, graisse)
    sur des IRM cérébrales, quelle que soit la vue (axiale, coronale, sagittale).

Stratégie (uniquement traitement d'image classique — cours/TP) :
    1. Filtrage médian pour réduire le bruit
    2. Seuillage d'Otsu → sépare le fond noir du tissu
    3. Remplissage des trous internes
    4. Plus grand composant connexe → toute la tête (mais pas les yeux isolés)
    5. ÉROSION FORTE avec disque → fait disparaître le scalp/crâne fin et
       sépare les yeux (qui sont plus petits) du cerveau (plus gros)
    6. Plus grand composant connexe après érosion → cerveau uniquement
    7. Dilatation pour récupérer les bords érodés
    8. Intersection avec le masque "tête" pour ne pas déborder
    9. Remplissage final des trous

Pourquoi ça marche pour toutes les vues ?
    Quelle que soit la coupe, le cerveau est TOUJOURS le plus gros objet
    convexe au centre. Les yeux, le nez, le scalp sont soit plus petits,
    soit reliés au cerveau seulement par une fine couche (le crâne) que
    l'érosion forte casse.
"""

import os
import glob
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu


# ============================================================
# 1. CHARGEMENT DES IMAGES .MAT
# ============================================================

def load_mat_image(path):
    """Charge une image et son masque tumeur depuis un .mat (format figshare)."""
    try:
        with h5py.File(path, 'r') as f:
            image = np.array(f['cjdata']['image']).T
            mask  = np.array(f['cjdata']['tumorMask']).T
            label_tumor = int(np.array(f['cjdata']['label']).flat[0])
        return image, mask, label_tumor
    except Exception as e:
        print(f"Erreur lecture {path} : {e}")
        return None, None, None


def normaliser_uint8(image):
    """Normalise une image en uint8 (0-255) — nécessaire pour OpenCV."""
    img = image.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# ============================================================
# 2. ISOLATION DU CERVEAU — PIPELINE COMPLET
# ============================================================

def isoler_cerveau(image, debug=False):
    """
    Isole le cerveau de tout ce qui l'entoure (crâne, yeux, scalp, os, visage).
    Robuste aux trois vues (axiale, coronale, sagittale).

    Idée clé : en IRM T1, le SCALP (graisse) est très brillant → on le détecte
    par seuillage haut, puis on le DILATE pour englober le crâne adjacent,
    et on SOUSTRAIT cette couronne du masque tête. Ce qui reste est le cerveau.

    Paramètres
    ----------
    image  : np.ndarray (H, W), image IRM brute ou normalisée
    debug  : bool, si True, retourne aussi les masques intermédiaires

    Retour
    ------
    cerveau     : image IRM avec uniquement le cerveau visible (reste = 0)
    masque_cerv : masque binaire du cerveau (uint8, 0 ou 1)
    etapes      : dict des étapes intermédiaires (si debug=True)
    """

    # --- Préparation : normalisation uint8 si besoin
    if image.dtype != np.uint8:
        img = normaliser_uint8(image)
    else:
        img = image.copy()

    etapes = {"0_originale": img.copy()}

    # --------------------------------------------------------
    # ÉTAPE 1 — Filtrage médian (réduction du bruit poivre & sel)
    # --------------------------------------------------------
    img_filtre = cv2.medianBlur(img, 5)
    etapes["1_filtree"] = img_filtre

    # --------------------------------------------------------
    # ÉTAPE 2 — Seuillage d'Otsu → silhouette de la tête
    # --------------------------------------------------------
    seuil = threshold_otsu(img_filtre)
    binaire = (img_filtre > seuil).astype(np.uint8)
    etapes["2_otsu"] = binaire * 255

    # --------------------------------------------------------
    # ÉTAPE 3 — Remplissage + plus grand composant = tête entière
    # --------------------------------------------------------
    rempli = binary_fill_holes(binaire).astype(np.uint8)
    labels_cc, n_comp = label(rempli)
    if n_comp == 0:
        masque_vide = np.zeros_like(img, dtype=np.uint8)
        if debug:
            return img, masque_vide, etapes
        return img, masque_vide

    tailles = np.bincount(labels_cc.ravel())
    tailles[0] = 0
    tete = (labels_cc == np.argmax(tailles)).astype(np.uint8)
    etapes["3_tete"] = tete * 255

    # --------------------------------------------------------
    # ÉTAPE 4 — Détection du SCALP brillant LIMITÉE à la bande externe
    # Idée : le scalp est brillant ET tout au bord de la tête.
    # Si on détectait globalement, on confondrait avec les vaisseaux/méninges
    # internes (et parfois la tumeur elle-même !).
    # → On crée une "bande externe" (transformée de distance < seuil),
    #   et on ne cherche le scalp QUE dans cette bande.
    # --------------------------------------------------------
    # Bande externe = pixels de la tête à moins de X pixels du bord
    epaisseur_bande = max(5, int(0.04 * min(img.shape)))  # ~4 % de la taille
    dist_au_bord = distance_transform_edt(tete)
    bande_externe = ((dist_au_bord > 0) & (dist_au_bord <= epaisseur_bande)).astype(np.uint8)

    # Scalp = pixels brillants DANS la bande externe uniquement
    pixels_tete = img_filtre[tete > 0]
    if len(pixels_tete) == 0:
        seuil_scalp = 255
    else:
        # Percentile 70 sur la tête entière (seuil de "brillant")
        seuil_scalp = np.percentile(pixels_tete, 70)
    scalp = ((img_filtre >= seuil_scalp) & (bande_externe > 0)).astype(np.uint8)
    etapes["4_scalp_brillant"] = scalp * 255

    # --------------------------------------------------------
    # ÉTAPE 5 — Couronne : scalp + une petite marge intérieure
    # On dilate le scalp UNIQUEMENT vers l'intérieur pour englober
    # le crâne adjacent. La couronne reste fine.
    # --------------------------------------------------------
    rayon_couronne = max(4, int(0.012 * min(img.shape)))
    elt_couronne = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rayon_couronne + 1, 2 * rayon_couronne + 1)
    )
    couronne = cv2.dilate(scalp, elt_couronne, iterations=1)
    # Limiter la couronne à la bande externe élargie (ne pas envahir le centre)
    bande_elargie = (dist_au_bord <= epaisseur_bande + rayon_couronne).astype(np.uint8)
    couronne = cv2.bitwise_and(couronne, bande_elargie)
    etapes["5_couronne"] = couronne * 255

    # --------------------------------------------------------
    # ÉTAPE 6 — Soustraction tête − couronne
    # --------------------------------------------------------
    cerveau_brut = cv2.bitwise_and(tete, 1 - couronne)
    etapes["6_apres_soustraction"] = cerveau_brut * 255

    # --------------------------------------------------------
    # ÉTAPE 7 — Plus grand CC le plus CENTRAL
    # Pour séparer mâchoire/nez (vues sagittales, coronales basses).
    # --------------------------------------------------------
    # Petite érosion pour casser des ponts éventuels
    rayon_erosion = max(2, int(0.006 * min(img.shape)))
    elt_erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rayon_erosion + 1, 2 * rayon_erosion + 1)
    )
    erode = cv2.erode(cerveau_brut, elt_erosion, iterations=1)

    labels_cc2, n_comp2 = label(erode)
    if n_comp2 == 0:
        cerveau_seul = cerveau_brut
    else:
        ys, xs = np.where(tete > 0)
        cy, cx = ys.mean(), xs.mean()
        sizes = np.bincount(labels_cc2.ravel())
        sizes[0] = 0
        diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
        scores = np.zeros(len(sizes))
        for lbl in range(1, len(sizes)):
            if sizes[lbl] < 200:
                continue
            yl, xl = np.where(labels_cc2 == lbl)
            d = np.sqrt((yl.mean() - cy) ** 2 + (xl.mean() - cx) ** 2)
            # taille pondérée par centralité (plus c'est près du centre, mieux)
            scores[lbl] = sizes[lbl] * (1 - d / diag)
        meilleur = int(np.argmax(scores))
        cerveau_seul = (labels_cc2 == meilleur).astype(np.uint8)
    etapes["7_cerveau_central"] = cerveau_seul * 255

    # --------------------------------------------------------
    # ÉTAPE 8 — Dilatation GÉNÉREUSE pour récupérer les bords du cerveau
    # + intersection avec (tête − scalp_pur).
    # On dilate plus que l'érosion + la couronne pour rattraper les méningiomes
    # en bordure.
    # --------------------------------------------------------
    rayon_dilatation = rayon_erosion + rayon_couronne + 2
    elt_dilatation = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rayon_dilatation + 1, 2 * rayon_dilatation + 1)
    )
    cerveau_dilate = cv2.dilate(cerveau_seul, elt_dilatation, iterations=1)
    # Zone autorisée = tête entière SAUF le scalp brillant pur (pas la couronne)
    zone_autorisee = cv2.bitwise_and(tete, 1 - scalp)
    masque_cerv = cv2.bitwise_and(cerveau_dilate, zone_autorisee)

    # --------------------------------------------------------
    # ÉTAPE 9 — Fermeture finale + remplissage des trous
    # --------------------------------------------------------
    elt_fermeture = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    masque_cerv = cv2.morphologyEx(
        masque_cerv, cv2.MORPH_CLOSE, elt_fermeture, iterations=2
    )
    masque_cerv = binary_fill_holes(masque_cerv).astype(np.uint8)
    etapes["8_final"] = masque_cerv * 255

    # --- Application du masque sur l'image originale ---
    cerveau = cv2.bitwise_and(img, img, mask=masque_cerv)

    if debug:
        return cerveau, masque_cerv, etapes
    return cerveau, masque_cerv


# ============================================================
# 3. VISUALISATION DES ÉTAPES (pour debug et rapport)
# ============================================================

def visualiser_etapes(image, titre="", save_path=None):
    """Affiche toutes les étapes du pipeline pour une image."""
    cerveau, masque, etapes = isoler_cerveau(image, debug=True)

    cles = [
        ("0_originale",          "0. Image originale"),
        ("1_filtree",            "1. Filtrage médian"),
        ("2_otsu",               "2. Seuillage Otsu"),
        ("3_tete",               "3. Plus grand CC\n(tête entière)"),
        ("4_scalp_brillant",     "4. Scalp brillant\n(top 20 % d'intensité)"),
        ("5_couronne",           "5. Couronne\n(scalp + crâne dilatés)"),
        ("6_apres_soustraction", "6. Tête − couronne\n≈ cerveau brut"),
        ("7_cerveau_central",    "7. Plus gros CC central\n(retire nez/mâchoire)"),
        ("8_final",              "8. Masque cerveau final"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(f"Pipeline d'isolation du cerveau — {titre}",
                 fontsize=14, fontweight='bold')

    for ax, (cle, titre_e) in zip(axes.ravel(), cles):
        ax.imshow(etapes[cle], cmap='gray')
        ax.set_title(titre_e, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  → Étapes sauvegardées : {save_path}")
    plt.show()

    # --- Figure résumé : avant / après ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle(f"Avant / Après — {titre}", fontsize=13, fontweight='bold')

    axes2[0].imshow(etapes["0_originale"], cmap='gray')
    axes2[0].set_title("Image originale")
    axes2[0].axis('off')

    axes2[1].imshow(masque, cmap='gray')
    axes2[1].set_title("Masque cerveau")
    axes2[1].axis('off')

    axes2[2].imshow(cerveau, cmap='gray')
    axes2[2].set_title("Cerveau isolé")
    axes2[2].axis('off')

    plt.tight_layout()
    if save_path:
        sp2 = save_path.replace(".png", "_resume.png")
        plt.savefig(sp2, dpi=130, bbox_inches='tight')
        print(f"  → Résumé sauvegardé : {sp2}")
    plt.show()


# ============================================================
# 4. TEST SUR PLUSIEURS IMAGES
# ============================================================

def tester_sur_lot(fichiers, n_images=6, save_dir="resultats_isolation"):
    """Teste l'isolation sur plusieurs images et compare."""
    os.makedirs(save_dir, exist_ok=True)

    # Sélection aléatoire de n_images
    fichiers_test = np.random.choice(fichiers, size=min(n_images, len(fichiers)),
                                     replace=False)

    n = len(fichiers_test)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Comparaison : original / masque / cerveau isolé",
                 fontsize=14, fontweight='bold')

    for i, f in enumerate(fichiers_test):
        img, mask_tumeur, label_t = load_mat_image(f)
        if img is None:
            continue

        img_norm = normaliser_uint8(img)
        cerveau, masque = isoler_cerveau(img_norm)

        nom = os.path.basename(f)
        noms_t = {1: "Méningiome", 2: "Gliome", 3: "Pituitaire"}

        axes[i, 0].imshow(img_norm, cmap='gray')
        axes[i, 0].set_title(f"{nom} — {noms_t.get(label_t, '?')}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masque, cmap='gray')
        axes[i, 1].set_title("Masque cerveau")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cerveau, cmap='gray')
        # Superposer la vraie tumeur en rouge pour vérifier qu'on ne l'a pas perdue
        if mask_tumeur is not None and mask_tumeur.max() > 0:
            axes[i, 2].contour(mask_tumeur, colors='red', linewidths=1.5)
            axes[i, 2].set_title("Cerveau isolé\n(contour tumeur en rouge)")
        else:
            axes[i, 2].set_title("Cerveau isolé")
        axes[i, 2].axis('off')

    plt.tight_layout()
    out = os.path.join(save_dir, "comparaison_lot.png")
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\n→ Sauvegardé : {out}")
    plt.show()


# ============================================================
# 5. VÉRIFICATION : la tumeur est-elle toujours dans le masque ?
# ============================================================

def verifier_preservation_tumeur(fichiers):
    """
    Vérifie que le masque cerveau contient bien la tumeur.
    Si on perd la tumeur, c'est que l'isolation est trop agressive.
    """
    print("\n--- VÉRIFICATION : préservation de la tumeur ---")
    pertes = []
    for f in fichiers:
        img, mask_tumeur, label_t = load_mat_image(f)
        if img is None or mask_tumeur is None or mask_tumeur.max() == 0:
            continue

        img_norm = normaliser_uint8(img)
        _, masque = isoler_cerveau(img_norm)

        # Pourcentage de la tumeur conservé dans le masque cerveau
        inter = np.sum((mask_tumeur > 0) & (masque > 0))
        total = np.sum(mask_tumeur > 0)
        pct = 100.0 * inter / max(total, 1)
        pertes.append(pct)

    pertes = np.array(pertes)
    print(f"  Nb images testées      : {len(pertes)}")
    print(f"  Tumeur préservée moy.  : {pertes.mean():.2f} %")
    print(f"  Tumeur préservée min.  : {pertes.min():.2f} %")
    print(f"  Images avec ≥ 95 %     : {(pertes >= 95).sum()} / {len(pertes)}")
    print(f"  Images avec ≥ 99 %     : {(pertes >= 99).sum()} / {len(pertes)}")
    return pertes


# ============================================================
# 6. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --- Recherche des fichiers .mat dans tous les dossiers ---
    dossiers = [
        './brainTumorDataPublic_1-766/*.mat',
        './brainTumorDataPublic_767-1532/*.mat',
        './brainTumorDataPublic_1533-2298/*.mat',
        './brainTumorDataPublic_2299-3064/*.mat',
    ]
    all_files = []
    for pattern in dossiers:
        all_files += sorted(glob.glob(pattern))

    print(f"Total fichiers trouvés : {len(all_files)}")

    if len(all_files) == 0:
        print("Aucun fichier trouvé. Vérifie le chemin des dossiers.")
        exit()

    np.random.seed(42)

    # ----------------------------------------------------------
    # TEST 1 : visualiser toutes les étapes sur 3 images de vues
    # potentiellement différentes (pris à différents endroits du
    # dataset pour augmenter la diversité)
    # ----------------------------------------------------------
    indices_demo = [0, len(all_files) // 3, 2 * len(all_files) // 3]

    for idx in indices_demo:
        f = all_files[idx]
        img, _, label_t = load_mat_image(f)
        if img is None:
            continue
        noms_t = {1: "Méningiome", 2: "Gliome", 3: "Pituitaire"}
        titre = f"{os.path.basename(f)} ({noms_t.get(label_t, '?')})"
        save_path = f"etapes_{os.path.basename(f).replace('.mat', '.png')}"
        visualiser_etapes(normaliser_uint8(img), titre=titre, save_path=save_path)

    # ----------------------------------------------------------
    # TEST 2 : comparaison sur un lot de 6 images aléatoires
    # ----------------------------------------------------------
    tester_sur_lot(all_files, n_images=6)

    # ----------------------------------------------------------
    # TEST 3 : vérification sur 50 images que la tumeur n'est pas perdue
    # ----------------------------------------------------------
    echantillon = np.random.choice(all_files, size=min(50, len(all_files)),
                                   replace=False)
    verifier_preservation_tumeur(echantillon)
