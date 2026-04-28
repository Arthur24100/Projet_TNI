"""
=============================================================
 Sujet 1 - Segmentation des tumeurs cérébrales sur IRM
 02_pretraitement.py — Pipeline de prétraitement (1 image)
=============================================================

Objectif :
    Développer et valider chaque étape du prétraitement sur UNE seule
    image avant d'appliquer le pipeline à tout le dataset.

Pipeline :
    0. Chargement         → image uint16 + masque
    1. Normalisation      → float32 [0, 1]
    2. Skull stripping     → suppression du fond noir + crâne
    3. Filtrage           → réduction du bruit (médian, gaussien, anisotropique)
    4. Égalisation CLAHE  → amélioration du contraste local
    5. Vérification       → affichage avant/après

Conseil du prof :
    → 1 seul pipeline générique pour les 3 classes
    → Tester sur 1 image, puis généraliser à tout le dataset
    → Garder un jeu de test figé, jamais utilisé pendant le dev

Dépendances :
    pip install numpy matplotlib scikit-image scipy
"""

import os, glob, zlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import (binary_closing, binary_opening,
                                 binary_dilation, remove_small_objects,
                                 disk)
from skimage import exposure

# ══════════════════════════════════════════════════════════════
#  LOADER (copié depuis Exploration_dataset.py)
# ══════════════════════════════════════════════════════════════

NOMS_TUMEURS = {1: "Méningiome", 2: "Gliome", 3: "Tumeur pituitaire"}
COULEURS     = {1: "#e74c3c",    2: "#3498db", 3: "#2ecc71"}
_SOUS_DOSSIERS = [
    "brainTumorDataPublic_1-766",
    "brainTumorDataPublic_767-1532",
    "brainTumorDataPublic_1533-2298",
    "brainTumorDataPublic_2299-3064",
]

def _decompress(raw, offset):
    for ml in [5_000, 10_000, 20_000, 40_000, 80_000]:
        try: return zlib.decompress(raw[offset:offset + ml])
        except: pass
    return None

def _find_chunks(raw):
    chunks, i = [], 0
    while i < len(raw) - 2:
        if raw[i] == 0x78 and raw[i+1] in (0x01, 0x5E, 0x9C, 0xDA):
            d = _decompress(raw, i)
            if d and len(d) == 65_536:
                chunks.append(d); i += 100; continue
        i += 1
    return chunks

def _find_label(raw):
    P = {1: b'\x00\x00\x00\x00\x00\x00\xf0\x3f',
         2: b'\x00\x00\x00\x00\x00\x00\x00\x40',
         3: b'\x00\x00\x00\x00\x00\x00\x08\x40'}
    for val, pat in P.items():
        pos = raw.find(pat)
        if 0 <= pos < 6000: return val
    return None

def lire_mat(chemin):
    with open(chemin, "rb") as f:
        f.seek(512); raw = f.read()
    if raw[:8] != b'\x89HDF\r\n\x1a\n':
        raise ValueError(f"Fichier invalide : {chemin}")
    chunks = _find_chunks(raw)
    if len(chunks) < 12:
        raise ValueError(f"Chunks insuffisants ({len(chunks)}) : {chemin}")
    image = np.hstack([np.frombuffer(c, dtype="<u2").reshape(512, 64)  for c in chunks[:8]])
    mask  = np.hstack([np.frombuffer(c, dtype=np.uint8).reshape(512, 128) for c in chunks[8:12]])
    mask  = (mask > 0).astype(np.uint8)
    return {"image": image, "mask": mask, "label": _find_label(raw),
            "fichier": os.path.basename(chemin)}

def charger_dataset(dossier, max_par_classe=None, verbose=True):
    tous = []
    for sd in _SOUS_DOSSIERS:
        mats = sorted(
            glob.glob(os.path.join(dossier, sd, "*.mat")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        tous.extend(mats)
    if not tous and os.path.isdir(os.path.join(dossier, "1512427")):
        return charger_dataset(os.path.join(dossier, "1512427"),
                               max_par_classe, verbose)
    if verbose: print(f"[INFO] {len(tous)} fichiers .mat trouvés.")
    donnees, cpt = [], {1:0, 2:0, 3:0}
    for chemin in tous:
        try:
            d = lire_mat(chemin)
            lbl = d["label"] or 0
            if max_par_classe and cpt.get(lbl,0) >= max_par_classe: continue
            cpt[lbl] = cpt.get(lbl,0) + 1
            donnees.append(d)
        except: pass
    if verbose:
        from collections import Counter
        c = Counter(d["label"] for d in donnees)
        print(f"[OK] {len(donnees)} images  ({sum(1 for d in donnees if d['label'] is None)} sans label)")
        for l,n in NOMS_TUMEURS.items():
            print(f"  {n:25s} : {c.get(l,0):4d}")
    return donnees


# ══════════════════════════════════════════════════════════════
#  PIPELINE DE PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════

def etape1_normaliser(image_uint16):
    """
    Étape 1 — Normalisation min-max vers [0, 1] en float32.
    Nécessaire pour homogénéiser les contrastes entre patients.
    """
    img = image_uint16.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def etape2_skull_stripping(img_norm):
    """
    Étape 2 — Skull stripping : isoler le cerveau, supprimer le fond.

    Méthode :
        1. Seuillage d'Otsu → séparer fond/tissu
        2. Fermeture morphologique → combler les trous
        3. Sélection du plus grand composant connexe → garder le cerveau
        4. Dilatation légère → récupérer les bords
        → Retourne le masque binaire du cerveau

    Pourquoi c'est général :
        Otsu s'adapte automatiquement à l'histogramme de chaque image,
        donc le seuil est recalculé pour chaque patient.
    """
    # Seuillage automatique (Otsu)
    seuil = threshold_otsu(img_norm)
    binaire = img_norm > seuil

    # Fermeture + ouverture pour nettoyer
    binaire = binary_closing(binaire, disk(8))
    binaire = binary_opening(binaire, disk(4))

    # Garder uniquement le plus grand objet (le cerveau)
    from skimage.measure import label as cc_label
    labelled = cc_label(binaire)
    if labelled.max() == 0:
        return binaire.astype(np.uint8)

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # ignorer le fond
    plus_grand = sizes.argmax()
    masque_cerveau = (labelled == plus_grand)

    # Dilatation légère pour récupérer les bords
    masque_cerveau = binary_dilation(masque_cerveau, disk(5))

    return masque_cerveau.astype(np.uint8)


def etape3_filtrage(img_norm, methode="median", force=3):
    """
    Étape 3 — Filtrage du bruit.

    Paramètres :
        methode : "median"     → préserve les bords (recommandé pour IRM)
                  "gaussien"   → lissage homogène
                  "aniso"      → diffusion anisotropique (préserve mieux les contours)
        force   : taille du noyau (médian/gaussien) ou nb itérations (aniso)

    Pourquoi le médian est recommandé :
        Le bruit dans les IRM est souvent du bruit de Rician (similaire au sel-poivre).
        Le filtre médian supprime ce bruit sans flouter les bords de la tumeur.
    """
    if methode == "median":
        return median_filter(img_norm, size=force).astype(np.float32)

    elif methode == "gaussien":
        return gaussian_filter(img_norm, sigma=force).astype(np.float32)

    elif methode == "aniso":
        # Diffusion anisotropique simple (Perona-Malik)
        # Préserve mieux les contours que le gaussien
        img = img_norm.copy()
        kappa = 0.1     # sensibilité aux bords (plus faible = plus de préservation)
        dt    = 0.1     # pas de temps
        for _ in range(force):
            # Gradients dans les 4 directions
            nord  = np.roll(img,  1, axis=0) - img
            sud   = np.roll(img, -1, axis=0) - img
            est   = np.roll(img, -1, axis=1) - img
            ouest = np.roll(img,  1, axis=1) - img
            # Coefficients de diffusion (décroît aux bords forts)
            cn = np.exp(-(nord  / kappa) ** 2)
            cs = np.exp(-(sud   / kappa) ** 2)
            ce = np.exp(-(est   / kappa) ** 2)
            co = np.exp(-(ouest / kappa) ** 2)
            img = img + dt * (cn*nord + cs*sud + ce*est + co*ouest)
        return img.astype(np.float32)

    else:
        raise ValueError(f"Méthode inconnue : {methode}")


def etape4_egalisation(img_norm, methode="clahe"):
    """
    Étape 4 — Égalisation du contraste.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) :
        - Contrairement à l'égalisation globale, CLAHE opère sur des tuiles locales
        - Permet de voir à la fois les détails dans les zones sombres ET claires
        - Le paramètre clip_limit empêche la sur-amplification du bruit

    Pourquoi c'est utile pour les tumeurs :
        Les tumeurs peuvent avoir des intensités similaires aux tissus sains.
        CLAHE améliore la séparation visuelle → aide la segmentation.
    """
    if methode == "clahe":
        return exposure.equalize_adapthist(
            img_norm, clip_limit=0.03
        ).astype(np.float32)

    elif methode == "globale":
        return exposure.equalize_hist(img_norm).astype(np.float32)

    elif methode == "aucune":
        return img_norm

    else:
        raise ValueError(f"Méthode inconnue : {methode}")


def pretraiter(image_uint16,
               methode_filtre="median",
               force_filtre=3,
               methode_egalisation="clahe",
               appliquer_skull_strip=True):
    """
    Pipeline complet de prétraitement — FONCTION GÉNÉRIQUE.

    Entrée  : image uint16 brute (512×512)
    Sortie  : dict avec toutes les étapes intermédiaires + image finale

    C'est cette fonction qui sera appliquée à tout le dataset.
    Elle ne dépend d'aucun label, donc elle est générale (1 seul algo pour les 3 classes).
    """
    etapes = {}

    # ── Étape 1 : Normalisation ──────────────────────────────
    img_norm = etape1_normaliser(image_uint16)
    etapes["1_normalise"] = img_norm

    # ── Étape 2 : Skull stripping ────────────────────────────
    if appliquer_skull_strip:
        masque_cerveau = etape2_skull_stripping(img_norm)
        img_stripped = img_norm * masque_cerveau
        etapes["2_skull_strip"] = img_stripped
        etapes["2_masque_cerveau"] = masque_cerveau
    else:
        img_stripped = img_norm
        etapes["2_skull_strip"] = img_stripped
        etapes["2_masque_cerveau"] = np.ones_like(img_norm, dtype=np.uint8)

    # ── Étape 3 : Filtrage ───────────────────────────────────
    img_filtre = etape3_filtrage(img_stripped, methode_filtre, force_filtre)
    etapes["3_filtre"] = img_filtre

    # ── Étape 4 : Égalisation ────────────────────────────────
    img_finale = etape4_egalisation(img_filtre, methode_egalisation)
    etapes["4_final"] = img_finale

    return etapes


# ══════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════

def afficher_pipeline(d, etapes):
    """Affiche toutes les étapes du prétraitement côte à côte."""
    img_orig = d["image"]
    mask_gt  = d["mask"]
    nom = NOMS_TUMEURS.get(d["label"], "?")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Pipeline de prétraitement — {nom} | {d['fichier']}",
        fontsize=14, fontweight="bold"
    )

    def show(ax, img, titre, cmap="gray", overlay_mask=None, couleur="#ff0000"):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=img.max() if img.max() > 1 else 1)
        if overlay_mask is not None and overlay_mask.sum() > 0:
            r = int(couleur[1:3], 16) / 255
            g = int(couleur[3:5], 16) / 255
            b = int(couleur[5:7], 16) / 255
            img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)
            rgb = np.stack([img_n]*3, axis=-1).copy()
            rgb[overlay_mask==1, 0] = 0.5*img_n[overlay_mask==1] + 0.5*r
            rgb[overlay_mask==1, 1] = 0.5*img_n[overlay_mask==1] * g
            rgb[overlay_mask==1, 2] = 0.5*img_n[overlay_mask==1] * b
            ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(titre, fontsize=9)
        ax.axis("off")

    col = COULEURS.get(d["label"], "#ff0000")

    # Ligne 1
    show(axes[0,0], img_orig,              "0 — Image brute (uint16)",        cmap="gray")
    show(axes[0,1], etapes["1_normalise"], "1 — Normalisée [0,1]",            cmap="gray")
    show(axes[0,2], etapes["2_masque_cerveau"], "2a — Masque cerveau (Otsu+morph)", cmap="hot")
    show(axes[0,3], etapes["2_skull_strip"],    "2b — Skull stripped",             cmap="gray")

    # Ligne 2
    show(axes[1,0], etapes["3_filtre"],    "3 — Après filtrage (médian)",     cmap="gray")
    show(axes[1,1], etapes["4_final"],     "4 — Après CLAHE",                 cmap="gray")
    show(axes[1,2], etapes["4_final"],     "4 — Avec masque tumeur (GT)",
         overlay_mask=mask_gt, couleur=col)
    # Comparaison histogrammes
    axes[1,3].hist(etapes["1_normalise"].flatten(),  bins=100, alpha=0.6,
                   color="steelblue", label="Avant")
    axes[1,3].hist(etapes["4_final"].flatten(),      bins=100, alpha=0.6,
                   color="darkorange", label="Après CLAHE")
    axes[1,3].set_title("Histogramme avant/après", fontsize=9)
    axes[1,3].set_xlabel("Intensité"); axes[1,3].set_ylabel("Nb pixels")
    axes[1,3].legend(fontsize=8); axes[1,3].grid(alpha=0.3)

    plt.tight_layout()
    nom_fig = f"pretraitement_{nom.replace(' ', '_')}.png"
    plt.savefig(nom_fig, dpi=130, bbox_inches="tight")
    print(f"[OK] Sauvegardé : {nom_fig}")
    plt.show()


def comparer_filtres(d):
    """Compare les 3 méthodes de filtrage sur la même image."""
    img_norm = etape1_normaliser(d["image"])
    img_strip = img_norm * etape2_skull_stripping(img_norm)

    methodes = [
        ("median 3",       etape3_filtrage(img_strip, "median",   3)),
        ("median 5",       etape3_filtrage(img_strip, "median",   5)),
        ("gaussien σ=1",   etape3_filtrage(img_strip, "gaussien", 1)),
        ("gaussien σ=2",   etape3_filtrage(img_strip, "gaussien", 2)),
        ("anisotropique 5",etape3_filtrage(img_strip, "aniso",    5)),
        ("anisotropique 10",etape3_filtrage(img_strip,"aniso",   10)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Comparaison des filtres — {NOMS_TUMEURS.get(d['label'],'?')} | {d['fichier']}",
        fontsize=13, fontweight="bold"
    )

    for ax, (nom_filtre, img_f) in zip(axes.flat, methodes):
        # Zoomer sur la zone tumorale
        rows, cols = np.where(d["mask"] == 1)
        if len(rows) > 0:
            r0, r1 = max(0, rows.min()-40), min(512, rows.max()+40)
            c0, c1 = max(0, cols.min()-40), min(512, cols.max()+40)
            crop_f = img_f[r0:r1, c0:c1]
            crop_m = d["mask"][r0:r1, c0:c1]
        else:
            crop_f, crop_m = img_f, d["mask"]

        ax.imshow(crop_f, cmap="gray")
        # Contour du masque
        if crop_m.sum() > 0:
            from skimage.segmentation import find_boundaries
            bord = find_boundaries(crop_m, mode="outer")
            ax.imshow(np.ma.masked_where(~bord, bord),
                      cmap="Reds", alpha=0.8, vmin=0, vmax=1)
        ax.set_title(f"Filtre : {nom_filtre}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("comparaison_filtres.png", dpi=130, bbox_inches="tight")
    print("[OK] Sauvegardé : comparaison_filtres.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DOSSIER_DATASET = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427"   # ← modifier si besoin


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  PRÉTRAITEMENT IRM — TEST SUR 1 IMAGE PAR CLASSE")
    print("=" * 55)

    # ── 1. Charger seulement 1 image par classe pour tester ──
    donnees = charger_dataset(DOSSIER_DATASET, max_par_classe=1)

    for d in donnees:
        lbl = d["label"]
        print(f"\n{'─'*50}")
        print(f"Image : {d['fichier']} | {NOMS_TUMEURS.get(lbl,'?')}")
        print(f"  Taille image    : {d['image'].shape}")
        print(f"  Intensité brute : min={d['image'].min()}, max={d['image'].max()}")
        print(f"  Pixels tumeur   : {d['mask'].sum()}")

        # ── 2. Appliquer le pipeline ──────────────────────────
        etapes = pretraiter(
            d["image"],
            methode_filtre="median",
            force_filtre=3,
            methode_egalisation="clahe",
            appliquer_skull_strip=True
        )

        print(f"  [Pipeline OK] Image finale : {etapes['4_final'].shape}, "
              f"min={etapes['4_final'].min():.3f}, max={etapes['4_final'].max():.3f}")

        # ── 3. Visualiser le pipeline complet ────────────────
        afficher_pipeline(d, etapes)

    # ── 4. Comparer les filtres (sur la 1ère image) ───────────
    print("\n[INFO] Comparaison des méthodes de filtrage...")
    comparer_filtres(donnees[0])

    print("\n" + "=" * 55)
    print("  PROCHAINE ÉTAPE : généraliser à tout le dataset")
    print("  → charger_dataset(max_par_classe=None)")
    print("  → for d in donnees: d['image_traitee'] = pretraiter(d['image'])['4_final']")
    print("=" * 55)
