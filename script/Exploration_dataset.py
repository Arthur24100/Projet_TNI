"""
=============================================================
 Sujet 1 - Segmentation des tumeurs cérébrales sur IRM
 Exploration_dataset.py  — Script principal d'exploration
=============================================================

Structure attendue (chemin relatif depuis ce script) :
    Dataset/
    └── 1512427/
        ├── brainTumorDataPublic_1-766/
        ├── brainTumorDataPublic_767-1532/
        ├── brainTumorDataPublic_1533-2298/
        └── brainTumorDataPublic_2299-3064/

Lancer :
    python Exploration_dataset.py

Dépendances (stdlib + numpy + matplotlib uniquement) :
    pip install numpy matplotlib scikit-image
"""

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  ← À MODIFIER
# ══════════════════════════════════════════════════════════════

DOSSIER_DATASET = "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427"

# Pour un test rapide sans tout charger :
#   MAX_PAR_CLASSE = 50  → charge 50 images par classe (150 au total)
#   MAX_PAR_CLASSE = None → charge les 3064 images complètes
MAX_PAR_CLASSE = None


# ══════════════════════════════════════════════════════════════
#  PARTIE 1 — LOADER  (lecture des .mat v7.3 sans h5py)
# ══════════════════════════════════════════════════════════════
#
#  Format HDF5 chunked stocké dans les .mat v7.3 :
#
#  IMAGE  : 8 chunks zlib, chaque chunk décompressé = 65 536 bytes
#           → uint16, 512 lignes × 64 colonnes
#           → assemblage horizontal (hstack) → image 512 × 512
#
#  MASQUE : 4 chunks zlib suivants, décompressés = 65 536 bytes
#           → uint8,  512 lignes × 128 colonnes
#           → assemblage horizontal (hstack) → masque 512 × 512
#
#  LABEL  : float64 stocké dans les métadonnées HDF5 (< 6000 bytes)
#           1 = Méningiome | 2 = Gliome | 3 = Tumeur pituitaire

import os, glob, zlib
import numpy as np

NOMS_TUMEURS = {1: "Méningiome", 2: "Gliome", 3: "Tumeur pituitaire"}
COULEURS     = {1: "#e74c3c",    2: "#3498db", 3: "#2ecc71"}

_SOUS_DOSSIERS = [
    "brainTumorDataPublic_1-766",
    "brainTumorDataPublic_767-1532",
    "brainTumorDataPublic_1533-2298",
    "brainTumorDataPublic_2299-3064",
]

# ── Décompression zlib ────────────────────────────────────────
def _decompress(raw, offset):
    for max_len in [5_000, 10_000, 20_000, 40_000, 80_000]:
        try:
            return zlib.decompress(raw[offset:offset + max_len])
        except Exception:
            pass
    return None

def _find_chunks(raw):
    """Trouve tous les chunks zlib dont la taille décompressée = 65 536 bytes."""
    chunks, i = [], 0
    while i < len(raw) - 2:
        if raw[i] == 0x78 and raw[i + 1] in (0x01, 0x5E, 0x9C, 0xDA):
            data = _decompress(raw, i)
            if data is not None and len(data) == 65_536:
                chunks.append(data)
                i += 100
                continue
        i += 1
    return chunks

def _find_label(raw):
    """Lit le label (1/2/3) dans les 6 000 premiers bytes des métadonnées."""
    PATTERNS = {
        1: b'\x00\x00\x00\x00\x00\x00\xf0\x3f',   # 1.0 en float64 LE
        2: b'\x00\x00\x00\x00\x00\x00\x00\x40',   # 2.0
        3: b'\x00\x00\x00\x00\x00\x00\x08\x40',   # 3.0
    }
    for val, pat in PATTERNS.items():
        pos = raw.find(pat)
        if 0 <= pos < 6000:
            return val
    return None

# ── Lecture d'un fichier .mat ─────────────────────────────────
def lire_mat(chemin):
    """
    Lit un fichier .mat v7.3 et retourne :
        {
          "image"  : np.ndarray (512, 512) uint16  — IRM T1-contraste
          "mask"   : np.ndarray (512, 512) uint8   — 0=sain, 1=tumeur
          "label"  : int   1=Méningiome / 2=Gliome / 3=Tumeur pituitaire
          "fichier": str   nom du fichier
        }
    """
    with open(chemin, "rb") as f:
        f.seek(512)            # sauter le header MATLAB (512 bytes)
        raw = f.read()

    if raw[:8] != b'\x89HDF\r\n\x1a\n':
        raise ValueError(f"Fichier .mat v7.3 invalide : {chemin}")

    chunks = _find_chunks(raw)
    if len(chunks) < 12:
        raise ValueError(f"Chunks insuffisants ({len(chunks)}/12) : {chemin}")

    # ── Image : 8 chunks uint16, 512×64 chacun → hstack → 512×512 ──
    img_bands = [np.frombuffer(c, dtype="<u2").reshape(512, 64) for c in chunks[:8]]
    image = np.hstack(img_bands)           # (512, 512) uint16

    # ── Masque : 4 chunks uint8, 512×128 chacun → hstack → 512×512 ──
    msk_bands = [np.frombuffer(c, dtype=np.uint8).reshape(512, 128) for c in chunks[8:12]]
    mask = (np.hstack(msk_bands) > 0).astype(np.uint8)   # binariser

    return {
        "image":   image,
        "mask":    mask,
        "label":   _find_label(raw),
        "fichier": os.path.basename(chemin),
    }

# ── Chargement du dataset complet ────────────────────────────
def charger_dataset(dossier_racine, max_par_classe=None, verbose=True):
    """
    Charge tous les .mat depuis dossier_racine.
    Cherche automatiquement les sous-dossiers brainTumorDataPublic_*.
    max_par_classe : entier pour limiter (utile pour tests rapides).
    """
    # Résoudre la racine
    racine = dossier_racine
    if not os.path.isdir(os.path.join(racine, _SOUS_DOSSIERS[0])):
        for candidate in [os.path.join(racine, "1512427"),
                           os.path.join(racine, "Dataset", "1512427")]:
            if os.path.isdir(os.path.join(candidate, _SOUS_DOSSIERS[0])):
                racine = candidate
                break

    tous = []
    for sd in _SOUS_DOSSIERS:
        mats = sorted(
            glob.glob(os.path.join(racine, sd, "*.mat")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        tous.extend(mats)

    if verbose:
        print(f"[INFO] {len(tous)} fichiers .mat trouvés dans : {racine}")

    donnees, erreurs = [], 0
    compteur = {1: 0, 2: 0, 3: 0}

    for i, chemin in enumerate(tous):
        try:
            d = lire_mat(chemin)
            lbl = d["label"] or 0
            if max_par_classe and compteur.get(lbl, 0) >= max_par_classe:
                continue
            compteur[lbl] = compteur.get(lbl, 0) + 1
            donnees.append(d)
        except Exception:
            erreurs += 1
        if verbose and (i + 1) % 300 == 0:
            print(f"  ... {i+1}/{len(tous)} fichiers traités")

    if verbose:
        from collections import Counter
        cpt = Counter(d["label"] for d in donnees)
        print(f"\n[OK] {len(donnees)} images chargées  ({erreurs} erreurs)")
        for lbl, nom in NOMS_TUMEURS.items():
            print(f"  Classe {lbl}  {nom:25s} : {cpt.get(lbl, 0):4d} images")

    return donnees

# ── Utilitaires ───────────────────────────────────────────────
def normaliser(image):
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def superposer_masque(img_norm, mask, couleur_hex="#ff0000"):
    r = int(couleur_hex[1:3], 16) / 255
    g = int(couleur_hex[3:5], 16) / 255
    b = int(couleur_hex[5:7], 16) / 255
    rgb = np.stack([img_norm] * 3, axis=-1).astype(np.float32).copy()
    rgb[mask == 1, 0] = 0.55 * img_norm[mask == 1] + 0.45 * r
    rgb[mask == 1, 1] = 0.55 * img_norm[mask == 1] + 0.45 * g
    rgb[mask == 1, 2] = 0.55 * img_norm[mask == 1] + 0.45 * b
    return np.clip(rgb, 0, 1)


# ══════════════════════════════════════════════════════════════
#  PARTIE 2 — EXPLORATION
# ══════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

def statistiques_dataset(donnees):
    cpt = Counter(d["label"] for d in donnees)
    dims = [d["image"].shape for d in donnees]
    print("\n" + "=" * 55)
    print("   STATISTIQUES DU DATASET")
    print("=" * 55)
    print(f"  Nombre total d'images      : {len(donnees)}")
    print(f"  Dimensions                 : {Counter(dims).most_common(1)[0][0]}")
    print()
    for lbl, nom in NOMS_TUMEURS.items():
        n = cpt.get(lbl, 0)
        pct = n / len(donnees) * 100
        bar = "█" * int(pct / 2)
        print(f"  {nom:25s} : {n:4d} ({pct:4.1f}%)  {bar}")
    print()
    print("  Intensité moyenne (uint16) par classe :")
    for lbl, nom in NOMS_TUMEURS.items():
        vals = [d["image"].mean() for d in donnees if d["label"] == lbl]
        if vals:
            print(f"    {nom:25s} : {np.mean(vals):.0f} ± {np.std(vals):.0f}")
    print("  Surface tumorale moyenne (pixels) :")
    for lbl, nom in NOMS_TUMEURS.items():
        vals = [d["mask"].sum() for d in donnees if d["label"] == lbl]
        if vals:
            print(f"    {nom:25s} : {int(np.mean(vals))} ± {int(np.std(vals))}")
    print("=" * 55)


def afficher_exemples(donnees, n_par_classe=3):
    fig, axes = plt.subplots(3, n_par_classe * 2, figsize=(4 * n_par_classe, 10))
    fig.suptitle("Exemples d'images IRM + masques par type de tumeur",
                 fontsize=14, fontweight="bold")

    for row, lbl in enumerate([1, 2, 3]):
        exemples = [d for d in donnees if d["label"] == lbl][:n_par_classe]
        for col, d in enumerate(exemples):
            img_n = normaliser(d["image"])
            c = col * 2
            # Image brute
            axes[row, c].imshow(img_n, cmap="gray")
            axes[row, c].set_title(d["fichier"], fontsize=7)
            axes[row, c].axis("off")
            if col == 0:
                axes[row, c].set_ylabel(NOMS_TUMEURS[lbl], fontsize=9,
                                         color=COULEURS[lbl], fontweight="bold")
                axes[row, c].axis("on")
                axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
                for sp in axes[row, c].spines.values():
                    sp.set_edgecolor(COULEURS[lbl]); sp.set_linewidth(2.5)
            # Superposition
            rgb = superposer_masque(img_n, d["mask"], COULEURS[lbl])
            axes[row, c + 1].imshow(rgb)
            axes[row, c + 1].set_title(f"+ masque ({d['mask'].sum()} px)", fontsize=7)
            axes[row, c + 1].axis("off")
        # Colonnes vides
        for col in range(len(exemples), n_par_classe):
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    plt.tight_layout()
    plt.savefig("exemples_classes.png", dpi=130, bbox_inches="tight")
    print("[OK] Sauvegardé : exemples_classes.png")
    plt.show()


def analyser_image(d):
    img = d["image"]; mask = d["mask"]
    img_n = normaliser(img)
    nom = NOMS_TUMEURS.get(d["label"], "?")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Analyse — {nom}  |  {d['fichier']}", fontsize=13, fontweight="bold")

    # ① Image brute
    im = axes[0, 0].imshow(img_n, cmap="gray")
    axes[0, 0].set_title("Image IRM (T1-contraste, normalisée)")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    axes[0, 0].axis("off")

    # ② Masque
    axes[0, 1].imshow(mask, cmap="hot")
    axes[0, 1].set_title(f"Masque tumoral\n({mask.sum()} pixels)")
    axes[0, 1].axis("off")

    # ③ Superposition
    axes[0, 2].imshow(superposer_masque(img_n, mask, COULEURS.get(d["label"], "#f00")))
    axes[0, 2].set_title("Image + masque superposé")
    axes[0, 2].axis("off")

    # ④ Histogramme tissu sain vs tumeur
    px_sain   = img_n[mask == 0].flatten()
    px_tumeur = img_n[mask == 1].flatten()
    axes[1, 0].hist(px_sain,   bins=80, alpha=0.6, color="steelblue",
                    label=f"Sain ({len(px_sain):,} px)")
    axes[1, 0].hist(px_tumeur, bins=80, alpha=0.7, color="red",
                    label=f"Tumeur ({len(px_tumeur):,} px)")
    axes[1, 0].set_title("Histogramme des intensités")
    axes[1, 0].set_xlabel("Intensité normalisée")
    axes[1, 0].set_ylabel("Nb pixels")
    axes[1, 0].legend(fontsize=8)

    # ⑤ Profil d'intensité horizontal
    mid = img_n.shape[0] // 2
    axes[1, 1].plot(img_n[mid, :], color="gray", lw=1.5, label=f"Ligne {mid}")
    if mask[mid].sum() > 0:
        idxs = np.where(mask[mid] == 1)[0]
        axes[1, 1].axvspan(idxs[0], idxs[-1], alpha=0.3, color="red", label="Zone tumorale")
    axes[1, 1].set_title("Profil d'intensité horizontal")
    axes[1, 1].set_xlabel("Colonne (pixels)"); axes[1, 1].set_ylabel("Intensité")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    # ⑥ Statistiques
    txt = (
        f"Fichier       : {d['fichier']}\n"
        f"Classe        : {nom}\n"
        f"Dimensions    : 512 × 512 px\n\n"
        f"Intensité (uint16)\n"
        f"  min   : {img.min()}\n"
        f"  max   : {img.max()}\n"
        f"  moy.  : {img.mean():.1f}\n"
        f"  σ     : {img.std():.1f}\n\n"
        f"Tumeur\n"
        f"  pixels  : {mask.sum()}\n"
        f"  surface : {mask.sum()/img.size*100:.2f}%\n"
    )
    axes[1, 2].text(0.05, 0.97, txt, transform=axes[1, 2].transAxes,
                    fontsize=9, va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))
    axes[1, 2].axis("off"); axes[1, 2].set_title("Statistiques")

    plt.tight_layout()
    nom_fig = f"analyse_{nom.replace(' ', '_')}.png"
    plt.savefig(nom_fig, dpi=120, bbox_inches="tight")
    print(f"[OK] Sauvegardé : {nom_fig}")
    plt.show()


def graphiques_distribution(donnees):
    cpt = Counter(d["label"] for d in donnees)
    noms = [NOMS_TUMEURS[k] for k in [1, 2, 3]]
    vals = [cpt.get(k, 0) for k in [1, 2, 3]]
    cols = [COULEURS[k] for k in [1, 2, 3]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribution globale du dataset", fontsize=13, fontweight="bold")

    axes[0].pie(vals, labels=noms, colors=cols, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Répartition par type de tumeur")

    bars = axes[1].bar(noms, vals, color=cols, edgecolor="white", width=0.6)
    axes[1].set_title("Nombre d'images par classe")
    axes[1].set_ylabel("Nombre d'images")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].set_ylim(0, max(vals) * 1.15)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 3,
                     str(v), ha="center", fontsize=11, fontweight="bold")

    data_box = [[d["mask"].sum() for d in donnees if d["label"] == lbl]
                for lbl in [1, 2, 3]]
    bp = axes[2].boxplot(data_box, labels=noms, patch_artist=True,
                          medianprops=dict(color="white", lw=2))
    for patch, lbl in zip(bp["boxes"], [1, 2, 3]):
        patch.set_facecolor(COULEURS[lbl]); patch.set_alpha(0.8)
    axes[2].set_title("Surface tumorale par classe (pixels)")
    axes[2].set_ylabel("Pixels dans le masque")
    axes[2].tick_params(axis="x", rotation=12); axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("distribution_dataset.png", dpi=130, bbox_inches="tight")
    print("[OK] Sauvegardé : distribution_dataset.png")
    plt.show()


def comparer_histogrammes(donnees):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Histogrammes d'intensité moyens par classe",
                 fontsize=13, fontweight="bold")
    for lbl in [1, 2, 3]:
        imgs = [normaliser(d["image"]).flatten() for d in donnees if d["label"] == lbl]
        if not imgs:
            continue
        hists = [np.histogram(im, bins=100, range=(0, 1))[0] for im in imgs]
        h = np.mean(hists, axis=0)
        h = h / h.sum()
        bins = np.linspace(0, 1, 100)
        ax.plot(bins, h, color=COULEURS[lbl], lw=2,
                label=NOMS_TUMEURS[lbl], alpha=0.9)
        ax.fill_between(bins, h, alpha=0.12, color=COULEURS[lbl])
    ax.set_xlabel("Intensité normalisée (0–1)")
    ax.set_ylabel("Fréquence normalisée")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("histogrammes_classes.png", dpi=130, bbox_inches="tight")
    print("[OK] Sauvegardé : histogrammes_classes.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  EXPLORATION DU DATASET — TUMEURS CÉRÉBRALES IRM")
    print("=" * 55)

    donnees = charger_dataset(DOSSIER_DATASET, max_par_classe=MAX_PAR_CLASSE)

    if not donnees:
        print(f"\n[ERREUR] Aucune donnée chargée.")
        print(f"  → Vérifiez DOSSIER_DATASET = '{DOSSIER_DATASET}'")
        exit(1)

    statistiques_dataset(donnees)
    afficher_exemples(donnees, n_par_classe=3)

    for lbl in [1, 2, 3]:
        ex = next((d for d in donnees if d["label"] == lbl), None)
        if ex:
            analyser_image(ex)

    graphiques_distribution(donnees)
    comparer_histogrammes(donnees)

    print("\n[TERMINÉ] Tous les graphiques ont été générés.")