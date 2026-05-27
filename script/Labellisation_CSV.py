"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           LABELLISATION MANUELLE DES COUPES IRM                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Ce script tire 50 images aléatoires dans le dataset, les affiche          ║
║  une par une et demande à l'utilisateur d'assigner une coupe.              ║
║                                                                              ║
║  Choix possibles pour chaque image :                                        ║
║    1 → Coupe AXIALE                                                         ║
║    2 → Coupe SAGITTALE                                                      ║
║    3 → Coupe CORONALE                                                       ║
║    4 → Watershed AMÉLIORÉ                                                   ║
║    0 → Ignorer cette image (ne sera pas utilisée)                          ║
║                                                                              ║
║  Résultat : fichier coupes.csv                                              ║
║    fichier,methode                                                           ║
║    /chemin/123.mat,1                                                         ║
║    /chemin/456.mat,3                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

UTILISATION :
    python labellisation_coupes.py

Le script reprend là où il s'est arrêté si vous l'interrompez
(les images déjà labellisées sont conservées dans coupes.csv).
"""

import h5py
import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶▶  PARAMÈTRES À MODIFIER ICI  ◀◀◀
# ══════════════════════════════════════════════════════════════════════════════

DOSSIERS_DATASET = [
    "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_1-766",
    "/Users/loeuljeanpierre/Library/Mobile Documents/com~apple~CloudDocs/Documents/2-ESEO/2-2eAnnee/8-TNI/5-Projet/Dataset/1512427/brainTumorDataPublic_767-1532",
]

# Fichier CSV de sortie
FICHIER_CSV = "./coupes.csv"

# Nombre d'images à labelliser
NB_IMAGES = 50

# Graine aléatoire
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════


def normaliser(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def lister_fichiers(dossiers):
    fichiers = []
    for d in dossiers:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.mat'):
                    fichiers.append(os.path.join(d, f))
    return fichiers


def charger_deja_labelises(fichier_csv):
    """Charge les fichiers déjà labellisés pour reprendre où on s'est arrêté."""
    deja = {}
    if os.path.isfile(fichier_csv):
        with open(fichier_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                deja[row['fichier']] = row['methode']
    return deja


def afficher_image(img_n, nom, numero, total, deja_faites):
    """
    Affiche l'image IRM avec un panneau d'aide visuel.
    Bloque jusqu'à fermeture → puis saisie dans le terminal.
    """
    fig = plt.figure(figsize=(10, 6))

    # Image principale
    ax_img = fig.add_axes([0.05, 0.1, 0.55, 0.8])
    ax_img.imshow(img_n, cmap='gray')
    ax_img.set_title(f"{nom}", fontsize=12, fontweight='bold')
    ax_img.axis('off')

    # Panneau d'aide à droite
    ax_help = fig.add_axes([0.63, 0.1, 0.34, 0.8])
    ax_help.axis('off')

    texte_aide = (
        f"Image {numero}/{total}\n"
        f"Déjà labellisées : {deja_faites}\n\n"
        "─────────────────────────\n"
        "CHOIX DE COUPE :\n\n"
        "  1 → AXIALE\n"
        "      (vue du dessus,\n"
        "       cerveau centré)\n\n"
        "  2 → SAGITTALE\n"
        "      (vue de côté,\n"
        "       colonne visible - 511)\n\n"
        "  3 → CORONALE\n"
        "      (vue de face,\n"
        "       symétrie gauche/droite)\n\n"
        "  4 → WATERSHED\n"
        "      (coupes complexes)\n\n"
        "  0 → IGNORER\n"
        "      (image inutilisable - 559)\n\n"
        "─────────────────────────\n"
        "Fermez la fenêtre\n"
        "puis tapez votre choix."
    )

    ax_help.text(0.05, 0.98, texte_aide,
                 transform=ax_help.transAxes,
                 fontsize=10, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           alpha=0.8, edgecolor='gray'))

    fig.suptitle(
        "→ Regardez l'image, FERMEZ la fenêtre, puis choisissez dans le terminal",
        fontsize=11, color='darkred', fontweight='bold'
    )

    plt.show(block=True)
    plt.close('all')


def saisir_choix(nom):
    """Demande le choix de coupe dans le terminal après fermeture de la fenêtre."""
    noms_methode = {
        '1': 'AXIALE',
        '2': 'SAGITTALE',
        '3': 'CORONALE',
        '4': 'WATERSHED',
        '0': 'IGNORÉE',
    }
    while True:
        try:
            choix = input(
                f"\n  {nom}\n"
                "  Coupe → 1=Axiale  2=Sagittale  3=Coronale  "
                "4=Watershed  0=Ignorer : "
            ).strip()
            if choix in ('0', '1', '2', '3', '4'):
                print(f"  ✓ Assigné : {noms_methode[choix]}")
                return choix
            print("  ⚠  Entrez 0, 1, 2, 3 ou 4.")
        except (ValueError, KeyboardInterrupt):
            print("\n  ⚠  Interruption — progression sauvegardée dans le CSV.")
            raise


def sauvegarder_csv(fichier_csv, donnees):
    """Écrit/réécrit le CSV complet avec toutes les entrées."""
    with open(fichier_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fichier', 'methode'])
        writer.writeheader()
        for chemin, methode in donnees.items():
            writer.writerow({'fichier': chemin, 'methode': methode})


def main():
    print("\n" + "═"*60)
    print("  LABELLISATION DES COUPES IRM")
    print("═"*60)

    # Liste tous les fichiers disponibles
    print("\n  Scan des dossiers dataset...")
    fichiers = lister_fichiers(DOSSIERS_DATASET)
    if not fichiers:
        print("  ✗ Aucun fichier .mat trouvé.")
        print("     → Vérifiez DOSSIERS_DATASET en haut du script.")
        return
    print(f"  {len(fichiers)} fichiers .mat trouvés")

    # Charger les labels déjà faits (les 0 ne sont pas dans le CSV)
    deja = charger_deja_labelises(FICHIER_CSV)
    if deja:
        print(f"  ↩  Déjà labellisées : {len(deja)} images utiles dans le CSV")

    # Exclure uniquement les images avec coupe valide (1-4) déjà sauvegardées
    # Les ignorées (0) ne sont pas dans le CSV → disponibles pour re-tirage
    fichiers_disponibles = [f for f in fichiers if f not in deja]
    print(f"  {len(fichiers_disponibles)} images disponibles")

    nb_a_tirer = min(NB_IMAGES, len(fichiers_disponibles))
    # Pas de seed fixe → nouvelles images à chaque relance
    selection = random.sample(fichiers_disponibles, nb_a_tirer)
    print(f"  {len(selection)} nouvelles images tirées aléatoirement")
    a_traiter = selection
    print("\n  → Fermez chaque fenêtre puis entrez votre choix dans le terminal.")
    print("  → Ctrl+C pour interrompre (progression sauvegardée).\n")
    print("═"*60)

    resultats = dict(deja)  # on repart des labels existants

    try:
        for i, chemin in enumerate(a_traiter):
            nom = os.path.basename(chemin)
            numero_global = len(deja) + i + 1

            # Chargement image
            try:
                with h5py.File(chemin, 'r') as f:
                    img = np.array(f['cjdata']['image']).T.astype(np.float32)
                img_n = normaliser(img)
            except Exception as e:
                print(f"\n  ✗ Erreur chargement {nom} : {e} — ignorée")
                # image non chargeable → ignorée, pas sauvegardée
                sauvegarder_csv(FICHIER_CSV, resultats)
                continue

            # Affichage
            afficher_image(img_n, nom, numero_global, NB_IMAGES, len(resultats))

            # Saisie du choix
            choix = saisir_choix(nom)

            # Sauvegarde immédiate — les 0 ne sont PAS sauvegardés
            # → l'image reste disponible pour une prochaine session
            if choix != '0':
                resultats[chemin] = choix
                sauvegarder_csv(FICHIER_CSV, resultats)
            else:
                print("  ↷  Ignorée — non sauvegardée, disponible pour la suite")

    except KeyboardInterrupt:
        print(f"\n\n  Interruption — {len(resultats)} images sauvegardées dans {FICHIER_CSV}")

    # ── Bilan final ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  BILAN LABELLISATION")
    print(f"{'═'*60}")

    compteurs = {'1': 0, '2': 0, '3': 0, '4': 0, '0': 0}
    for v in resultats.values():
        if v in compteurs:
            compteurs[v] += 1

    noms = {'1': 'Axiale', '2': 'Sagittale', '3': 'Coronale',
            '4': 'Watershed', '0': 'Ignorées'}
    for k, n in noms.items():
        print(f"  {n:<12} : {compteurs[k]:>3} images")

    total_utiles = sum(compteurs[k] for k in ('1','2','3','4'))
    print(f"  {'─'*25}")
    print(f"  {'Total utiles':<12} : {total_utiles:>3} images")
    print(f"\n  ✓ CSV sauvegardé : {FICHIER_CSV}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()