# -*- coding: utf-8 -*-
"""
Chemins communs de la chaine d'analyse (aucun chemin n'est ecrit ailleurs).
Depot publie (v0.5.2) : chemins relatifs a la racine du depot. Lecture seule sur tous les fichiers d'entree (results/) ;
toutes les sorties vont sous OUT_DIR, fixe par la variable d'environnement ANALYSIS_OUT (defaut : analysis_output/ a la
racine du depot).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
REFERENCE_DIR = os.path.join(RESULTS, "reference")  # sorties de versions anterieures, lues pour les comparaisons
RERUN = os.path.join(REPO, "rerun")                  # sorties d'une nouvelle execution de scripts/run (non distribuees)
OUT_DIR = os.environ.get("ANALYSIS_OUT", os.path.join(REPO, "analysis_output"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
ROB_DIR = os.path.join(OUT_DIR, "robustesse")
for _d in (OUT_DIR, TABLES_DIR, FIG_DIR, ROB_DIR):
    os.makedirs(_d, exist_ok=True)

# --- resultats (lecture seule)
BENCH_CSV = os.path.join(RESULTS, "resultats_benchmark_v0.5.csv")                  # 1080 lignes, colonne code_version
BENCH_V04_CSV = os.path.join(REFERENCE_DIR, "resultats_benchmark_v0.4.csv")
MS_V04_CSV = os.path.join(REFERENCE_DIR, "results_multistart_v0.4.csv")
MS_V05_CSV = os.path.join(RERUN, "results_multistart_v0.5.csv")                    # 4 solveurs modifies (scripts/run)
G3_V03_CSV = os.path.join(REFERENCE_DIR, "results_G3_v0.3.csv")
G3_V05_CSV = os.path.join(RERUN, "results_G3_v0.5.csv")                            # 4 solveurs modifies (scripts/run)
MS_COMPLET_CSV = os.path.join(RESULTS, "results_multistart_v0.5_complet.csv")      # construit par p0_fichiers_complets
G3_COMPLET_CSV = os.path.join(RESULTS, "results_G3_v0.5_complet.csv")              # construit par p0_fichiers_complets
OVERRIDES_LONG_TEST_CSV = os.path.join(RERUN, "test_memoize_reprise_ecarts.csv")   # facultatif (p0 note s'il existe)

# --- chronometrage v0.5 (passage A = mesure principale ; passage B = replication)
TIMING_A_CSV = os.path.join(RESULTS, "timing", "timing_catA_v0.5_A.csv")
TIMING_B_CSV = os.path.join(RESULTS, "timing", "timing_catA_v0.5_B.csv")
CONTROL_A_CSV = os.path.join(RESULTS, "timing", "control_measurement_v0.5_A.csv")
CONTROL_B_CSV = os.path.join(RESULTS, "timing", "control_measurement_v0.5_B.csv")

# --- references anterieures (lecture seule)
DIR_008D_BIS = os.path.join(REFERENCE_DIR, "analyse_v0.4")


def out(*parts):
    return os.path.join(OUT_DIR, *parts)
