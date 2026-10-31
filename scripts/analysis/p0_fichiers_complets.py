# -*- coding: utf-8 -*-
"""
Consigne 009e, priorite 0.1 : fichiers complets.

  results_multistart_v0.5_complet.csv = lignes v0.4 de Cauchy Point et de Steihaug-CG (code inchange) + lignes v0.5 des
                                        quatre solveurs modifies ; 2772 lignes ; colonne `code_version`.
  results_G3_v0.5_complet.csv         = lignes v0.3 de cauchy et steihaug + lignes v0.5 des quatre solveurs modifies ;
                                        1080 lignes ; colonne `code_version`.

Controles : nombre de lignes, absence de doublons de cle, absence de manques (meme ensemble de cles que le fichier de
reference qui porte les six solveurs), valeurs de 462 departs par solveur (multistart) et 180 par solveur (G3).
Ecrit aussi p0_controles.json. Lecture seule des entrees ; aucune execution de solveur.
"""
import csv
import json
import os
import sys

from paths import (MS_V04_CSV, MS_V05_CSV, G3_V03_CSV, G3_V05_CSV, MS_COMPLET_CSV, G3_COMPLET_CSV, OUT_DIR,
                   OVERRIDES_LONG_TEST_CSV)

UNCHANGED_MS = {"Cauchy_Point", "Steihaug_CG"}
MODIFIED_MS = {"Dogleg", "BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"}
UNCHANGED_G3 = {"cauchy", "steihaug"}
MODIFIED_G3 = {"dogleg", "bfgs_armijo", "bfgs_wolfe", "newton_bt"}
KEY_MS = ("category", "problem", "n", "radius", "draw", "method")
KEY_G3 = ("panel", "method", "kappa", "n", "seed")


def read(path):
    with open(path, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def key(r, cols):
    return tuple(r[c] for c in cols)


def build(ref_path, new_path, cols, unchanged, modified, ref_version, out_path, expected_total, expected_per_method):
    f_ref, ref = read(ref_path)
    f_new, new = read(new_path)
    assert f_ref == f_new, (f_ref, f_new)
    ctl = {"fichier_reference": ref_path, "fichier_nouveau": new_path, "lignes_reference": len(ref),
           "lignes_nouveau": len(new)}
    # doublons de cle dans chaque entree
    for name, rows in (("reference", ref), ("nouveau", new)):
        keys = [key(r, cols) for r in rows]
        ctl[f"doublons_de_cle_{name}"] = len(keys) - len(set(keys))
    new_by_key = {key(r, cols): r for r in new}
    assert all(r["method"] in modified for r in new), "le fichier nouveau contient un solveur non modifie"
    out_rows, missing = [], []
    for r in ref:
        if r["method"] in unchanged:
            r2 = dict(r)
            r2["code_version"] = ref_version
        elif r["method"] in modified:
            k = key(r, cols)
            if k not in new_by_key:
                missing.append(k)
                continue
            r2 = dict(new_by_key[k])
            r2["code_version"] = "v0.5"
        else:
            sys.exit(f"ARRET : solveur inattendu {r['method']}")
        out_rows.append(r2)
    used = {key(r, cols) for r in out_rows if r["code_version"] == "v0.5"}
    surplus = [k for k in new_by_key if k not in used]
    ctl.update({"cles_manquantes_dans_le_nouveau": len(missing), "cles_du_nouveau_sans_correspondant": len(surplus)})
    keys_out = [key(r, cols) for r in out_rows]
    ctl["lignes_complet"] = len(out_rows)
    ctl["doublons_de_cle_complet"] = len(keys_out) - len(set(keys_out))
    ctl["memes_cles_que_la_reference"] = set(keys_out) == {key(r, cols) for r in ref}
    ctl["lignes_par_solveur"] = {m: sum(1 for r in out_rows if r["method"] == m) for m in sorted(unchanged | modified)}
    ctl["lignes_par_code_version"] = {v: sum(1 for r in out_rows if r["code_version"] == v) for v in sorted({r["code_version"] for r in out_rows})}
    ok = (not missing and not surplus and len(out_rows) == expected_total and ctl["doublons_de_cle_complet"] == 0
          and ctl["memes_cles_que_la_reference"] and all(v == expected_per_method for v in ctl["lignes_par_solveur"].values()))
    ctl["controles_reussis"] = ok
    if not ok:
        print(json.dumps(ctl, indent=2, ensure_ascii=False))
        sys.exit("ARRET : controle echoue")
    fields = list(f_ref) + ["code_version"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return ctl


def main():
    ctl = {}
    ctl["multistart"] = build(MS_V04_CSV, MS_V05_CSV, KEY_MS, UNCHANGED_MS, MODIFIED_MS, "v0.4", MS_COMPLET_CSV, 2772, 462)
    ctl["G3"] = build(G3_V03_CSV, G3_V05_CSV, KEY_G3, UNCHANGED_G3, MODIFIED_G3, "v0.3", G3_COMPLET_CSV, 1080, 180)
    ctl["surcharges_test_long"] = {
        "fichier": OVERRIDES_LONG_TEST_CSV, "existe": os.path.exists(OVERRIDES_LONG_TEST_CSV),
        "note": "test_memoize_reprise_cache.jsonl : 427 unites sur 427 attendues (31 s1 + 396 t3), 0 echec, 0 ecart ; "
                "le fichier d'ecarts n'est ecrit que s'il y a des ecarts : aucune surcharge a ajouter. "
                "L'unite (4) securite de la memoire n'est pas enregistree dans le cache : son resultat sur ce passage est non determinable "
                "(reussie lors du passage partiel de 009d)."}
    with open(os.path.join(OUT_DIR, "p0_controles.json"), "w", encoding="utf-8") as fh:
        json.dump(ctl, fh, indent=2, ensure_ascii=False)
    print(json.dumps(ctl, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
