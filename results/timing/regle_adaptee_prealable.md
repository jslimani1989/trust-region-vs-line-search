# Règle de robustesse (priorité 3) — adaptation consignée AVANT le calcul des variantes R1, R2, R3

Écrit le 2026-10-13, après le calcul de l'analyse principale (passage A) et avant celui de R1, R2 et R3.

**Fait observé dans l'analyse principale** (`b1_seuil_noeud2.json`) : le rapport de temps Newton-LM / Steihaug-CG (moyenne géométrique, cellules d'au moins 3 problèmes) vaut 0,51 (n = 10), 0,36 (50), 0,39 (100), 0,63 (500), 0,65 (1000) : il ne passe jamais au-dessus de 1. **n\* est non déterminable** (aucun croisement montant). La consigne 008c prévoit ce cas : « Si le rapport ne franchit jamais 1 : l'écrire, et le nœud 2 est candidat à la suppression ».

**Conséquence pour la règle de la consigne 009e.** Le critère (i) « n\* reste dans l'intervalle de l'analyse principale » n'a pas d'intervalle de référence. Adaptation retenue, fixée ici :

- (i') la conclusion de l'analyse principale (« aucun croisement montant dans les cellules retenues ») est conservée dans la variante : n\* est aussi non déterminable dans R1, R2 (lecture « l'une des deux ») et R3. Si une variante fait apparaître un croisement, (i') n'est pas satisfait ; ce croisement est rapporté avec sa valeur.
- (ii) inchangé : les solveurs du front de Pareto sont les mêmes (ensemble de solveurs sur le front) ;
- (iii) inchangé : l'ordre des quatre règles sur la **perte de succès** est inchangé (ordre faible, égalité à 10⁻⁹ près) ; les ordres sur le regret sont rapportés sans critère.

**Convention de calcul** (nécessaire pour appliquer les quatre règles et la validation hors échantillon quand n\* n'existe pas) : seuil n\* = +∞, c'est-à-dire qu'aucune instance n'est « non abordable ». La règle « temps d'abord » (Newton-LM si n < n\*, Steihaug-CG sinon) devient alors « Newton-LM toujours » et coïncide avec « robustesse d'abord » dans tout pli où n\* n'est pas défini ; dans l'arbre appris, les feuilles L4 et L5 sont vides. Dans la validation hors échantillon, n\* est réappris dans chaque pli ; un pli où il n'est pas défini est **conservé** (seuil +∞), non écarté.
