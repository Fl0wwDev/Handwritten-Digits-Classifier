## Analyse des résultats

- **Influence des paramètres :**
    - **Learning rate (0.1)** : Trop élevé peut faire diverger l'apprentissage (c'est-à-dire que le modèle s'éloigne de la bonne solution au lieu de s'en rapprocher), trop bas ralentit la convergence (le modèle apprend trop lentement). Ici, stable mais pourrait être optimisé (ex : 0.01 pour un apprentissage plus lent mais plus précis).
    - **Batch size (32)** : Le mini-batch accélère par rapport au full batch (au lieu de traiter toutes les données d'un coup, on en prend un petit groupe, ce qui rend l'apprentissage plus rapide), mais peut introduire du bruit (des variations aléatoires qui rendent l'apprentissage moins stable). Testé avec 64 pour plus de stabilité (moins de bruit).
    - **Régularisation L2 (0.01)** : Réduit l'overfitting (quand le modèle apprend trop bien les données d'entraînement et ne généralise pas bien aux nouvelles données), améliore la précision de 91% à 97%. Sans, risque de surapprentissage (le modèle mémorise au lieu d'apprendre des patterns utiles).
    - **Iterations (1000)** : Suffisant pour la convergence (atteindre une bonne solution), mais `epsilon=1e-6` peut arrêter tôt si le gradient est faible (si les changements deviennent trop petits, on arrête pour éviter de tourner en rond).

- **Interprétation des coefficients :**
    - Les coefficients positifs/négatifs indiquent les pixels importants pour la classe (ex : pour 0, pixels centraux positifs signifient que ces pixels aident à reconnaître un 0, négatifs signifient qu'ils indiquent le contraire). La heatmap montre que le modèle apprend des patterns locaux (des formes spécifiques dans l'image), similaires à scikit-learn mais moins raffinés (moins précis dans les détails).

- **Explications des erreurs :**
    - Erreurs sur chiffres similaires (ex : 4 et 9, 3 et 8) dues à des similarités visuelles (ils se ressemblent beaucoup, comme un 4 qui pourrait être confondu avec un 9 mal écrit).
    - Bruit dans les données (pixels mal échantillonnés, comme des taches ou des imperfections dans l'image) ou manque de données d'entraînement (pas assez d'exemples pour bien apprendre).
    - Notre modèle atteint 97% contre 99% pour scikit-learn car GD basique vs. solveurs avancés (`lbfgs`, qui sont des méthodes plus intelligentes pour trouver la meilleure solution).

- **Réflexion sur le travail :**
    - **Difficultés** : Implémentation from-scratch difficulté pour obtenir la même précision que le modèle de scikit learn.  
      Dans la première version du code, lorsqu'on comparait avec scikit-learn, on obtenait une précision de **66%** contre **99%**.  
      On a donc commencé par ajouter un `StandardScaler` pour normaliser les données (mettre toutes les valeurs sur la même échelle pour que le modèle apprenne mieux), ce qui a permis d'augmenter déjà la précision.  
      Ensuite, on ajouté une régularisation **L2** dans le calcul du gradient (une pénalité pour éviter que le modèle ne devienne trop complexe), ce qui a permis d'atteindre **97%**. Le processus a vraiment ralenti car il fallait tester et ajuster chaque partie manuellement.
    - **Limites** : Pas de validation croisée (vérifier le modèle sur des données séparées pour éviter le surapprentissage), optimisation manuelle des hyperparamètres (choisir les bons paramètres à la main au lieu d'une recherche automatique), sensibilité aux initialisations (le résultat dépend du point de départ).
    - **Apports** : Compréhension profonde des mécanismes (sigmoïde pour la probabilité, gradient pour ajuster les poids), base pour extensions (CNN pour des images plus complexes, etc.).
    - **Améliorations possibles** : Ajouter early stopping (arrêter l'apprentissage tôt si ça ne s'améliore plus), tester d'autres régularisations (comme L1 pour simplifier le modèle), ou intégrer dans un pipeline ML (une chaîne d'outils automatisée).

---

## Interprétation des visualisations

### 1. Comparaison des précisions (Bar Chart)
- **Description** : Graphique à barres comparant les précisions des deux modèles (from-scratch vs scikit-learn).
- **Interprétation** : Le modèle custom atteint **97%** de précision, proche des **99%** de scikit-learn. L'écart de 2% s'explique par les solveurs plus avancés de sklearn (`lbfgs`) et une optimisation plus poussée. Cette comparaison montre que notre implémentation from-scratch capture l'essentiel de la logique de régression logistique malgré sa simplicité.

### 2. Prédictions sur échantillons (show_predictions_plt)
- **Description** : Affichage de quelques images du dataset avec leurs prédictions par chaque modèle.
- **Interprétation** : Permet de vérifier visuellement que les modèles prédisent correctement les chiffres. La plupart des prédictions sont exactes, ce qui confirme la bonne généralisation des modèles. Les erreurs visibles permettent d'identifier les cas difficiles (chiffres mal écrits, ambigus).

### 3. Matrices de confusion
- **Description** : Matrices montrant pour chaque classe réelle (ligne) combien de prédictions sont correctes (diagonal) ou erronées (hors diagonal).
- **Interprétation** : 
    - Les valeurs sur la diagonale indiquent les bonnes classifications (ex : tous les "0" correctement prédits comme "0").
    - Les erreurs hors diagonale révèlent les confusions fréquentes (ex : certains "4" prédits comme "9", ou "3" confondus avec "8").
    - Les deux modèles montrent des patterns d'erreurs similaires, confirmant que les limites viennent des données (similarités visuelles) plutôt que de l'algorithme.
    - Une matrice bien concentrée sur la diagonale = bonne performance.

### 4. Heatmaps des poids (patterns moyens)
- **Description** : Visualisation des coefficients appris par chaque modèle pour chaque chiffre, organisés en grille 8×8 (forme des images).
- **Interprétation** :
    - **Rouge** (valeurs positives) : Pixels qui augmentent la probabilité d'appartenir à cette classe (zones importantes pour reconnaître le chiffre).
    - **Bleu** (valeurs négatives) : Pixels qui diminuent cette probabilité (zones qui indiquent que ce n'est PAS ce chiffre).
    - **Blanc** (proche de 0) : Pixels non informatifs pour cette classe.
    - **Comparaison Custom vs Sklearn** : Les patterns sont similaires mais sklearn montre des poids plus contrastés et précis, résultat d'une optimisation plus fine.
    - Exemple : Pour le chiffre "0", on voit du rouge au centre (forme ronde) et du bleu à l'extérieur. Pour "1", du rouge sur une ligne verticale.
    - Ces heatmaps montrent que le modèle a bien appris les formes caractéristiques de chaque chiffre.

### 5. Exemples de prédictions erronées
- **Description** : Affichage des 5 premières erreurs du modèle custom avec la réalité vs la prédiction.
- **Interprétation** :
    - Permet d'analyser pourquoi le modèle échoue : chiffres mal formés, ambigus, ou pixelisés.
    - Souvent, les erreurs sont compréhensibles même pour un humain (ex : un "9" écrit comme un "4").
    - Identifie les limites du modèle linéaire : ne capture pas les variations complexes d'écriture manuscrite.
    - Aide à comprendre qu'un modèle plus complexe (CNN) pourrait mieux gérer ces cas difficiles.

---

## Bonus

- Rajout du dataset mnist: beaucoup plus lent car dataset de 60k image et image de 28x28, précision plus des prédictions alors que les images sont plus "détaillées"
Essaie avec des iterations beaucoup plus élevées (5000, 10000) mais process très lent
- Neural network: précision assez basse, mais c'est une implémantation assez basique basée sur le cours de fouille de données sur le machine learning

# Sources:
- https://scikit-learn.org/0.17/modules/generated/sklearn.datasets.load_digits.html
- https://stackoverflow.com/questions/3823752/display-image-as-grayscale (bug affichage)
- https://www.youtube.com/watch?v=2ztuQKtW7So
- cours
- td5 (ex 2 regression linéaire)
- doc matplotlib