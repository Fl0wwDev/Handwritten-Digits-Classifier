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


# Sources:
https://scikit-learn.org/0.17/modules/generated/sklearn.datasets.load_digits.html
https://stackoverflow.com/questions/3823752/display-image-as-grayscale (bug affichage)
https://www.youtube.com/watch?v=2ztuQKtW7So
cours
td5 (ex 2 regression linéaire)
doc matplotlib