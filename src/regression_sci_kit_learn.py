import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# on charge le dataset
digits = load_digits()
X = digits.data  # l'image des chiffres en 8x8
y = digits.target  # label des chiffres

# le découpage des données - train_test_split avec 80% train / 20% test + stratification
# 80% des données (1437 images des 1797 dispo) -> ensemble d'entraînement pour apprendre le modèle
# 20% des données (360 images des 1797 dispo) -> ensemble de test pour évaluer le modèle sur des données jamais vues
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entraînement du modèle
# max_iter=1000 : nombre maximum d'itérations pour la convergence
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions sur les deux ensembles
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy (train et test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Précision train: {train_accuracy:.4f}")
print(f"Précision test: {test_accuracy:.4f}")

# Le report de classification (précision, recall, f1-score par classe)
print("\nRapport de classification:")
print(classification_report(y_test, y_test_pred))

# Matrice de confusion
print("\nMatrice de confusion:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)


# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vraie classe")
plt.xticks(range(10))
plt.yticks(range(10))

# Ajouter les valeurs dans chaque case
for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )

plt.tight_layout()
# plt.savefig('img/matrice_confusion_sklearn.png') si tu veux save

# Visualisation des erreurs de classification
errors_idx = np.where(y_test != y_test_pred)[0]
if len(errors_idx) > 0:
    print(f"Nombre d'erreurs: {len(errors_idx)}/{len(y_test)}")

    n_show = min(10, len(errors_idx))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flat):
        if i < n_show:
            idx = errors_idx[i]
            ax.imshow(X_test[idx].reshape(8, 8), cmap="gray")
            ax.set_title(f"Vrai: {y_test[idx]} | Préd: {y_test_pred[idx]}", color="red")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("Exemples d'erreurs de classification")
    plt.tight_layout()
    # plt.savefig('img/erreurs_classification_sklearn.png')

plt.show()
