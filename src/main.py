import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from regression_logistique import LogisticRegressionCustom
from regression_sci_kit_learn import LogisticRegressionSklearn

custom_model = LogisticRegressionCustom()
sklearn_model = LogisticRegressionSklearn()

# on entraîne le modèle from-scratch
custom_model.fit()

# Ici le but c'est d'afficher des visualisations pour voir si les modèles ont bien prédit ou pas
custom_model.show_predictions_plt()
sklearn_model.show_predictions_plt()

# Là on affiche les matrices de confusion pour voir les performances des deux modèles
custom_model.confusion_matrix_display()
sklearn_model.confusion_matrix_display()

# Calculer les précisions
accuracy_custom = custom_model.get_accuracy()
accuracy_sklearn = sklearn_model.get_accuracy()

print(f"Précision modèle from-scratch: {accuracy_custom * 100:.2f}%")
print(f"Précision scikit-learn: {accuracy_sklearn * 100:.2f}%")

# Différence de précision entre les deux modèles
labels = ['From-scratch', 'Scikit-learn']
accuracies = [accuracy_custom, accuracy_sklearn]
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Précision')
plt.title('Comparaison des précisions')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v*100:.2f}%', ha='center')
plt.show()

# exemple d'affichage de fois où le modèle custom s'est trompé
predictions_custom = custom_model.predict(custom_model.x_normalized)
errors = np.where(predictions_custom != custom_model.digits.target)[0][:5]
fig, axes = plt.subplots(1, 5)
for i, idx in enumerate(errors):
    axes[i].imshow(custom_model.digits.images[idx], cmap='gray')
    axes[i].set_title(f'Réalité: {custom_model.digits.target[idx]}\nPrédiction: {predictions_custom[idx]}')
    axes[i].axis('off')
plt.suptitle('Exemples de prédictions erronées')
plt.show()

'''Pour la réponse à cette question: 
- analyser les résultats obtenus (influence des paramètres, interprétation des coefficients appris, proposition
d’explications des erreurs etc.) et prendre du recul sur le travail effectué (réflexion, difficultés, limites etc.)

Consulter également le rapport écrit.'''