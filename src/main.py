import numpy as np
import matplotlib.pyplot as plt
from regression_logistique import LogisticRegressionCustom
from regression_sci_kit_learn import LogisticRegressionSklearn
from sklearn.datasets import load_digits
from tensorflow.keras.datasets import mnist

choice = input("\n--------------------------------\nChoisissez le dataset à utiliser \n 1- load_digits de sklearn.datasets (default) \n 2- mnist de tensorflow.keras.datasets \n Choisir (1/2): ")

def load_sklearn_digits_data():
    dataset = load_digits()
    digits_data = dataset.data
    digits_target = dataset.target
    image_shape = (8, 8)
    max_iter = 1000
    return dataset, digits_data, digits_target, image_shape, max_iter

def load_mnist_data():
    (x_train, y_train), _ = mnist.load_data()
    dataset = (x_train, y_train)
    digits_data = x_train.reshape(-1, 28*28)
    digits_target = y_train
    image_shape = (28, 28)
    max_iter = 5000
    return dataset, digits_data, digits_target, image_shape, max_iter

if int(choice) == 1:
    dataset, digits_data, digits_target, image_shape, max_iter = load_sklearn_digits_data()
elif int(choice) == 2:
    dataset, digits_data, digits_target, image_shape, max_iter = load_mnist_data()
else:
    dataset, digits_data, digits_target, image_shape, max_iter = load_sklearn_digits_data()
custom_model = LogisticRegressionCustom(max_iterations=max_iter, dataset=dataset, digits_data=digits_data, digits_target=digits_target, image_shape=image_shape)
sklearn_model = LogisticRegressionSklearn(max_iterations=max_iter, dataset=dataset, digits_data=digits_data, digits_target=digits_target, image_shape=image_shape)
# on entraîne le modèle from-scratch
custom_model.fit()

#--------------------------------------------------------------
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
    plt.text(i, v + 0.01, f'{v*100:.2f}%')
plt.show()


#--------------------------------------------------------------
# Ici le but c'est d'afficher des visualisations pour voir si les modèles ont bien prédit ou pas
custom_model.show_predictions_plt()
sklearn_model.show_predictions_plt()


#--------------------------------------------------------------
# Là on affiche les matrices de confusion pour voir les performances des deux modèles
custom_model.confusion_matrix_display()
sklearn_model.confusion_matrix_display()


#--------------------------------------------------------------
# Visualisation des poids appris par les deux modèles pour voir les patterns moyens appris pour chaque chiffre
fig, axes = plt.subplots(2, 10, figsize=(25, 8))

for digit in range(10):
    # Modèle custom
    weights_custom = custom_model.all_weights[:, digit].reshape(image_shape)
    vmin = -max(np.abs(weights_custom).max(), np.abs(sklearn_model.model.coef_[digit]).max())
    vmax = -vmin
    im1 = axes[0, digit].imshow(weights_custom, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0, digit].set_title(f'Digit {digit}\nCustom', fontsize=9)
    axes[0, digit].axis('off')

    # Modèle sklearn
    weights_sklearn = sklearn_model.model.coef_[digit].reshape(image_shape)
    im2 = axes[1, digit].imshow(weights_sklearn, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[1, digit].set_title(f'Digit {digit}\nSklearn', fontsize=9)
    axes[1, digit].axis('off')

plt.suptitle('Patterns moyens appris par les deux modèles', fontsize=14, y=0.98)
plt.tight_layout()
plt.show()


#--------------------------------------------------------------
# exemple d'affichage de fois où le modèle custom s'est trompé
predictions_custom = custom_model.predict(custom_model.x_normalized)
errors = np.where(predictions_custom != custom_model.digits_target)[0][:5]
fig, axes = plt.subplots(1, 5)
for i, idx in enumerate(errors):
    if hasattr(custom_model.digits, 'images'):
        image = custom_model.digits.images[idx]
    else:
        image = custom_model.digits_data[idx].reshape(image_shape)
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Réalité: {custom_model.digits_target[idx]}\nPrédiction: {predictions_custom[idx]}')
    axes[i].axis('off')
plt.suptitle('Exemples de prédictions erronées')
plt.show()

'''Pour la réponse à cette question: 
- analyser les résultats obtenus (influence des paramètres, interprétation des coefficients appris, proposition
d’explications des erreurs etc.) et prendre du recul sur le travail effectué (réflexion, difficultés, limites etc.)

Consulter également le rapport écrit.'''