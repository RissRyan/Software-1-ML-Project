import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('data/aoe_data.csv')
df = df.dropna()

# Utiliser 1/n % du jeu de données (ajustez n en fonction de vos besoins)
subset_percentage = 3  # Utilisez 10% du jeu de données
df_subset = df.sample(frac=(subset_percentage / 100), random_state=1)

# Sélectionnez les colonnes pertinentes
features = df_subset[['duration', 'elo', 'p1_civ', 'p2_civ', 'p1_xpos', 'p2_xpos', 'p1_ypos', 'p2_ypos', 'winner']]

# One-hot encode the civilization columns for p1 and p2
features = pd.get_dummies(features, columns=['p1_civ', 'p2_civ'], prefix=['p1_civ', 'p2_civ'])

# Sélectionnez la variable cible (winner)
target = features['winner']
features = features.drop('winner', axis=1)

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1, stratify=target)

# Normalisez les données (pour RandomForest, la normalisation n'est généralement pas nécessaire)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Créez le modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Entraînez le modèle sur l'ensemble d'entraînement
rf_model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_proba_rf > 0.5).astype(int)

# Calculez la matrice de confusion
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Créez un DataFrame pour afficher la matrice de confusion
conf_df_rf = pd.DataFrame(conf_matrix_rf, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

# Affichez la matrice de confusion
print(conf_df_rf)

# Visualisez la matrice de confusion avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Matrice de Confusion - Random Forest')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()
