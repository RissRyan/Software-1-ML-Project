import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/aoe_data.csv')

df = df.dropna()

# map_size, dataset & difficulty unique value (coz 1v1 ranked)

# Sélectionnez les colonnes pertinentes
features = df[['duration', 'elo', 'p1_civ', 'p2_civ', 'p1_xpos', 'p2_xpos', 'p1_ypos', 'p2_ypos']]

# Encodez les variables catégorielles (p1_civ, p2_civ) si nécessaire
features = pd.get_dummies(features, columns=['p1_civ', 'p2_civ'])

# Sélectionnez la variable cible (winner)
target = df['winner']

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

# Choisissez un modèle de classification (par exemple, RandomForestClassifier)
model = RandomForestClassifier()

# Entraînez le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluez la performance du modèle
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')