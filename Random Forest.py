import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

aoeData = pd.read_csv('data/aoe_data.csv')
aoeData = aoeData.dropna()

features = aoeData[['duration', 'map', 'elo', 'p1_civ', 'p2_civ', 'p1_xpos', 'p2_xpos', 'p1_ypos', 'p2_ypos', 'winner']]

target = features['winner']
features = features.drop('winner', axis=1)

featuresScaled = features.copy()

# Scaling
standardScaler = StandardScaler()
minMaxScaler = MinMaxScaler()

# Outliers values => standadization
featuresScaled['duration'] = standardScaler.fit_transform(features[['duration']])

# Uniform/gaussian => normalization
columnsToMinMax = ['elo', 'p1_xpos', 'p2_xpos', 'p1_ypos', 'p2_ypos']
featuresScaled[columnsToMinMax] = minMaxScaler.fit_transform(features[columnsToMinMax])

# Categorical values
featuresScaled = pd.get_dummies(featuresScaled, columns=['p1_civ', 'p2_civ'], prefix=['p1_civ', 'p2_civ'])
featuresScaled = pd.get_dummies(featuresScaled, columns=['map'], prefix=['map'])

# Splitting
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    featuresScaled,
    target,
    test_size=0.25,
    random_state=1,
    stratify=target
)

# Model
rf_model = RandomForestClassifier(n_estimators=20, random_state=1)

# Cross validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Mean precision
cross_val_accuracy = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"Cross-Validation Accuracy: {cross_val_accuracy:.2f}")