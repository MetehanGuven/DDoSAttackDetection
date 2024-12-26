import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from imblearn.over_sampling import SMOTE

# Veri yükleme
data = pd.read_csv('dataset_sdn.csv', sep=';').dropna()

X = data.drop(['dt', 'src', 'dst', 'label'], axis=1, errors='ignore')
y = data['label']
X = pd.get_dummies(X, columns=['Protocol'], drop_first=False)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='passthrough'
)

best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced', random_state=42))
])

best_model.fit(X_train_res, y_train_res)
all_features = X.columns.tolist()

# 1. Özellik önem sıralaması
if isinstance(best_model.named_steps['clf'], RandomForestClassifier):
    feature_importances = best_model.named_steps['clf'].feature_importances_
    sorted_idx = np.argsort(feature_importances)
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(all_features)[sorted_idx])
    plt.title('Feature Importances')
    plt.show()

# 2. Gürültülü veri ile test
noisy_data = X_test.copy()
noisy_data += np.random.normal(0, 0.1, noisy_data.shape)
y_pred_noisy = best_model.predict(noisy_data)
print("Performance on Noisy Data:")
print(classification_report(y_test, y_pred_noisy))

# 3. Çapraz doğrulama
scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print("Cross-Validation Scores:", scores)
print("Mean F1 Score:", np.mean(scores))

# 4. Model tahmin hızı ve boyutu
start_time = time.time()
best_model.predict(X_test[:5000])
end_time = time.time()
print(f"Prediction Time for 5000 samples: {end_time - start_time:.4f} seconds")
model_size = os.path.getsize("best_model.pkl") / (1024 ** 2)
print(f"Model Size: {model_size:.2f} MB")
