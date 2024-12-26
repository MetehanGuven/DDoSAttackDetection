import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from imblearn.over_sampling import SMOTE

# Veri yükleme
data = pd.read_csv('dataset_sdn.csv', sep=';')

print("Data Shape:", data.shape)
print(data.info())
print(data.head())

print("Missing Values:\n", data.isnull().sum())

# NaN verileri çıkarma
data = data.dropna()

# Özellik ve etiket ayrımı
X = data.drop(['dt', 'src', 'dst', 'label'], axis=1, errors='ignore')
y = data['label']

# Kategorik sütunları one-hot encode edelim
# Böylece Protocol sütunu TCP, UDP, ICMP gibi kategoriler sayısala çevrilir
X = pd.get_dummies(X, columns=['Protocol'], drop_first=False)
# Artık X tamamen sayısal, SMOTE uygulanabilir.

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# Artık kategorik sütun yok, hepsi sayısal.
categorical_cols = []  # Boş çünkü get_dummies ile encode ettik.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE uygulama
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE, class distribution:")
print(y_train_res.value_counts())

# Preprocessor: Artık sadece scale yapacağız, çünkü kategorik encoding yaptık.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
        # cat yok artık, çünkü get_dummies yaptık
    ],
    remainder='passthrough'  # Geri kalan sütunları olduğu gibi bırak
)

models_and_parameters = [
    (LogisticRegression(max_iter=1000),
     {
         'clf__C': [0.1, 1, 10],
         'clf__solver': ['liblinear', 'lbfgs'],
         'clf__class_weight': ['balanced', None]
     }),
    (RandomForestClassifier(random_state=42),
     {
         'clf__n_estimators': [100, 200, 500],
         'clf__max_depth': [5, 10, 20],
         'clf__min_samples_split': [2, 5],
         'clf__class_weight': ['balanced', None]
     })
]

best_models = {}
best_scores = {}

for model, params in models_and_parameters:
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

    print(f"Training {model.__class__.__name__}...")
    # Artık SMOTE uygulanmış verilerle eğitiyoruz
    grid = GridSearchCV(pipe, params, scoring='f1', cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train_res, y_train_res)
    
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Model:", model.__class__.__name__)
    print("Best Params:", grid.best_params_)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    print("\n" + "="*50 + "\n")
    
    best_models[model.__class__.__name__] = grid.best_estimator_
    best_scores[model.__class__.__name__] = f1

# En iyi modeli seç
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

all_features = X_train.columns.tolist()

dump((best_model, all_features), "best_model.pkl")
print("Model and features saved to best_model.pkl")
print("Best Model:", best_model_name)
print("Best Model F1:", best_scores[best_model_name])
