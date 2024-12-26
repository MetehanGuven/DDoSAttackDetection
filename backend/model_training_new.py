import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri Yükleme
data = pd.read_csv('combined_dataset.csv')

# 2. Modelin Beklediği Sütunları Yükle
model, expected_columns = joblib.load('best_model.pkl')
print("Modelin Beklediği Sütunlar:", expected_columns)

# 3. Veri Ön İşleme
# Gereksiz sütunları çıkar
data = data.drop(['dt', 'src', 'dst', 'port_no'], axis=1, errors='ignore')

# Eksik değerleri kontrol et
print("Eksik Değerler:\n", data.isnull().sum())

# Eksik değerleri doldur
data = data.fillna(data.mean())

# Eksik değerlerin tekrar kontrolü
print("Eksik Değerler Temizlendikten Sonra:\n", data.isnull().sum())

# Protocol sütununu sayısal değerlere dönüştür
data['Protocol'] = data['Protocol'].astype(str)
data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

# Eksik sütunları sıfır ile doldur
for col in expected_columns:
    if col not in data.columns:
        data[col] = 0

# Fazla sütunları çıkar
data = data[expected_columns + ['label']]  # 'label' sütununu koruyoruz

# 4. Hedef Değişken ve Özellik Ayrımı
X = data.drop(['label'], axis=1)  # Hedef sütunu çıkar
y = data['label']  # Hedef sütun

# 5. Eğitim ve Test Verilerini Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. SMOTE ile Dengesiz Veri İşleme
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("SMOTE Sonrası Sınıf Dağılımı:\n", y_train_res.value_counts())

# 7. Model Tanımlama ve Eğitimi
new_model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
new_model.fit(X_train_res, y_train_res)

# 8. Model Değerlendirmesi
y_pred = new_model.predict(X_test)
y_proba = new_model.predict_proba(X_test)[:, 1]

# Performans Metrikleri
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))

# Confusion Matrix Görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. Özellik Önem Sıralaması
if hasattr(new_model, 'feature_importances_'):
    feature_importances = new_model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importances)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Feature Importances')
    plt.show()

# 10. Yeni Modeli Kaydetme
joblib.dump((new_model, expected_columns), 'ddos_detection_new_model.pkl')
print("\nYeni model başarıyla kaydedildi: ddos_detection_new_model.pkl")
