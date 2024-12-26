from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Model ve beklenen kolonları yükle
model, expected_columns = joblib.load("best_model.pkl")

def process_wireshark_data(df):
    """
    Wireshark CSV verisini modelin beklediği formata dönüştür.
    Bu fonksiyonu modelinize göre uyarlayın.
    Örneğin 'Protocol' sütununu TCP/UDP/ICMP olarak sınıflandırıp one-hot encode edebilirsiniz.
    """
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].apply(lambda x: 'TCP' if 'TCP' in str(x) 
                                              else ('UDP' if 'UDP' in str(x) 
                                                    else ('ICMP' if 'ICMP' in str(x) else 'Other')))
        df = pd.get_dummies(df, columns=['Protocol'], drop_first=False)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    return df

def process_single_data(df):
    """
    Tekil JSON veriyi modelin beklediği formata dönüştür.
    """
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].apply(lambda x: 'TCP' if x == 'TCP'
                                              else ('UDP' if x == 'UDP'
                                                    else ('ICMP' if x == 'ICMP' else 'Other')))
        df = pd.get_dummies(df, columns=['Protocol'], drop_first=False)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    return df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' in request.files:
            # Dosya yüklendi, toplu tahmin
            file = request.files['file']
            df = pd.read_csv(file)
            processed_df = process_wireshark_data(df)
            
            predictions = model.predict(processed_df)
            prediction_probas = model.predict_proba(processed_df)

            # Dosya bazında özet:
            # Eğer herhangi bir satır saldırı ise 'Attack Detected', aksi halde 'No Attack Detected'
            if any(pred == 1 for pred in predictions):
                final_result = "Attack Detected"
            else:
                final_result = "No Attack Detected"
            
            # Ortalamaya dayalı bir genel olasılık
            avg_confidence = sum(max(proba) for proba in prediction_probas) / len(prediction_probas)
            final_prediction_value = f"{round(avg_confidence*100, 2)}%"

            # Tek bir özet sonuç döndürüyoruz
            return jsonify({
                "predictions": [
                    {
                        "result": final_result,
                        "predictionValue": final_prediction_value
                    }
                ]
            })

        else:
            # Tekil JSON veri
            data = request.get_json()
            df = pd.DataFrame([data])
            processed_df = process_single_data(df)

            prediction = model.predict(processed_df)[0]
            prediction_proba = model.predict_proba(processed_df)[0]

            result = "Attack Detected" if prediction == 1 else "No Attack Detected"
            prediction_value = f"{round(max(prediction_proba)*100, 2)}%"

            return jsonify({
                "predictions": [
                    {
                        "result": result,
                        "predictionValue": prediction_value
                    }
                ]
            })

    except Exception as e:
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
