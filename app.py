#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ACIKLAMALAR - EXPLANATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from flask import Flask, request, jsonify
from flask_cors import CORS
# Sadece üst seviye fonksiyonları çağırıyoruz
from ann import train_all_models, test_all_models, predict_single_vector

app = Flask(__name__)
CORS(app)

#//////////////////////////////////////////\
@app.route('/train', methods=['POST'])
def train():
    data = request.json
    raw_text = data.get("trainingData", "")
    
    # Tüm işi ann.py içindeki bu fonksiyon üstleniyor
    success, sample_count = train_all_models(raw_text)
    
    if not success:
        return jsonify({"error": "Failed to parse training data or data is empty"}), 400
        
    return jsonify({"status": "trained", "samples": sample_count})
#//////////////////////////////////////////\

#//////////////////////////////////////////\
@app.route('/test_batch', methods=['POST'])
def test_batch():
    data = request.json
    raw_text = data.get("testData", "")
    
    # ann.py gidip testleri yapar ve sonucu sözlük olarak döner
    metrics = test_all_models(raw_text)
    
    if metrics is None:
        return jsonify({"error": "Models are not trained yet!"}), 400
        
    return jsonify({
        "perceptron": metrics["p"],
        "multilayer": metrics["m"],
        "p": metrics["p"],
        "m": metrics["m"]
    })
#//////////////////////////////////////////\

#//////////////////////////////////////////\
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    grid_vector = data.get("grid", [])
    
    if not grid_vector or len(grid_vector) != 63:
        return jsonify({"error": "Invalid grid data"}), 400
        
    # Tekil tahmin işi de tamamen delege edildi
    predictions = predict_single_vector(grid_vector)
    
    if predictions is None:
        return jsonify({"error": "Models are not trained yet!"}), 400
        
    return jsonify({
        "perceptron": predictions["p"],
        "multilayer": predictions["m"],
        "p": predictions["p"],
        "m": predictions["m"]
    })
#//////////////////////////////////////////\

if __name__ == '__main__':
    app.run(debug=True, port=5000)