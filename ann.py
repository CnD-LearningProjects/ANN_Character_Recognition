import numpy as np

# //////////////////////////////////////////\
# CLASS LABELS
# //////////////////////////////////////////\
CLASS_LABELS = ['A','B','C','D','E','J','K']

# Global model nesneleri (Eğitildikten sonra hafızada yaşayacaklar)
perceptron_model = None
mlp_model = None

# //////////////////////////////////////////\
# DATA LOADER
# //////////////////////////////////////////\
def load_training_data(raw_text):
    if not raw_text:
        return np.array([]), []

    lines = []
    for line in raw_text.split("\n"):
        cleaned_line = line.strip()
        if cleaned_line != "":
            lines.append(cleaned_line)

    input_vectors = []
    target_labels = []

    i = 0
    while i < len(lines):
        label = lines[i]
        matrix_lines = []

        for j in range(i+1, i+10):
            if j < len(lines):
                matrix_lines.append(lines[j])

        input_vector = []
        for row in matrix_lines:
            for char in row:
                if char == '#':
                    input_vector.append(1)
                elif char == '.':
                    input_vector.append(-1)

        if len(input_vector) == 63:
            input_vectors.append(np.array(input_vector, dtype=np.float64))
            target_labels.append(label)
            i += 10
        else:
            i += 1

    return input_vectors, target_labels

# //////////////////////////////////////////\
# TARGET VECTORS
# //////////////////////////////////////////\
def create_target_vectors(target_labels):
    target_vectors = []
    for label in target_labels:
        vector = []
        for class_label in CLASS_LABELS:
            if class_label == label:
                vector.append(1)
            else:
                vector.append(-1)
        target_vectors.append(vector)
    return np.array(target_vectors)

# //////////////////////////////////////////\
# PERCEPTRON CLASS
# //////////////////////////////////////////\
class Perceptron:
    def __init__(self, input_size=63, output_size=7, learning_rate=0.1):
        self.weights = np.zeros((output_size, input_size))
        self.bias_vector = np.zeros(output_size)
        self.learning_rate = learning_rate

    def predict_class(self, input_vector):
        net_input = np.dot(self.weights, input_vector) + self.bias_vector
        return np.argmax(net_input)

    def train(self, input_vectors, target_vectors, epochs=100):
        for a in range(epochs):
            error_count = 0
            for input_vector, target_vector in zip(input_vectors, target_vectors):
                net_input = np.dot(self.weights, input_vector) + self.bias_vector
                predicted = []
                for b in net_input:
                    predicted.append(1 if b >= 0 else -1)

                for i in range(len(target_vector)):
                    if predicted[i] != target_vector[i]:
                        self.weights[i] += self.learning_rate * target_vector[i] * input_vector
                        self.bias_vector[i] += self.learning_rate * target_vector[i]
                        error_count += 1
            if error_count == 0:
                break

# //////////////////////////////////////////\
# MLP CLASS
# //////////////////////////////////////////\
class MLP:
    def __init__(self, input_size=63, hidden_size=32, output_size=7, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.1
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.1
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate

    def forward(self, input_vector):
        self.hidden_layer_input = np.dot(self.weights_input_hidden, input_vector) + self.bias_hidden
        self.hidden_layer_output = np.tanh(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.weights_hidden_output, self.hidden_layer_output) + self.bias_output
        self.final_output = np.tanh(self.output_layer_input)
        return self.final_output

    def predict_class(self, input_vector):
        output = self.forward(input_vector)
        return np.argmax(output)

    def train(self, input_vectors, target_vectors, epochs=1000):
        X = np.array(input_vectors)
        y = np.array(target_vectors)
        num_samples = len(X)
        
        for i in range(epochs):
            # Modelin harf sırasını ezberlememesi için karıştırıyoruz
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for j in indices:
                input_vector = X[j]
                target_vector = y[j]
                
                # İleri besleme
                output = self.forward(input_vector)
                
                # Hata hesaplamaları
                output_error = (target_vector - output) * (1 - output ** 2)
                hidden_error = np.dot(self.weights_hidden_output.T, output_error) * (1 - self.hidden_layer_output ** 2)

                # Ağırlık ve Bias güncelleme
                self.weights_hidden_output += self.learning_rate * np.outer(output_error, self.hidden_layer_output)
                self.bias_output += self.learning_rate * output_error
                self.weights_input_hidden += self.learning_rate * np.outer(hidden_error, input_vector)
                self.bias_hidden += self.learning_rate * hidden_error

# //////////////////////////////////////////\
# METRICS & CORE OPERATIONS
# //////////////////////////////////////////\
def calculate_metrics(model, input_vectors, true_labels):
    if len(input_vectors) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    predicted_labels = []
    for input_vector in input_vectors:
        predicted_index = model.predict_class(input_vector)
        predicted_labels.append(CLASS_LABELS[predicted_index])

    correct = 0
    for t, p in zip(true_labels, predicted_labels):
        if t == p:
            correct += 1
    accuracy = correct / len(true_labels)

    precisions = []
    recalls = []

    for c in CLASS_LABELS:
        tp = 0  
        fp = 0  
        fn = 0  
        
        for t, p in zip(true_labels, predicted_labels):
            if p == c and t == c: tp += 1
            if p == c and t != c: fp += 1
            if p != c and t == c: fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)

    if (macro_precision + macro_recall) > 0:
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    else:
        macro_f1 = 0

    return {
        "accuracy": float(accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1)
    }

# //////////////////////////////////////////\
# APP.PY İÇİN ÜST SEVİYE YÖNETİM FONKSİYONLARI
# //////////////////////////////////////////\

def train_all_models(raw_text):
    """app.py tarafından çağrılır. Modelleri sıfırdan kurar ve eğitir."""
    global perceptron_model, mlp_model
    input_vectors, target_labels = load_training_data(raw_text)
    
    if len(input_vectors) == 0:
        return False, 0
        
    target_vectors = create_target_vectors(target_labels)
    
    perceptron_model = Perceptron(input_size=63, output_size=len(CLASS_LABELS))
    mlp_model = MLP(input_size=63, hidden_size=32, output_size=len(CLASS_LABELS))
    
    perceptron_model.train(input_vectors, target_vectors, epochs=100)
    mlp_model.train(input_vectors, target_vectors, epochs=1000) # MLP eğitimi 1000 epoch'a çekildi
    
    return True, len(input_vectors)

def test_all_models(raw_text):
    """app.py tarafından çağrılır. Test verisiyle metrikleri hesaplar."""
    global perceptron_model, mlp_model
    if perceptron_model is None or mlp_model is None:
        return None
        
    input_vectors, true_labels = load_training_data(raw_text)
    p_metrics = calculate_metrics(perceptron_model, input_vectors, true_labels)
    m_metrics = calculate_metrics(mlp_model, input_vectors, true_labels)
    
    return {"p": p_metrics, "m": m_metrics}

def predict_single_vector(grid_vector):
    """app.py tarafından çağrılır. Çizim alanından gelen anlık veriyi tahmin eder."""
    global perceptron_model, mlp_model
    if perceptron_model is None or mlp_model is None:
        return None
        
    processed_vector = np.array([1.0 if x == 1 else -1.0 for x in grid_vector], dtype=np.float64)
    p_idx = perceptron_model.predict_class(processed_vector)
    m_idx = mlp_model.predict_class(processed_vector)
    
    return {"p": CLASS_LABELS[p_idx], "m": CLASS_LABELS[m_idx]}