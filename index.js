const gridElement = document.getElementById("grid");
let gridData = new Array(63).fill(0);
window.trainingData = null;
window.testFileData = null;
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GRID OLUŞTURMA
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for (let i = 0; i < 63; i++) {
    const cell = document.createElement("div");
    cell.classList.add("cell");
    cell.addEventListener("click", () => {
        cell.classList.toggle("active");
        gridData[i] = cell.classList.contains("active") ? 1 : 0;
        updateTextArea();
    });
    gridElement.appendChild(cell);
}

function updateTextArea() {
    let text = "";
    for (let i = 0; i < 63; i++) {
        text += gridData[i] === 1 ? "# " : ". ";
        if ((i + 1) % 7 === 0) text += "\n";
    }
    document.getElementById("textInput").value = text;
}

function applyText() {
    const text = document.getElementById("textInput").value;
    const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    const cells = document.querySelectorAll("#grid .cell");
    let index = 0;
    for (let r = 0; r < 9; r++) {
        const line = (lines[r] || "").replace(/\s/g, "");
        for (let c = 0; c < 7; c++) {
            if (index < 63) {
                if (line[c] === "#") {
                    cells[index].classList.add("active"); gridData[index] = 1;
                } else {
                    cells[index].classList.remove("active"); gridData[index] = 0;
                }
                index++;
            }
        }
    }
}

function clearGrid() {
    document.querySelectorAll("#grid .cell").forEach((cell, i) => {
        cell.classList.remove("active"); gridData[i] = 0;
    });
    updateTextArea();
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// FILE READERS
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
document.getElementById("fileInput").addEventListener("change", (e) => {
    const reader = new FileReader();
    reader.onload = (ev) => { document.getElementById("textInput").value = ev.target.result; applyText(); };
    reader.readAsText(e.target.files[0]);
});

document.getElementById("trainFileInput").addEventListener("change", (e) => {
    const reader = new FileReader();
    reader.onload = (ev) => { window.trainingData = ev.target.result; renderPreview("trainingPreview", ev.target.result); };
    reader.readAsText(e.target.files[0]);
});

document.getElementById("testFileInput").addEventListener("change", (e) => {
    const reader = new FileReader();
    reader.onload = (ev) => { window.testFileData = ev.target.result; renderPreview("testPreview", ev.target.result); };
    reader.readAsText(e.target.files[0]);
});

function renderPreview(elementId, content) {
    const container = document.getElementById(elementId);
    container.innerHTML = "";
    const lines = content.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    for (let i = 0; i < lines.length; i++) {
        if (i % 10 === 0) continue;
        const line = lines[i].replace(/\s/g, "");
        for (let c = 0; c < 7; c++) {
            const cell = document.createElement("div");
            cell.classList.add("cell");
            if (line[c] === "#") cell.classList.add("active");
            container.appendChild(cell);
        }
    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// API BAĞLANTILARI
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 1. Modeli Eğitme (POST /train)
async function trainModel() {
    if (!window.trainingData) return alert("Please load Training Set first!");
    document.getElementById("result").innerText = "Training models, please wait...";
    
    try {
        const res = await fetch("http://127.0.0.1:5000/train", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ trainingData: window.trainingData })
        });
        const data = await res.json();
        document.getElementById("result").innerText = `Training Complete! (${data.samples || 21} samples)`;
    } catch (err) {
        document.getElementById("result").innerText = "Error connecting to server.";
    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 2. Modeli Test Etme (POST /test_batch)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
async function testModel() {
    if (!window.testFileData) return alert("Please load Test Set first!");
    document.getElementById("result").innerText = "Testing model, please wait...";
    
    try {
        const res = await fetch("http://127.0.0.1:5000/test_batch", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ testData: window.testFileData })
        });
        const data = await res.json();
        if (data.error) return alert(data.error);
        
        // Backend'den gelen nesneleri yakala (perceptron/multilayer veya p/m)
        const pData = data.perceptron || data.p;
        const mData = data.multilayer || data.m;

        // Metrik kutularını güncelle
        updateMetricsDisplay("p_metrics", pData);
        updateMetricsDisplay("m_metrics", mData);
        document.getElementById("result").innerText = "Test Completed. Metrics Updated!";
    } catch (err) {
        alert("Failed to calculate metrics. Check backend connection.");
    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 3. Çizimi Tahmin Et (POST /predict)~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
async function predictGrid() {
    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ grid: gridData })
        });
        const data = await res.json();
        if (data.error) return alert(data.error);
        
        // 1. Tahmin harflerini üstteki alanlara yerleştir
        // Backend'den p_guess/m_guess veya perceptron/multilayer gelme durumuna göre esneklik sağlandı
        document.getElementById("p_guess").innerText = data.perceptron_guess || data.perceptron || data.p || "-";
        document.getElementById("m_guess").innerText = data.multilayer_guess || data.multilayer || data.m || "-";
        
        // 2. Eğer /predict rotası yanıtın içinde metrikleri de gönderiyorsa (3. görseldeki gibi) alt kutuları da doldurur
        const pData = data.perceptron_metrics || data.perceptron || data.p;
        const mData = data.multilayer_metrics || data.multilayer || data.m;
        
        if (pData && (pData.accuracy !== undefined || pData.acc !== undefined)) {
            updateMetricsDisplay("p_metrics", pData);
        }
        if (mData && (mData.accuracy !== undefined || mData.acc !== undefined)) {
            updateMetricsDisplay("m_metrics", mData);
        }

        document.getElementById("result").innerText = "Prediction Completed!";
    } catch (err) {
        alert("Prediction failed. Ensure the server is running.");
    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


// Esnek ve Güvenli Metrik Yazdırma Fonksiyonu 
function updateMetricsDisplay(id, metrics) {
    if (!metrics || typeof metrics !== 'object') return;
    
    // Değişken isimlerinin backend ile uyuşması için alternatifli okuma (accuracy veya acc)
    let accVal = metrics.accuracy !== undefined ? metrics.accuracy : (metrics.acc !== undefined ? metrics.acc : null);
    let preVal = metrics.precision !== undefined ? metrics.precision : (metrics.pre !== undefined ? metrics.pre : null);
    let recVal = metrics.recall !== undefined ? metrics.recall : (metrics.rec !== undefined ? metrics.rec : null);
    let f1Val = metrics.f1 !== undefined ? metrics.f1 : null;

    let htmlContent = "";
    
    if (accVal !== null) htmlContent += `Acc: ${accVal.toFixed(3)}`;
    if (preVal !== null) htmlContent += `<br>Pre: ${preVal.toFixed(3)}`;
    if (recVal !== null) htmlContent += `<br>Rec: ${recVal.toFixed(3)}`;
    if (f1Val !== null) htmlContent += `<br>F1: ${f1Val.toFixed(3)}`;
    
    if (htmlContent !== "") {
        document.getElementById(id).innerHTML = htmlContent;
    }
}

function clearTrainingPreview() { window.trainingData = null; document.getElementById("trainingPreview").innerHTML = ""; }
function clearTestPreview() { window.testFileData = null; document.getElementById("testPreview").innerHTML = ""; }