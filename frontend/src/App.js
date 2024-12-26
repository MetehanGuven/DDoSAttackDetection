import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [filePredictions, setFilePredictions] = useState([]);
  const [fileLoading, setFileLoading] = useState(false);

  const [byteperflow, setByteperflow] = useState("");
  const [bytecount, setBytecount] = useState("");
  const [pktcount, setPktcount] = useState("");
  const [protocol, setProtocol] = useState("UDP");
  const [singlePredictions, setSinglePredictions] = useState([]);
  const [singleLoading, setSingleLoading] = useState(false);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleFileSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please upload a file.");
    setFileLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setFilePredictions(data.predictions || []);
    } catch {
      alert("Error connecting to the server.");
      setFilePredictions([]);
    } finally {
      setFileLoading(false);
    }
  };

  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    setSingleLoading(true);
    const payload = {
      byteperflow: parseFloat(byteperflow),
      bytecount: parseFloat(bytecount),
      pktcount: parseFloat(pktcount),
      Protocol: protocol,
    };

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      setSinglePredictions(data.predictions || []);
    } catch {
      alert("Error connecting to the server.");
      setSinglePredictions([]);
    } finally {
      setSingleLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="title">DDoS Attack Detection</h1>

      <div className="form-wrapper">
        <div className="form-card">
          <h2>Upload Wireshark CSV</h2>
          <form onSubmit={handleFileSubmit}>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="file-input"
              required
            />
            <button
              type="submit"
              className="btn primary"
              disabled={fileLoading}
            >
              {fileLoading ? "Predicting..." : "Upload & Predict"}
            </button>
          </form>

          {filePredictions.length > 0 && (
            <div className="results">
              <h3>Batch Predictions:</h3>
              {filePredictions.map((res, idx) => (
                <div key={idx} className="result">
                  <p>
                    <strong>Result:</strong> {res.result}
                  </p>
                  <p>
                    <strong>Prediction Value:</strong> {res.predictionValue}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="form-card">
          <h2>Single Data Prediction</h2>
          <form onSubmit={handleSingleSubmit}>
            <label>Byte per Flow:</label>
            <input
              type="number"
              step="any"
              value={byteperflow}
              onChange={(e) => setByteperflow(e.target.value)}
              required
            />
            <label>Byte Count:</label>
            <input
              type="number"
              step="any"
              value={bytecount}
              onChange={(e) => setBytecount(e.target.value)}
              required
            />
            <label>Packet Count:</label>
            <input
              type="number"
              step="any"
              value={pktcount}
              onChange={(e) => setPktcount(e.target.value)}
              required
            />
            <label>Protocol:</label>
            <select
              value={protocol}
              onChange={(e) => setProtocol(e.target.value)}
              required
            >
              <option value="UDP">UDP</option>
              <option value="TCP">TCP</option>
              <option value="ICMP">ICMP</option>
            </select>
            <button
              type="submit"
              className="btn secondary"
              disabled={singleLoading}
            >
              {singleLoading ? "Predicting..." : "Predict Single"}
            </button>
          </form>

          {singlePredictions.length > 0 && (
            <div className="results">
              <h3>Single Prediction Result:</h3>
              {singlePredictions.map((res, idx) => (
                <div key={idx} className="result">
                  <p>
                    <strong>Result:</strong> {res.result}
                  </p>
                  <p>
                    <strong>Prediction Value:</strong> {res.predictionValue}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
