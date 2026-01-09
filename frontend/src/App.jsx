import { useState } from "react";

const styles = {
  gradient: {
    background: 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #ccfbf1 100%)',
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px'
  },
  card: {
    background: 'white',
    borderRadius: '24px',
    boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
    padding: '40px',
    maxWidth: '600px',
    width: '100%',
    margin: '0 auto'
  },
  button: {
    background: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
    color: 'white',
    border: 'none',
    padding: '16px 32px',
    borderRadius: '16px',
    fontSize: '18px',
    fontWeight: '600',
    cursor: 'pointer',
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)'
  },
  uploadZone: {
    border: '3px dashed #6ee7b7',
    borderRadius: '16px',
    padding: '60px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    marginBottom: '24px'
  }
};

// ---------------- UPLOAD ----------------
function Upload({ onResult }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const analyze = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
      });

      const result = await res.json();

      onResult({
        disease: result.final_disease,
        note: result.note,
        image: preview.split(",")[1]
      });
    } catch {
      alert("Backend not reachable");
    }

    setLoading(false);
  };

  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: "700px", width: "100%" }}>
        <div style={{ textAlign: "center", marginBottom: "40px" }}>
          <h1 style={{ fontSize: "56px", fontWeight: "bold", color: "#059669" }}>
            🌿 PlantDoc
          </h1>
          <p style={{ color: "#6b7280" }}>
            AI-Powered Plant Disease Detection
          </p>
        </div>

        <div style={styles.card}>
          {preview ? (
            <img
              src={preview}
              alt="preview"
              style={{ width: "100%", borderRadius: "16px", marginBottom: "24px" }}
            />
          ) : (
            <label>
              <div style={styles.uploadZone}>
                <p>Click to upload plant image</p>
              </div>
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
            </label>
          )}

          <button
            onClick={analyze}
            disabled={!image || loading}
            style={{ ...styles.button, opacity: loading ? 0.6 : 1 }}
          >
            {loading ? "Analyzing..." : "Analyze Plant"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------- RESULT ----------------
function Result({ data, onBack, onChat }) {
  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: "700px", width: "100%" }}>
        <button onClick={onBack} style={{ marginBottom: "20px" }}>← Back</button>

        <div style={styles.card}>
          <img
            src={`data:image/png;base64,${data.image}`}
            alt="plant"
            style={{ width: "100%", borderRadius: "16px", marginBottom: "24px" }}
          />

          <h2 style={{ fontSize: "32px", marginBottom: "12px" }}>
            {data.disease}
          </h2>

          <p style={{ color: "#374151", marginBottom: "24px" }}>
            {data.note}
          </p>

          <button style={styles.button} onClick={onChat}>
            Chat for General Advice
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------- CHAT ----------------
function Chat({ onBack }) {
  const [messages, setMessages] = useState([
    {
      from: "bot",
      text:
        "I can help with general plant care questions. Diagnosis comes from the scan."
    }
  ]);
  const [msg, setMsg] = useState("");

  const send = () => {
    if (!msg.trim()) return;
    setMessages(m => [...m, { from: "user", text: msg }]);
    setMsg("");
    setMessages(m => [...m, { from: "bot", text: "Please follow standard plant care practices 🌱" }]);
  };

  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: "700px", width: "100%" }}>
        <button onClick={onBack}>← Back</button>

        <div style={styles.card}>
          <div style={{ minHeight: "300px", marginBottom: "20px" }}>
            {messages.map((m, i) => (
              <p key={i}><b>{m.from}:</b> {m.text}</p>
            ))}
          </div>

          <input
            value={msg}
            onChange={e => setMsg(e.target.value)}
            placeholder="Ask plant care question"
            style={{ width: "100%", padding: "12px" }}
          />
          <button style={styles.button} onClick={send}>Send</button>
        </div>
      </div>
    </div>
  );
}

// ---------------- MAIN ----------------
export default function App() {
  const [page, setPage] = useState("upload");
  const [data, setData] = useState(null);

  if (page === "upload")
    return <Upload onResult={(d) => { setData(d); setPage("result"); }} />;

  if (page === "result")
    return <Result data={data} onBack={() => setPage("upload")} onChat={() => setPage("chat")} />;

  return <Chat onBack={() => setPage("result")} />;
}