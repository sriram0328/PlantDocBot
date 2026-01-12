import React, { useState, useEffect, useRef } from "react";
import "./index.css";

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const chatEndRef = useRef(null);

  const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const analyze = async () => {
    if (!image) {
      alert("Please select an image first");
      return;
    }

    setLoading(true);
    const form = new FormData();
    form.append("image", image);
    if (text.trim()) form.append("text", text);

    try {
      const res = await fetch(`${BACKEND}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || "Server error");
      }

      const data = await res.json();
      setResult(data);
      setMessages([]);
      setChatOpen(false);
    } catch (error) {
      console.error("Analysis error:", error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    const trimmedInput = chatInput.trim();
    if (!trimmedInput) return;
    if (!result?.raw_label) {
      alert("No diagnosis available for chat");
      return;
    }

    setMessages((prev) => [...prev, { from: "You", text: trimmedInput }]);
    setChatInput("");

    try {
      const res = await fetch(`${BACKEND}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease: result.raw_label,
          message: trimmedInput,
        }),
      });

      if (!res.ok) throw new Error("Chat request failed");

      const data = await res.json();
      setMessages((prev) => [...prev, { from: "Assistant", text: data.reply }]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { from: "Assistant", text: "An error occurred. Please try again." },
      ]);
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setImage(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setText("");
    setMessages([]);
    setChatOpen(false);
  };

  return (
    <div className="container">
      <h1>AI PlantDocBot</h1>
      <p className="subtitle">AI-Based Plant Disease Detection</p>

      {!result ? (
        <div className="card">
          <div className="upload-section">
            <label htmlFor="file-input" className="file-label">
              Upload Plant Image
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="file-input"
            />
          </div>

          {preview && (
            <div className="preview-section">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          <textarea
            placeholder="Optional: Describe visible symptoms (e.g., yellow spots, wilting leaves)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            className="symptom-input"
          />

          <button
            onClick={analyze}
            disabled={loading || !image}
            className="btn-primary"
          >
            {loading ? "Analyzing..." : "Identify Disease"}
          </button>

          {text.trim() && (
            <p className="info-text">
              Symptom description will be used to validate the diagnosis
            </p>
          )}
        </div>
      ) : (
        <div className="card result-card">
          <div className="result-header">
            <h2 className="disease-name">{result.final_disease}</h2>
            {result.confidence && (
              <span className="confidence-badge">
                {(result.confidence * 100).toFixed(1)}% confidence
              </span>
            )}
          </div>

          <p className={`diagnosis-note ${result.text_used ? "validated" : ""}`}>
            {result.note}
          </p>

          <div className="info-section">
  <h3>Disease Description</h3>
  <p className="disease-description">
    {result.description}
  </p>
</div>



          <div className="action-buttons">
            <button
              onClick={() => setChatOpen(!chatOpen)}
              className="btn-secondary"
            >
              {chatOpen ? "Hide Chat" : "Ask Questions"}
            </button>

            <button onClick={resetAnalysis} className="btn-primary">
              New Analysis
            </button>
          </div>

          {chatOpen && (
            <div className="chat-container">
              <h3>Disease-Specific Questions</h3>
              <div className="messages">
                {messages.length === 0 ? (
                  <p className="chat-placeholder">
                    Ask about treatment, prevention, causes, or spread.
                  </p>
                ) : (
                  messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`message ${
                        msg.from === "You" ? "user" : "assistant"
                      }`}
                    >
                      <strong>{msg.from}:</strong> {msg.text}
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="chat-input-container">
                <input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                  placeholder="Type your question"
                  className="chat-input"
                />
                <button
                  onClick={sendMessage}
                  disabled={!chatInput.trim()}
                  className="btn-send"
                >
                  Send
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
