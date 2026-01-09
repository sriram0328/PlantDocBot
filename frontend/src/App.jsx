import { useState } from "react";

export default function App() {
  const [image, setImage] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");

  const analyze = async () => {
    if (!image) return alert("Image required");

    const form = new FormData();
    form.append("image", image);
    if (text.trim()) form.append("text", text);

    setLoading(true);
    const res = await fetch("https://YOUR-BACKEND.onrender.com/predict", {
      method: "POST",
      body: form
    });

    setResult(await res.json());
    setLoading(false);
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMsg = chatInput;
    setMessages(m => [...m, { from: "user", text: userMsg }]);
    setChatInput("");

    const res = await fetch("https://YOUR-BACKEND.onrender.com/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disease: result.final_disease,
        message: userMsg
      })
    });

    const data = await res.json();
    setMessages(m => [...m, { from: "bot", text: data.reply }]);
  };

  if (result) {
    return (
      <div style={{ padding: 30 }}>
        <h2>Disease Detected</h2>
        <h3>{result.final_disease}</h3>

        <ul>
          {result.info.map((i, idx) => <li key={idx}>{i}</li>)}
        </ul>

        <p><i>{result.note}</i></p>

        <button onClick={() => setChatOpen(true)}>Chat for Guidance</button>
        <button onClick={() => setResult(null)}>New Scan</button>

        {chatOpen && (
          <div style={{ marginTop: 20 }}>
            <h4>Plant Care Assistant</h4>
            <div style={{ height: 200, overflowY: "auto" }}>
              {messages.map((m, i) => (
                <p key={i}><b>{m.from}:</b> {m.text}</p>
              ))}
            </div>

            <input
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder="Ask about treatment, symptoms, prevention..."
              style={{ width: "100%" }}
            />
            <button onClick={sendMessage}>Send</button>
            <button onClick={() => setChatOpen(false)}>Close</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ padding: 30 }}>
      <h2>PlantDoc</h2>

      <input type="file" accept="image/*"
        onChange={e => setImage(e.target.files[0])} />

      <br /><br />

      <textarea
        placeholder="Optional symptom description"
        rows={4}
        value={text}
        onChange={e => setText(e.target.value)}
      />

      <br /><br />

      <button onClick={analyze} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Plant"}
      </button>
    </div>
  );
}