import { useState } from "react";

export default function App() {
  const [image, setImage] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyze = async () => {
    if (!image) return alert("Image is required");

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

  if (result) {
    return (
      <div style={{ padding: 30 }}>
        <h2>Disease Detected</h2>
        <h3>{result.final_disease}</h3>

        <ul>
          {result.info.map((i, idx) => (
            <li key={idx}>{i}</li>
          ))}
        </ul>

        <p><i>{result.note}</i></p>
        <button onClick={() => setResult(null)}>New Scan</button>
      </div>
    );
  }

  return (
    <div style={{ padding: 30 }}>
      <h2>PlantDoc</h2>

      <input
        type="file"
        accept="image/*"
        onChange={e => setImage(e.target.files[0])}
      />

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