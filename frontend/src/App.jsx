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
  },
  featureCard: {
    background: 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    padding: '20px',
    textAlign: 'center',
    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
    transition: 'all 0.3s ease'
  }
};

// Upload Component
function Upload({ onResult }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hoverUpload, setHoverUpload] = useState(false);
  const [hoverButton, setHoverButton] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const analyze = async () => {
    setLoading(true);
    setTimeout(() => {
      onResult({
        disease: "Tomato Early Blight",
        info: "Early blight is a common fungal disease affecting tomato plants. It causes dark spots on leaves and can reduce yield. Treatment includes removing affected leaves and applying fungicide.",
        image: preview.split(',')[1]
      });
      setLoading(false);
    }, 1500);
  };

  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: '700px', width: '100%' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '80px',
            height: '80px',
            background: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
            borderRadius: '24px',
            marginBottom: '20px',
            boxShadow: '0 8px 24px rgba(16, 185, 129, 0.3)',
            fontSize: '40px'
          }}>
            🌿
          </div>
          <h1 style={{
            fontSize: '56px',
            fontWeight: 'bold',
            background: 'linear-gradient(135deg, #059669 0%, #0d9488 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '8px'
          }}>
            PlantDoc
          </h1>
          <p style={{ color: '#6b7280', fontSize: '18px' }}>AI-Powered Plant Disease Detection</p>
        </div>

        {/* Upload Card */}
        <div style={styles.card}>
          <div style={{ textAlign: 'center', marginBottom: '30px' }}>
            <h2 style={{ fontSize: '28px', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>Upload Plant Image</h2>
            <p style={{ color: '#6b7280' }}>Get instant disease diagnosis and treatment recommendations</p>
          </div>

          {preview ? (
            <div style={{ marginBottom: '24px', position: 'relative' }}>
              <img 
                src={preview} 
                alt="Preview" 
                style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: '16px', boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}
              />
              <button
                onClick={() => { setPreview(null); setImage(null); }}
                style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  background: '#ef4444',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '20px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)'
                }}
              >
                ✕ Remove
              </button>
            </div>
          ) : (
            <label style={{ display: 'block', marginBottom: '24px', cursor: 'pointer' }}>
              <div 
                style={{
                  ...styles.uploadZone,
                  borderColor: hoverUpload ? '#34d399' : '#6ee7b7',
                  background: hoverUpload ? '#ecfdf5' : 'transparent'
                }}
                onMouseEnter={() => setHoverUpload(true)}
                onMouseLeave={() => setHoverUpload(false)}
              >
                <div style={{ fontSize: '64px', marginBottom: '16px', transform: hoverUpload ? 'scale(1.1)' : 'scale(1)', transition: 'transform 0.3s' }}>📷</div>
                <p style={{ color: '#4b5563', fontWeight: '600', marginBottom: '8px' }}>Click to upload or drag and drop</p>
                <p style={{ color: '#9ca3af', fontSize: '14px' }}>PNG, JPG up to 10MB</p>
              </div>
              <input 
                type="file" 
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
              />
            </label>
          )}

          <button
            onClick={analyze}
            disabled={!image || loading}
            style={{
              ...styles.button,
              opacity: (!image || loading) ? 0.5 : 1,
              cursor: (!image || loading) ? 'not-allowed' : 'pointer',
              transform: hoverButton && image && !loading ? 'translateY(-2px)' : 'translateY(0)',
              boxShadow: hoverButton && image && !loading ? '0 8px 24px rgba(16, 185, 129, 0.4)' : '0 4px 12px rgba(16, 185, 129, 0.3)'
            }}
            onMouseEnter={() => setHoverButton(true)}
            onMouseLeave={() => setHoverButton(false)}
          >
            {loading ? (
              <>
                <div style={{
                  width: '24px',
                  height: '24px',
                  border: '3px solid white',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
                Analyzing...
              </>
            ) : (
              <>
                <span style={{ fontSize: '20px' }}>✨</span>
                Analyze Plant
              </>
            )}
          </button>
        </div>

        {/* Features */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginTop: '32px' }}>
          {[
            { icon: "🎯", text: "Accurate Detection" },
            { icon: "⚡", text: "Instant Results" },
            { icon: "🌱", text: "Treatment Tips" }
          ].map((feature, i) => (
            <div key={i} style={styles.featureCard}>
              <div style={{ fontSize: '32px', marginBottom: '8px' }}>{feature.icon}</div>
              <p style={{ fontSize: '14px', color: '#374151', fontWeight: '600' }}>{feature.text}</p>
            </div>
          ))}
        </div>
      </div>
      
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// Result Component
function Result({ data, onChat, onBack }) {
  const [hoverChat, setHoverChat] = useState(false);
  const [hoverNew, setHoverNew] = useState(false);

  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: '700px', width: '100%' }}>
        <button
          onClick={onBack}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'none',
            border: 'none',
            color: '#6b7280',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            marginBottom: '24px',
            transition: 'color 0.3s'
          }}
        >
          <span style={{ fontSize: '20px' }}>←</span>
          Back to Upload
        </button>

        <div style={styles.card}>
          <div style={{ position: 'relative', height: '350px', marginBottom: '32px', borderRadius: '16px', overflow: 'hidden' }}>
            <img 
              src={`data:image/png;base64,${data.image}`}
              alt="Plant"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            <div style={{
              position: 'absolute',
              inset: 0,
              background: 'linear-gradient(to top, rgba(0,0,0,0.5), transparent)'
            }}></div>
          </div>

          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: '#fee2e2',
            color: '#b91c1c',
            padding: '8px 16px',
            borderRadius: '20px',
            fontSize: '14px',
            fontWeight: '600',
            marginBottom: '16px'
          }}>
            <div style={{ width: '8px', height: '8px', background: '#ef4444', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
            Disease Detected
          </div>

          <h2 style={{ fontSize: '32px', fontWeight: 'bold', color: '#1f2937', marginBottom: '16px' }}>
            {data.disease}
          </h2>

          <div style={{
            background: 'linear-gradient(135deg, #ecfdf5 0%, #ccfbf1 100%)',
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '24px',
            borderLeft: '4px solid #10b981'
          }}>
            <h3 style={{ fontWeight: '600', color: '#1f2937', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>💡</span>
              Treatment Recommendation
            </h3>
            <p style={{ color: '#374151', lineHeight: '1.6' }}>{data.info}</p>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              onClick={onChat}
              style={{
                ...styles.button,
                flex: 1,
                transform: hoverChat ? 'translateY(-2px)' : 'translateY(0)',
                boxShadow: hoverChat ? '0 8px 24px rgba(16, 185, 129, 0.4)' : '0 4px 12px rgba(16, 185, 129, 0.3)'
              }}
              onMouseEnter={() => setHoverChat(true)}
              onMouseLeave={() => setHoverChat(false)}
            >
              <span style={{ fontSize: '20px' }}>💬</span>
              Chat with Expert Bot
            </button>
            <button
              onClick={onBack}
              style={{
                padding: '16px 24px',
                border: '2px solid #d1d5db',
                background: 'white',
                color: '#374151',
                borderRadius: '16px',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s',
                borderColor: hoverNew ? '#10b981' : '#d1d5db',
                color: hoverNew ? '#10b981' : '#374151'
              }}
              onMouseEnter={() => setHoverNew(true)}
              onMouseLeave={() => setHoverNew(false)}
            >
              🔄 New Scan
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginTop: '24px' }}>
          {[
            { label: "Confidence", value: "94%", icon: "📊" },
            { label: "Severity", value: "Medium", icon: "⚠️" },
            { label: "Action", value: "Required", icon: "✅" }
          ].map((stat, i) => (
            <div key={i} style={styles.featureCard}>
              <div style={{ fontSize: '24px', marginBottom: '4px' }}>{stat.icon}</div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#059669', marginBottom: '4px' }}>{stat.value}</div>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
      
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

// Chat Component
function Chat({ onBack }) {
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hello! 👋 I'm here to help you with plant care questions. What would you like to know?" }
  ]);
  const [msg, setMsg] = useState("");
  const [hoverSend, setHoverSend] = useState(false);

  const send = async () => {
    if (!msg.trim()) return;

    setMessages(m => [...m, { from: "user", text: msg }]);
    const userMsg = msg;
    setMsg("");

    setTimeout(() => {
      setMessages(m => [...m, { 
        from: "bot", 
        text: `I understand you're asking about "${userMsg}". Based on the diagnosis, I recommend regular monitoring and applying appropriate treatments. Would you like more specific advice? 🌱` 
      }]);
    }, 1000);
  };

  return (
    <div style={styles.gradient}>
      <div style={{ maxWidth: '900px', width: '100%', height: '90vh', display: 'flex', flexDirection: 'column' }}>
        <div style={{
          background: 'white',
          borderTopLeftRadius: '24px',
          borderTopRightRadius: '24px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          padding: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              background: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px'
            }}>
              🤖
            </div>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: '#1f2937', marginBottom: '4px' }}>Plant Expert Bot</h2>
              <p style={{ fontSize: '14px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{ width: '8px', height: '8px', background: '#10b981', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
                Online
              </p>
            </div>
          </div>
          <button
            onClick={onBack}
            style={{
              background: 'none',
              border: 'none',
              color: '#6b7280',
              fontSize: '24px',
              cursor: 'pointer',
              transition: 'color 0.3s'
            }}
          >
            ←
          </button>
        </div>

        <div style={{
          flex: 1,
          background: 'white',
          padding: '24px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                justifyContent: m.from === "user" ? "flex-end" : "flex-start"
              }}
            >
              <div
                style={{
                  maxWidth: '70%',
                  padding: '16px',
                  borderRadius: '16px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  background: m.from === "user" ? 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)' : '#f3f4f6',
                  color: m.from === "user" ? 'white' : '#1f2937',
                  borderBottomRightRadius: m.from === "user" ? '4px' : '16px',
                  borderBottomLeftRadius: m.from === "user" ? '16px' : '4px'
                }}
              >
                <p style={{ lineHeight: '1.6', margin: 0 }}>{m.text}</p>
              </div>
            </div>
          ))}
        </div>

        <div style={{
          background: 'white',
          borderBottomLeftRadius: '24px',
          borderBottomRightRadius: '24px',
          boxShadow: '0 -4px 12px rgba(0,0,0,0.08)',
          padding: '24px'
        }}>
          <div style={{ display: 'flex', gap: '12px' }}>
            <input
              value={msg}
              onChange={e => setMsg(e.target.value)}
              onKeyPress={e => e.key === "Enter" && send()}
              placeholder="Ask me anything about plant care... 🌿"
              style={{
                flex: 1,
                padding: '16px 24px',
                background: '#f3f4f6',
                border: 'none',
                borderRadius: '16px',
                fontSize: '16px',
                color: '#1f2937',
                outline: 'none'
              }}
            />
            <button
              onClick={send}
              disabled={!msg.trim()}
              style={{
                background: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
                color: 'white',
                border: 'none',
                padding: '16px 32px',
                borderRadius: '16px',
                fontSize: '18px',
                fontWeight: '600',
                cursor: msg.trim() ? 'pointer' : 'not-allowed',
                opacity: msg.trim() ? 1 : 0.5,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s',
                transform: hoverSend && msg.trim() ? 'translateY(-2px)' : 'translateY(0)',
                boxShadow: hoverSend && msg.trim() ? '0 8px 24px rgba(16, 185, 129, 0.4)' : '0 4px 12px rgba(16, 185, 129, 0.3)'
              }}
              onMouseEnter={() => setHoverSend(true)}
              onMouseLeave={() => setHoverSend(false)}
            >
              <span style={{ fontSize: '20px' }}>📤</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main App
export default function App() {
  const [page, setPage] = useState("upload");
  const [data, setData] = useState(null);

  if (page === "upload")
    return <Upload onResult={(d) => { setData(d); setPage("result"); }} />;

  if (page === "result")
    return <Result data={data} onChat={() => setPage("chat")} onBack={() => setPage("upload")} />;

  return <Chat onBack={() => setPage("result")} />;
}