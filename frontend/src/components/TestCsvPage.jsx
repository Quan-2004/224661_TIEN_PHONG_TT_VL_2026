import { useState, useCallback } from "react";

const API = "http://localhost:8000";

// Màu sắc cho từng class (xoay vòng)
const CLASS_COLORS = [
  { bg: "rgba(79,142,247,0.12)",   border: "rgba(79,142,247,0.4)",  text: "#4f8ef7",  label: "blue"   },
  { bg: "rgba(52,211,153,0.12)",   border: "rgba(52,211,153,0.4)",  text: "#34d399",  label: "green"  },
  { bg: "rgba(167,139,250,0.12)",  border: "rgba(167,139,250,0.4)", text: "#a78bfa",  label: "purple" },
  { bg: "rgba(34,211,238,0.12)",   border: "rgba(34,211,238,0.4)",  text: "#22d3ee",  label: "cyan"   },
];
const UNKNOWN_COLOR = { bg: "rgba(251,146,60,0.12)", border: "rgba(251,146,60,0.4)", text: "#fb923c" };

export default function TestCsvPage() {
  const [file, setFile]       = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [result, setResult]   = useState(null);

  const reset = () => { setFile(null); setResult(null); setError(null); };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.name.endsWith(".csv")) { setFile(f); setError(null); setResult(null); }
    else setError("Chỉ chấp nhận file .csv");
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setError(null); setResult(null); }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("csv_file", file);
    try {
      const res = await fetch(`${API}/test-csv/upload`, { method: "POST", body: fd });
      if (!res.ok) {
        const e2 = await res.json();
        throw new Error(e2.detail || "Lỗi server");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Tính tổng & phần trăm
  const total = result ? Object.values(result.results_summary).reduce((a, b) => a + b, 0) : 0;
  const entries = result
    ? Object.entries(result.results_summary).sort((a, b) => b[1] - a[1])
    : [];

  let colorIdx = 0;

  return (
    <div style={{ animation: "fadeIn .35s ease" }}>
      {/* ── Header ── */}
      <div style={{ marginBottom: 32 }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 10,
          background: "rgba(79,142,247,0.1)", border: "1px solid rgba(79,142,247,0.25)",
          borderRadius: 10, padding: "6px 14px", marginBottom: 14,
        }}>
          <span style={{ fontSize: 15 }}>🔬</span>
          <span style={{ color: "var(--accent-blue)", fontSize: 13, fontWeight: 600 }}>Logic Test – One-vs-Rest</span>
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 800, color: "var(--text-primary)", marginBottom: 8, letterSpacing: "-0.5px" }}>
          Kiểm thử Dữ liệu CSV Mới
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: 14, maxWidth: 600 }}>
          Tải lên file CSV chứa dữ liệu cần phân loại. Hệ thống OvR sẽ quét qua tất cả các model đã huấn luyện và trả về nhãn lớp cho từng dòng.
        </p>
      </div>

      {/* ── Upload Card ── */}
      <div style={{
        background: "var(--bg-card)", border: "1px solid var(--border-color)",
        borderRadius: 16, padding: 28, marginBottom: 24, backdropFilter: "blur(12px)",
      }}>
        <div style={{ marginBottom: 18, display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>📂</span>
          <span style={{ fontWeight: 700, fontSize: 16, color: "var(--text-primary)" }}>Chọn File Dữ Liệu</span>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragOver ? "var(--accent-blue)" : file ? "var(--accent-green)" : "var(--border-color)"}`,
            borderRadius: 12, padding: "36px 24px", textAlign: "center",
            background: dragOver
              ? "rgba(79,142,247,0.07)"
              : file ? "rgba(52,211,153,0.06)" : "rgba(255,255,255,0.02)",
            transition: "var(--transition)", cursor: "pointer",
            position: "relative",
          }}
          onClick={() => !file && document.getElementById("csv-input").click()}
        >
          <input id="csv-input" type="file" accept=".csv" onChange={handleFileChange} style={{ display: "none" }} />
          {file ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}>
              <div style={{
                width: 48, height: 48, borderRadius: 10,
                background: "rgba(52,211,153,0.15)", display: "flex",
                alignItems: "center", justifyContent: "center", fontSize: 22,
              }}>📄</div>
              <div style={{ textAlign: "left" }}>
                <div style={{ fontWeight: 700, color: "var(--accent-green)", fontSize: 15 }}>{file.name}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 2 }}>
                  {(file.size / 1024).toFixed(1)} KB &nbsp;·&nbsp; Sẵn sàng kiểm thử
                </div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); reset(); }}
                style={{
                  marginLeft: "auto", background: "rgba(248,113,113,0.1)",
                  border: "1px solid rgba(248,113,113,0.3)", color: "var(--accent-red)",
                  borderRadius: 7, width: 32, height: 32, cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
                }}
              >✕</button>
            </div>
          ) : (
            <>
              <div style={{ fontSize: 36, marginBottom: 10 }}>
                {dragOver ? "🎯" : "📁"}
              </div>
              <div style={{ color: "var(--text-primary)", fontWeight: 600, fontSize: 15, marginBottom: 6 }}>
                {dragOver ? "Thả file vào đây!" : "Kéo & Thả file CSV vào đây"}
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 18 }}>
                hoặc nhấn để chọn file từ máy tính
              </div>
              <span style={{
                display: "inline-block", padding: "8px 20px",
                background: "rgba(79,142,247,0.12)", border: "1px solid rgba(79,142,247,0.3)",
                borderRadius: 8, color: "var(--accent-blue)", fontWeight: 600, fontSize: 13,
              }}>
                Duyệt File...
              </span>
            </>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="alert alert-error" style={{ marginTop: 16 }}>
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Run Button */}
        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 20 }}>
          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            className="btn btn-primary"
            style={{ minWidth: 160, justifyContent: "center", fontSize: 14, padding: "11px 24px" }}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                <span>Đang phân loại...</span>
              </>
            ) : (
              <><span>🚀</span> Bắt đầu Kiểm thử</>
            )}
          </button>
        </div>
      </div>

      {/* ── Kết quả ── */}
      {result && (
        <div style={{ animation: "slideUp .35s ease" }}>
          {/* Tổng quan */}
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 14, marginBottom: 24,
          }}>
            <div className="stat-card">
              <div className="stat-icon blue">📊</div>
              <div className="stat-info">
                <div className="stat-value">{total}</div>
                <div className="stat-label">Tổng mẫu test</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon green">✅</div>
              <div className="stat-info">
                <div className="stat-value">
                  {entries.filter(([k]) => k.toLowerCase() !== "unknown").reduce((s, [, v]) => s + v, 0)}
                </div>
                <div className="stat-label">Đã phân loại</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon orange">❓</div>
              <div className="stat-info">
                <div className="stat-value">{result.results_summary["unknown"] ?? 0}</div>
                <div className="stat-label">Dữ liệu lạ (Unknown)</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon purple">🏷️</div>
              <div className="stat-info">
                <div className="stat-value">{entries.filter(([k]) => k.toLowerCase() !== "unknown").length}</div>
                <div className="stat-label">Số lớp phát hiện</div>
              </div>
            </div>
          </div>

          {/* Chi tiết từng lớp */}
          <div style={{
            background: "var(--bg-card)", border: "1px solid var(--border-color)",
            borderRadius: 16, padding: 28, marginBottom: 24,
          }}>
            <h3 style={{ fontWeight: 700, fontSize: 16, marginBottom: 20, color: "var(--text-primary)" }}>
              Phân bổ theo Lớp
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {entries.map(([className, count]) => {
                const isUnknown = className.toLowerCase() === "unknown";
                const color = isUnknown ? UNKNOWN_COLOR : CLASS_COLORS[colorIdx++ % CLASS_COLORS.length];
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
                return (
                  <div key={className}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{
                          display: "inline-flex", padding: "3px 10px",
                          background: color.bg, border: `1px solid ${color.border}`,
                          borderRadius: 20, color: color.text, fontSize: 12, fontWeight: 600,
                        }}>
                          {isUnknown ? "❓ Unknown" : `🏷️ ${className}`}
                        </span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>{pct}%</span>
                        <span style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: 16, minWidth: 40, textAlign: "right" }}>
                          {count}
                        </span>
                        <span style={{ color: "var(--text-muted)", fontSize: 12 }}>mẫu</span>
                      </div>
                    </div>
                    <div style={{
                      height: 8, borderRadius: 4, background: "rgba(255,255,255,0.06)", overflow: "hidden"
                    }}>
                      <div style={{
                        height: "100%", width: `${pct}%`,
                        background: color.text, borderRadius: 4,
                        transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                        opacity: 0.8,
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Download */}
          <div style={{
            background: "rgba(79,142,247,0.06)", border: "1px solid rgba(79,142,247,0.2)",
            borderRadius: 14, padding: "20px 24px",
            display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16,
            flexWrap: "wrap",
          }}>
            <div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
                📋 File Kết quả Chi tiết
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                Mỗi dòng dữ liệu kèm nhãn dự đoán và điểm Decision Score của từng lớp
              </div>
            </div>
            <a
              href={`${API}${result.download_url}`}
              download="test_results_detailed.csv"
              target="_blank"
              rel="noreferrer"
              className="btn btn-primary"
              style={{ textDecoration: "none", whiteSpace: "nowrap" }}
            >
              <span>⬇️</span> Tải File .CSV
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
