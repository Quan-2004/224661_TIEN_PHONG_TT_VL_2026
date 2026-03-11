import { useState, useEffect, useRef } from 'react';

// ── Bảng màu cho các lớp ─────────────────────────────────────────────────────
const CLASS_COLORS = [
  '#4f9ffe', // blue
  '#ff6b6b', // red
  '#51cf66', // green
  '#ffd43b', // yellow
  '#cc5de8', // purple
  '#ff922b', // orange
  '#20c997', // teal
  '#f06595', // pink
  '#74c0fc', // light blue
  '#a9e34b', // lime
];

function getColor(idx) {
  return CLASS_COLORS[idx % CLASS_COLORS.length];
}

/**
 * SupportVectorPlot
 * Biểu đồ scatter plot 2D các support vectors của tất cả lớp (PCA projection).
 * Mỗi lớp một màu, các điểm được nối với convex hull để thấy rõ vùng phân bố.
 */
export default function SupportVectorPlot() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [hovered, setHovered] = useState(null); // { className, x, y, screenX, screenY }
  const [hiddenClasses, setHiddenClasses] = useState(new Set());
  const canvasRef = useRef(null);
  const API = 'http://localhost:8000';

  // ── Fetch dữ liệu ───────────────────────────────────────────────────────────
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/models/sv-plot`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Lỗi server');
      }
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  // ── Vẽ canvas khi data thay đổi ─────────────────────────────────────────────
  useEffect(() => {
    if (!data || !canvasRef.current) return;
    drawCanvas();
  }, [data, hiddenClasses]);

  const toggleClass = (cname) => {
    setHiddenClasses(prev => {
      const next = new Set(prev);
      if (next.has(cname)) next.delete(cname);
      else next.add(cname);
      return next;
    });
  };

  // ── Compute min/max để normalize ────────────────────────────────────────────
  function computeRange(classes) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const cls of classes) {
      for (const [x, y] of cls.points) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    return { minX, maxX, minY, maxY };
  }

  function toCanvas(x, y, range, pad, W, H) {
    const { minX, maxX, minY, maxY } = range;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const cx = pad + ((x - minX) / spanX) * (W - 2 * pad);
    const cy = H - pad - ((y - minY) / spanY) * (H - 2 * pad); // flip Y
    return [cx, cy];
  }

  // Simple convex hull – Graham scan
  function convexHull(points) {
    const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (pts.length < 3) return pts;
    const cross = (O, A, B) => (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
        lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
        upper.pop();
      upper.push(p);
    }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const pad = 48;
    const classes = data.classes.filter(c => !hiddenClasses.has(c.class_name));
    if (classes.length === 0) return;

    const range = computeRange(classes);

    // ── Grid ────────────────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const GRID = 8;
    for (let i = 0; i <= GRID; i++) {
      const xLine = pad + (i / GRID) * (W - 2 * pad);
      const yLine = pad + (i / GRID) * (H - 2 * pad);
      ctx.beginPath(); ctx.moveTo(xLine, pad); ctx.lineTo(xLine, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, yLine); ctx.lineTo(W - pad, yLine); ctx.stroke();
    }

    // ── Axes ────────────────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();

    // Axis labels
    const pct = data.pca_variance_explained;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`PC1 (${pct[0]}%)`, W / 2, H - 8);
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`PC2 (${pct[1]}%)`, 0, 0);
    ctx.restore();

    // ── Convex Hull (filled, semi-transparent) ───────────────────────────────
    data.classes.forEach((cls, idx) => {
      if (hiddenClasses.has(cls.class_name) || cls.points.length < 3) return;
      const color = getColor(idx);
      const hull = convexHull(cls.points.map(([x, y]) => toCanvas(x, y, range, pad, W, H)));
      ctx.beginPath();
      ctx.moveTo(hull[0][0], hull[0][1]);
      hull.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = color + '22';
      ctx.fill();
      ctx.strokeStyle = color + '88';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // ── Points ───────────────────────────────────────────────────────────────
    data.classes.forEach((cls, idx) => {
      if (hiddenClasses.has(cls.class_name)) return;
      const color = getColor(idx);
      for (const [x, y] of cls.points) {
        const [cx, cy] = toCanvas(x, y, range, pad, W, H);
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff3';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    });
  }

  // ── Mouse hover để hiển thị tooltip ─────────────────────────────────────────
  function handleMouseMove(e) {
    if (!data || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const cx = mx * scaleX;
    const cy = my * scaleY;

    const W = canvasRef.current.width;
    const H = canvasRef.current.height;
    const pad = 48;
    const classes = data.classes.filter(c => !hiddenClasses.has(c.class_name));
    const range = computeRange(classes);

    let found = null;
    let minDist = 10;
    data.classes.forEach((cls, idx) => {
      if (hiddenClasses.has(cls.class_name)) return;
      cls.points.forEach(([px, py]) => {
        const [pcx, pcy] = toCanvas(px, py, range, pad, W, H);
        const dist = Math.hypot(cx - pcx, cy - pcy);
        if (dist < minDist) {
          minDist = dist;
          found = { className: cls.class_name, x: px, y: py, screenX: e.clientX, screenY: e.clientY, colorIdx: idx };
        }
      });
    });
    setHovered(found);
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="page-content" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: 'var(--text-primary, #f0f4ff)' }}>
            📌 Support Vector Map
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-secondary, #8b9ac9)' }}>
            Toàn bộ Support Vectors của các lớp được chiếu lên không gian 2D (PCA) để kiểm tra sự chồng lấp.
          </p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: 'var(--accent-blue, #4f9ffe)', color: '#fff',
            fontWeight: 600, fontSize: 13, opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? '⏳ Đang tải...' : '🔄 Làm mới'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: '#ff4d4f22', border: '1px solid #ff4d4f66',
          borderRadius: 10, padding: '14px 18px', color: '#ff6b6b', marginBottom: 20,
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Warning từ backend (ví dụ feature mismatch) */}
      {data?.warning && (
        <div style={{
          background: '#ffd43b22', border: '1px solid #ffd43b66',
          borderRadius: 10, padding: '14px 18px', color: '#ffd43b', marginBottom: 20,
        }}>
          ⚠️ {data.warning}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && !data && (
        <div style={{ textAlign: 'center', padding: 80, color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⏳</div>
          <div>Đang load và chiếu Support Vectors…</div>
        </div>
      )}

      {/* Main content */}
      {data && !data.warning && (
        <>
          {/* Meta Info Bar */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
            <InfoBadge label="Số lớp" value={data.classes.length} />
            <InfoBadge label="PC1 giải thích" value={`${data.pca_variance_explained[0]}%`} />
            <InfoBadge label="PC2 giải thích" value={`${data.pca_variance_explained[1]}%`} />
            <InfoBadge label="Tổng PC1+PC2" value={`${(data.pca_variance_explained[0] + data.pca_variance_explained[1]).toFixed(1)}%`} />
            <InfoBadge label="Features gốc" value={data.n_features_original} />
          </div>

          {/* Legend + Toggle */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
            {data.classes.map((cls, idx) => {
              const color = getColor(idx);
              const hidden = hiddenClasses.has(cls.class_name);
              return (
                <button
                  key={cls.class_name}
                  onClick={() => toggleClass(cls.class_name)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 7,
                    padding: '6px 14px', borderRadius: 20,
                    border: `1.5px solid ${hidden ? '#444' : color}`,
                    background: hidden ? '#1a1d2e' : color + '22',
                    color: hidden ? '#555' : color,
                    cursor: 'pointer', fontSize: 12, fontWeight: 600,
                    transition: 'all 0.2s',
                  }}
                >
                  <span style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: hidden ? '#444' : color, display: 'inline-block',
                  }} />
                  {cls.class_name}
                  <span style={{ opacity: 0.6, fontSize: 11 }}>({cls.n_sv} SVs)</span>
                </button>
              );
            })}
          </div>

          {/* Canvas scatter plot */}
          <div style={{ position: 'relative' }}>
            <canvas
              ref={canvasRef}
              width={920}
              height={520}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHovered(null)}
              style={{
                width: '100%', height: 'auto',
                borderRadius: 14,
                background: 'linear-gradient(135deg, #0d0f1c 0%, #141728 100%)',
                border: '1px solid rgba(255,255,255,0.08)',
                cursor: 'crosshair',
              }}
            />

            {/* Tooltip */}
            {hovered && (
              <div style={{
                position: 'fixed',
                left: hovered.screenX + 14,
                top: hovered.screenY - 10,
                background: '#1e2236ee',
                border: `1px solid ${getColor(hovered.colorIdx)}88`,
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 12,
                color: '#f0f4ff',
                pointerEvents: 'none',
                zIndex: 9999,
                backdropFilter: 'blur(8px)',
              }}>
                <div style={{ color: getColor(hovered.colorIdx), fontWeight: 700, marginBottom: 2 }}>
                  {hovered.className}
                </div>
                <div>PC1: <b>{hovered.x.toFixed(4)}</b></div>
                <div>PC2: <b>{hovered.y.toFixed(4)}</b></div>
              </div>
            )}
          </div>

          {/* Overlap analysis table */}
          <OverlapTable classes={data.classes} />
        </>
      )}
    </div>
  );
}

// ── Overlap Analysis Table ───────────────────────────────────────────────────
function OverlapTable({ classes }) {
  if (classes.length < 2) return null;

  // Tính bounding box mỗi lớp
  function bbox(points) {
    if (!points.length) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    return { minX, maxX, minY, maxY };
  }

  function overlapPercent(b1, b2) {
    if (!b1 || !b2) return 0;
    const ox = Math.max(0, Math.min(b1.maxX, b2.maxX) - Math.max(b1.minX, b2.minX));
    const oy = Math.max(0, Math.min(b1.maxY, b2.maxY) - Math.max(b1.minY, b2.minY));
    const area = ox * oy;
    const a1 = (b1.maxX - b1.minX) * (b1.maxY - b1.minY);
    const a2 = (b2.maxX - b2.minX) * (b2.maxY - b2.minY);
    const minArea = Math.min(a1, a2);
    if (!minArea) return 0;
    return Math.round((area / minArea) * 100);
  }

  const rows = [];
  for (let i = 0; i < classes.length; i++) {
    for (let j = i + 1; j < classes.length; j++) {
      const b1 = bbox(classes[i].points);
      const b2 = bbox(classes[j].points);
      const pct = overlapPercent(b1, b2);
      rows.push({ a: classes[i].class_name, b: classes[j].class_name, pct, colorA: getColor(i), colorB: getColor(j) });
    }
  }

  return (
    <div style={{ marginTop: 24 }}>
      <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary, #f0f4ff)', marginBottom: 12 }}>
        🔍 Phân tích Bounding Box Overlap (2D)
      </h3>
      <p style={{ fontSize: 12, color: 'var(--text-secondary, #8b9ac9)', marginBottom: 14, lineHeight: 1.6 }}>
        Phần trăm diện tích vùng giao nhau / diện tích bounding box nhỏ hơn.
        Nếu &gt; 0% → các lớp có thể có support vectors chồng lấp trong không gian PCA 2D.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
              <th style={thStyle}>Lớp A</th>
              <th style={thStyle}>Lớp B</th>
              <th style={thStyle}>Overlap %</th>
              <th style={thStyle}>Đánh giá</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ a, b, pct, colorA, colorB }) => {
              const risk = pct === 0 ? { label: '✅ Tốt – Tách biệt hoàn toàn', color: '#51cf66' }
                : pct < 20    ? { label: '⚠️ Nhẹ – Chú ý khi test', color: '#ffd43b' }
                : pct < 60    ? { label: '🔶 Trung bình – Khả năng nhầm lẫn', color: '#ff922b' }
                              : { label: '🔴 Cao – Rất dễ nhầm lẫn', color: '#ff6b6b' };
              return (
                <tr key={`${a}-${b}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ ...tdStyle, color: colorA, fontWeight: 600 }}>{a}</td>
                  <td style={{ ...tdStyle, color: colorB, fontWeight: 600 }}>{b}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ flex: 1, height: 6, background: '#1a1d2e', borderRadius: 3 }}>
                        <div style={{ width: `${Math.min(pct, 100)}%`, height: '100%', background: risk.color, borderRadius: 3, transition: 'width 0.5s' }} />
                      </div>
                      <span style={{ minWidth: 36, fontWeight: 700, color: risk.color }}>{pct}%</span>
                    </div>
                  </td>
                  <td style={{ ...tdStyle, color: risk.color, fontSize: 12 }}>{risk.label}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function InfoBadge({ label, value }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 10, padding: '8px 16px', minWidth: 100,
    }}>
      <div style={{ fontSize: 11, color: '#8b9ac9', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: '#f0f4ff' }}>{value}</div>
    </div>
  );
}

const thStyle = {
  padding: '10px 16px', textAlign: 'left',
  color: '#8b9ac9', fontWeight: 600, fontSize: 12,
};
const tdStyle = {
  padding: '10px 16px', color: '#f0f4ff',
};
