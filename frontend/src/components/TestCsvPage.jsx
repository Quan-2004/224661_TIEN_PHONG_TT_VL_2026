import { useState, useRef } from 'react';
import { predictAPI, uploadAPI } from '../services/api';

/**
 * TestCSVPage – Phase 3: Luồng Kiểm Thử (Test Workflow)
 * ======================================================
 * Upload file test.csv → Backend tự động:
 *   1. Load GlobalScaler đóng băng → transform X_test
 *   2. Đưa qua tất cả model song song
 *   3. Phân xử bằng Euclidean Nearest-SV Tie-break
 *   4. Trả về nhãn dự đoán ŷ + báo cáo
 */
export default function TestCSVPage() {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [classColumn, setClassColumn] = useState('');
  const [availableColumns, setAvailableColumns] = useState([]);
  const [minMargin, setMinMargin] = useState(0.0);
  const [loading, setLoading] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileRef = useRef();

  const handleFile = async (f) => {
    if (!f || !f.name.endsWith('.csv')) {
      setError('Chỉ chấp nhận file .csv');
      return;
    }
    setFile(f);
    setError('');
    setResult(null);
    setAvailableColumns([]);
    setClassColumn('');

    // Đọc tên cột qua API preview
    setPreviewing(true);
    try {
      const pv = await uploadAPI.previewCSV(f);
      if (pv && pv.columns) {
        setAvailableColumns(pv.columns.map(c => c.name));
        if (pv.suggested?.class_column) {
            setClassColumn(pv.suggested.class_column);
        }
      }
    } catch (err) {
      console.error("Lỗi đọc cột CSV:", err);
    } finally {
      setPreviewing(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) { setError('Chưa chọn file test.csv'); return; }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await predictAPI.predictCSV(file, {
        class_column: classColumn.trim() || undefined,
        min_margin:   minMargin,
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const predClasses     = result ? [...new Set(result.results.map(r => r.predicted_class))] : [];
  const labelDist       = result?.label_distribution || {};
  const unknownCount    = result?.n_unknown || 0;
  const totalCount      = result?.n_samples || 0;
  const knownCount      = totalCount - unknownCount;

  const getLabelColor = (label) => {
    if (label === 'unknown') return '#ef4444';
    if (label.startsWith('low_confidence/')) return '#f59e0b';
    return '#22c55e';
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🧪 Kiểm Thử CSV (Phase 3)</h1>
        <p className="page-subtitle">
          Upload file test.csv thô → GlobalScaler scale → OC-SVM predict → Kết quả phân loại
        </p>
      </div>

      {/* Upload Zone */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title">📂 File Test CSV</div>
        </div>

        <div
          className={`upload-zone ${dragOver ? 'active' : ''}`}
          style={{
            border: `2px dashed ${dragOver ? 'var(--accent-blue)' : 'var(--border-color)'}`,
            borderRadius: 12,
            padding: 40,
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s',
            background: dragOver ? 'rgba(59,130,246,0.07)' : 'var(--bg-secondary)',
            marginBottom: 20,
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <div style={{ fontSize: 40, marginBottom: 8 }}>{file ? '📄' : '📥'}</div>
          {file ? (
            <>
              <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 15 }}>
                {file.name}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 4 }}>
                {(file.size / 1024).toFixed(1)} KB – Click để đổi file
              </div>
            </>
          ) : (
            <>
              <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>
                Kéo thả file <strong>test.csv</strong> vào đây hoặc click để chọn
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 6 }}>
                File không cần cột nhãn – hệ thống sẽ dự đoán tự động
              </div>
            </>
          )}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {/* Config options */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>
          <div style={{ position: 'relative' }}>
            <label className="form-label">
              Cột nhãn thật (tuỳ chọn)
              <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>
                – để tính accuracy
              </span>
            </label>
            {previewing ? (
                 <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: '8px 12px', border: '1px solid var(--border-color)', borderRadius: 8 }}>
                    <span className="spinner" style={{ width: 14, height: 14, marginRight: 8 }} /> Đang đọc cột...
                 </div>
            ) : availableColumns.length > 0 ? (
                <div style={{ position: 'relative' }}>
                    <select
                        className="form-control"
                        value={classColumn}
                        onChange={(e) => setClassColumn(e.target.value)}
                        style={{ cursor: 'pointer', appearance: 'none' }}
                    >
                        <option value="">-- Không chọn (Chỉ dự đoán) --</option>
                        {availableColumns.map(col => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                    <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-muted)'}}>▼</div>
                </div>
            ) : (
                <input
                  className="form-control"
                  placeholder='Ví dụ: "state", "label", "class"'
                  value={classColumn}
                  onChange={(e) => setClassColumn(e.target.value)}
                  disabled={!file}
                />
            )}
          </div>
          <div>
            <label className="form-label">
              Biên độ phân xử (min_margin)
              <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>
                – 0 = tắt
              </span>
            </label>
            <input
              className="form-control"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={minMargin}
              onChange={(e) => setMinMargin(parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginTop: 16 }}>⚠️ {error}</div>
        )}

        <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
          <button
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={loading || !file}
            style={{ minWidth: 160 }}
          >
            {loading ? (
              <><span className="spinner" style={{ width: 16, height: 16, marginRight: 8 }} />Đang dự đoán...</>
            ) : (
              '🚀 Chạy Kiểm Thử'
            )}
          </button>
          {file && (
            <button
              className="btn btn-outline"
              onClick={() => { setFile(null); setResult(null); setError(''); }}
            >
              ✕ Xoá file
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary Stats */}
          <div className="stat-grid" style={{ marginBottom: 24 }}>
            <div className="stat-card">
              <div className="stat-icon blue">📊</div>
              <div className="stat-info">
                <div className="stat-value">{totalCount.toLocaleString()}</div>
                <div className="stat-label">Tổng mẫu</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon" style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e' }}>✅</div>
              <div className="stat-info">
                <div className="stat-value">{knownCount.toLocaleString()}</div>
                <div className="stat-label">Đã phân loại</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon" style={{ background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}>❓</div>
              <div className="stat-info">
                <div className="stat-value">{unknownCount.toLocaleString()}</div>
                <div className="stat-label">Unknown Concept</div>
              </div>
            </div>
            {result.accuracy != null && (
              <div className="stat-card">
                <div className="stat-icon orange">🎯</div>
                <div className="stat-info">
                  <div className="stat-value">{(result.accuracy * 100).toFixed(1)}%</div>
                  <div className="stat-label">Độ chính xác</div>
                </div>
              </div>
            )}
          </div>

          {/* Label Distribution */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">📈 Phân phối nhãn dự đoán</div>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {result.n_classes_active} model active · GlobalScaler: {result.scaler_info?.n_features} features
              </span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, padding: '8px 0' }}>
              {Object.entries(labelDist).sort((a, b) => b[1] - a[1]).map(([label, count]) => (
                <div key={label} style={{
                  background: 'var(--bg-tertiary)',
                  borderRadius: 10,
                  padding: '10px 18px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  border: '1px solid var(--border-color)',
                }}>
                  <span style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: getLabelColor(label), flexShrink: 0,
                  }} />
                  <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 14 }}>
                    {label}
                  </span>
                  <span style={{
                    background: 'var(--bg-secondary)',
                    borderRadius: 6, padding: '2px 8px',
                    fontSize: 13, color: 'var(--text-muted)', fontFamily: 'monospace',
                  }}>
                    {count} ({((count / totalCount) * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Results Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">📋 Chi tiết kết quả dự đoán</div>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Hiển thị {Math.min(200, result.results.length)} / {result.results.length} mẫu
              </span>
            </div>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Nhãn dự đoán ŷ</th>
                    <th>Confidence</th>
                    {result.results[0]?.true_class != null && <th>Nhãn thật</th>}
                    {result.results[0]?.correct != null && <th>Đúng/Sai</th>}
                  </tr>
                </thead>
                <tbody>
                  {result.results.slice(0, 200).map((r, i) => (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{i + 1}</td>
                      <td>
                        <span style={{
                          display: 'inline-block',
                          padding: '3px 10px',
                          borderRadius: 6,
                          fontSize: 13,
                          fontWeight: 600,
                          background: getLabelColor(r.predicted_class) + '22',
                          color: getLabelColor(r.predicted_class),
                          border: `1px solid ${getLabelColor(r.predicted_class)}44`,
                        }}>
                          {r.predicted_class}
                        </span>
                        {r.is_low_confidence && (
                          <span style={{ marginLeft: 6, fontSize: 11, color: '#f59e0b' }}>⚠</span>
                        )}
                      </td>
                      <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
                        {(r.confidence * 100).toFixed(2)}%
                      </td>
                      {r.true_class != null && (
                        <td style={{ color: 'var(--text-muted)', fontSize: 13 }}>{r.true_class}</td>
                      )}
                      {r.correct != null && (
                        <td style={{ fontSize: 18 }}>
                          {r.correct ? '✅' : '❌'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
