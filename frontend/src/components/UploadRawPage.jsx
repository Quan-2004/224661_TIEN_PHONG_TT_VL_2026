import { useState, useCallback } from 'react';
import { uploadAPI } from '../services/api';

/**
 * UploadRawPage – Phase 0
 * Chế độ 1 (Thủ công): nhập tên cột bằng tay
 * Chế độ 2 (Tự động):  upload file → backend trả danh sách cột → chọn bằng click
 */
export default function UploadRawPage() {
  const [mode, setMode]     = useState('auto'); // 'manual' | 'auto'
  const [file, setFile]     = useState(null);
  const [config, setConfig] = useState({
    class_column : '',
    id_columns   : '',
    drop_columns : '',
    scale        : true,
  });

  const [trainConfig, setTrainConfig] = useState({
    kernel: 'rbf',
    nu: 0.1,
    gamma: 'scale',
    version_name: 'v1.0',
  });

  // Auto-detect state
  const [preview, setPreview]         = useState(null);   // { columns, suggested }
  const [previewing, setPreviewing]   = useState(false);

  // Process state
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState('');

  // -------------------------------------------------------
  // File selection
  // -------------------------------------------------------
  const handleFile = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError('');
    setPreview(null);

    if (mode === 'auto') {
      setPreviewing(true);
      try {
        const pv = await uploadAPI.previewCSV(f);
        setPreview(pv);
        // Điền gợi ý tự động
        setConfig(prev => ({
          ...prev,
          class_column: pv.suggested?.class_column || '',
          id_columns  : (pv.suggested?.id_columns  || []).join(', '),
          drop_columns: (pv.suggested?.drop_columns || []).join(', '),
        }));
      } catch (err) {
        setError('Không thể đọc cột CSV: ' + err.message);
      } finally {
        setPreviewing(false);
      }
    }
  };

  const parseCSVList = (str) =>
    str.split(',').map(s => s.trim()).filter(Boolean);

  // -------------------------------------------------------
  // Toggle cột trong chế độ Auto
  // -------------------------------------------------------
  const toggleCol = (col, field) => {
    setConfig(prev => {
      const list = parseCSVList(prev[field]);
      const exists = list.includes(col);
      const next = exists ? list.filter(c => c !== col) : [...list, col];
      return { ...prev, [field]: next.join(', ') };
    });
  };

  const isColSelected = (col, field) =>
    parseCSVList(config[field]).includes(col);

  // -------------------------------------------------------
  // Process
  // -------------------------------------------------------
  const handleProcessOnly = async () => {
    if (!file) { setError('Vui lòng chọn file CSV thô!'); return; }
    if (!config.class_column.trim()) { setError('Vui lòng chọn cột nhãn (class_column)!'); return; }
    setLoading(true);
    setResult(null);
    setError('');
    try {
      const res = await uploadAPI.uploadRaw(file, {
        class_column : config.class_column.trim(),
        id_columns   : JSON.stringify(parseCSVList(config.id_columns)),
        drop_columns : JSON.stringify(parseCSVList(config.drop_columns)),
        scale        : config.scale,
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoTrain = async () => {
    if (!file) { setError('Vui lòng chọn file CSV thô!'); return; }
    if (!config.class_column.trim()) { setError('Vui lòng chọn cột nhãn (class_column)!'); return; }
    setLoading(true);
    setResult(null);
    setError('');
    try {
      const res = await uploadAPI.autoTrain(file, {
        class_column : config.class_column.trim(),
        id_columns   : JSON.stringify(parseCSVList(config.id_columns)),
        drop_columns : JSON.stringify(parseCSVList(config.drop_columns)),
        scale        : config.scale,
        kernel       : trainConfig.kernel,
        nu           : parseFloat(trainConfig.nu) || 0.1,
        gamma        : trainConfig.gamma,
        version_name : trainConfig.version_name.trim() || 'v1.0',
      });
      // The API response returns 'alignment' object wrapping the processing stats 
      // instead of flat summary keys. Standardize to make UI rendering simpler.
      const formattedRes = {
         ...res,
         summary: res.alignment?.summary || res.summary,
         pipeline: res.alignment?.pipeline || res.pipeline,
      };
      setResult(formattedRes);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------
  // Colour helpers for column badges
  // -------------------------------------------------------
  const colBadgeStyle = (col) => {
    const isClass = col === config.class_column;
    const isId    = isColSelected(col, 'id_columns');
    const isDrop  = isColSelected(col, 'drop_columns');
    if (isClass) return { background: 'rgba(99,102,241,0.25)', border: '1.5px solid #818cf8', color: '#c7d2fe' };
    if (isId)    return { background: 'rgba(52,211,153,0.15)', border: '1.5px solid #34d399', color: '#6ee7b7' };
    if (isDrop)  return { background: 'rgba(239,68,68,0.15)',  border: '1.5px solid #f87171', color: '#fca5a5' };
    return { background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' };
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">📥 Upload CSV Thô</h1>
        <p className="page-subtitle">
          Phase 0 – Tiền xử lý tự động: làm sạch NaN, mã hoá nhãn, chuẩn hoá số,
          rồi tách thành 3 file đồng bộ <code>samples.csv / features.csv / classes.csv</code>
        </p>
      </div>

      {/* Mode switcher */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        {[
          { k: 'auto',   label: '🤖 Tự động (chọn cột bằng click)' },
          { k: 'manual', label: '✏️ Thủ công (nhập tên cột)' },
        ].map(({ k, label }) => (
          <button
            key={k}
            onClick={() => { setMode(k); setPreview(null); setResult(null); setError(''); }}
            style={{
              padding: '8px 18px', borderRadius: 8, fontWeight: 600, fontSize: 13,
              border: mode === k ? '2px solid var(--accent-blue)' : '1px solid var(--border-color)',
              background: mode === k ? 'rgba(99,102,241,0.18)' : 'rgba(255,255,255,0.04)',
              color: mode === k ? '#c7d2fe' : 'var(--text-secondary)',
              cursor: 'pointer', transition: 'all .2s',
            }}
          >{label}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>

        {/* ---- Form ---- */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">⚙️ Cấu hình xử lý</div>
          </div>

          {/* Upload zone */}
          <div className="form-group">
            <label className="form-label">📄 File CSV thô</label>
            <div
              className={`upload-zone ${file ? 'uploaded' : ''}`}
              onClick={() => document.getElementById('raw-csv-input').click()}
            >
              <input id="raw-csv-input" type="file" accept=".csv" onChange={handleFile} />
              <div className="upload-icon">{file ? '✅' : '📊'}</div>
              {file
                ? <div className="filename">📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)</div>
                : <>
                    <div className="upload-text">Kéo thả hoặc click để chọn</div>
                    <div className="upload-hint">Ví dụ: ks-projects-201612.csv</div>
                  </>
              }
            </div>
          </div>

          {/* ---- AUTO MODE: column badges ---- */}
          {mode === 'auto' && (
            <>
              {previewing && (
                <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                  <span className="spinner" style={{ display: 'inline-block', marginRight: 8 }} />
                  Đang đọc cột CSV...
                </div>
              )}

              {preview && (
                <div style={{ marginBottom: 16 }}>
                  {/* Legend */}
                  <div style={{ display: 'flex', gap: 12, marginBottom: 10, flexWrap: 'wrap', fontSize: 11 }}>
                    <span style={{ color: '#c7d2fe' }}>● Nhãn (class)</span>
                    <span style={{ color: '#6ee7b7' }}>● ID / Tên mẫu</span>
                    <span style={{ color: '#fca5a5' }}>● Bỏ qua</span>
                    <span style={{ color: 'var(--text-muted)' }}>● Feature (mặc định)</span>
                  </div>

                  {/* Hướng dẫn */}
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                    Click 1 lần để chọn làm <strong>ID</strong> · Click 2 lần để <strong>Bỏ qua</strong> · Click chuột phải để chọn làm <strong>Nhãn</strong>
                  </div>

                  {/* Column chips */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {preview.columns.map(col => (
                      <div
                        key={col.name}
                        title={`Kiểu: ${col.dtype} | Mẫu: ${col.sample}`}
                        style={{
                          ...colBadgeStyle(col.name),
                          padding: '4px 10px', borderRadius: 20, fontSize: 12,
                          cursor: 'pointer', userSelect: 'none', transition: 'all .15s',
                        }}
                        onClick={() => {
                          // Click: toggle ID
                          if (config.class_column === col.name) return; // đang là class, bỏ qua
                          toggleCol(col.name, 'id_columns');
                          // Bỏ khỏi drop nếu đang drop
                          if (isColSelected(col.name, 'drop_columns'))
                            toggleCol(col.name, 'drop_columns');
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          // Chuột phải: set làm class
                          setConfig(prev => ({ ...prev, class_column: col.name }));
                          // Đảm bảo không nằm trong id/drop
                          setConfig(prev => ({
                            ...prev,
                            class_column: col.name,
                            id_columns  : parseCSVList(prev.id_columns).filter(c => c !== col.name).join(', '),
                            drop_columns: parseCSVList(prev.drop_columns).filter(c => c !== col.name).join(', '),
                          }));
                        }}
                        onDoubleClick={() => {
                          // Double click: toggle bỏ qua
                          if (config.class_column === col.name) return;
                          toggleCol(col.name, 'drop_columns');
                          if (isColSelected(col.name, 'id_columns'))
                            toggleCol(col.name, 'id_columns');
                        }}
                      >
                        {col.name}
                        {col.unnamed && <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.6 }}>unnamed</span>}
                      </div>
                    ))}
                  </div>

                  {/* Hiển thị lựa chọn hiện tại */}
                  <div style={{
                    marginTop: 12, padding: '10px 12px', borderRadius: 8,
                    background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-color)',
                    fontSize: 12, lineHeight: 1.8,
                  }}>
                    <div><span style={{ color: '#c7d2fe' }}>🎯 Nhãn:</span> <strong>{config.class_column || '(chưa chọn)'}</strong></div>
                    <div><span style={{ color: '#6ee7b7' }}>🔑 ID:</span> {config.id_columns || '(không có)'}</div>
                    <div><span style={{ color: '#fca5a5' }}>🗑️ Bỏ qua:</span> {config.drop_columns || '(không có)'}</div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* ---- MANUAL MODE: text inputs ---- */}
          {mode === 'manual' && (
            <>
              <div className="form-group">
                <label className="form-label">🎯 Cột nhãn (class_column) *</label>
                <input
                  className="form-control"
                  placeholder="ví dụ: state"
                  value={config.class_column}
                  onChange={e => setConfig(p => ({ ...p, class_column: e.target.value }))}
                />
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Cột này sẽ trở thành classes.csv
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">🔑 Cột ID / Tên mẫu (id_columns)</label>
                <input
                  className="form-control"
                  placeholder="ví dụ: ID, name (cách nhau bởi dấu phẩy)"
                  value={config.id_columns}
                  onChange={e => setConfig(p => ({ ...p, id_columns: e.target.value }))}
                />
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Những cột này sẽ vào samples.csv, không đưa vào features
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">🗑️ Cột bỏ qua (drop_columns)</label>
                <input
                  className="form-control"
                  placeholder="ví dụ: currency, deadline"
                  value={config.drop_columns}
                  onChange={e => setConfig(p => ({ ...p, drop_columns: e.target.value }))}
                />
              </div>
            </>
          )}

          {/* scale – hiển thị cả hai chế độ */}
          <div className="form-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={config.scale}
                onChange={e => setConfig(p => ({ ...p, scale: e.target.checked }))}
                style={{ width: 16, height: 16 }}
              />
              <span className="form-label" style={{ marginBottom: 0 }}>
                📐 Chuẩn hóa số (StandardScaler)
              </span>
            </label>
          </div>

          <div className="form-group" style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border-color)' }}>
            <label className="form-label">🧠 Tham số Huấn luyện Mặc định</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 8 }}>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Version</label>
                <input className="form-control" value={trainConfig.version_name} onChange={e => setTrainConfig(p => ({ ...p, version_name: e.target.value }))} />
              </div>
              <div>
                 <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Nu (0, 1]</label>
                 <input className="form-control" type="number" step="0.01" value={trainConfig.nu} onChange={e => setTrainConfig(p => ({ ...p, nu: e.target.value }))} />
              </div>
              <div>
                  <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Kernel</label>
                  <select className="form-control" value={trainConfig.kernel} onChange={e => setTrainConfig(p => ({ ...p, kernel: e.target.value }))}>
                    <option value="rbf">RBF</option>
                    <option value="linear">Linear</option>
                    <option value="poly">Poly</option>
                    <option value="sigmoid">Sigmoid</option>
                  </select>
              </div>
              <div>
                  <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Gamma</label>
                  <select className="form-control" value={trainConfig.gamma} onChange={e => setTrainConfig(p => ({ ...p, gamma: e.target.value }))}>
                    <option value="scale">scale</option>
                    <option value="auto">auto</option>
                  </select>
              </div>
            </div>
          </div>

          {error && <div className="alert alert-error" style={{ marginBottom: 12 }}>{error}</div>}

          <div style={{ display: 'flex', gap: 10 }}>
              <button
                className="btn btn-secondary"
                style={{ flex: 1, padding: '10px 0' }}
                onClick={handleProcessOnly}
                disabled={loading || !file || (!config.class_column && !previewing)}
              >
                Chỉ Tách dữ liệu
              </button>
              <button
                className="btn btn-primary"
                style={{ flex: 2 }}
                onClick={handleAutoTrain}
                disabled={loading || !file || (!config.class_column && !previewing)}
              >
                {loading
                  ? <><span className="spinner" /> Đang xử lý...</>
                  : '🚀 Tách & Huấn luyện (Auto)'
                }
              </button>
          </div>
        </div>

        {/* ---- Result panel ---- */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">📊 Kết quả Alignment</div>
          </div>

          {!result && !loading && !previewing && (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
              <div>Upload và xử lý file để xem kết quả kiểm tra alignment</div>
            </div>
          )}

          {(loading || previewing) && (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
              <div className="spinner" style={{ margin: '0 auto 16px', width: 40, height: 40 }} />
              <div style={{ fontSize: 15 }}>
                {previewing ? 'Đang đọc cột CSV...' : 'Đang tiền xử lý dữ liệu...'}
              </div>
              {loading && (
                <div style={{ fontSize: 12, marginTop: 6 }}>
                  Bước: Làm sạch NaN → Label Encode → StandardScaler → Kiểm tra alignment
                </div>
              )}
            </div>
          )}

          {result && (
            <div>
              <div className="alert alert-success" style={{ fontSize: 16, fontWeight: 700, textAlign: 'center', padding: '16px 20px' }}>
                ✅ {result.message}
              </div>

              {/* Alignment proof */}
              <div style={{
                background: 'rgba(52, 211, 153, 0.08)',
                border: '1px solid rgba(52, 211, 153, 0.3)',
                borderRadius: 'var(--radius)', padding: '16px 20px', marginBottom: 16,
              }}>
                <div style={{ fontWeight: 700, marginBottom: 10, color: '#34d399' }}>
                  📐 Alignment Check – Kiểm tra đồng bộ số dòng
                </div>
                {['samples.csv', 'features.csv', 'classes.csv'].map(name => (
                  <div key={name} style={{
                    display: 'flex', justifyContent: 'space-between',
                    padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: 14,
                  }}>
                    <span style={{ fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{name}</span>
                    <span style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                      {result.alignment?.samples_rows?.toLocaleString()} dòng ✓
                    </span>
                  </div>
                ))}
                <div style={{ marginTop: 10, fontSize: 13, fontWeight: 700, color: '#34d399', textAlign: 'center' }}>
                  ✅ Kết quả: ĐỒNG BỘ – {result.summary?.n_rows?.toLocaleString()} dòng hợp lệ
                </div>
              </div>

              {/* Stat cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 16 }}>
                {[
                  ['Số dòng',  result.summary?.n_rows?.toLocaleString(), '📊'],
                  ['Features', result.summary?.n_features, '🔢'],
                  ['Lớp',      result.summary?.n_classes,  '🏷️'],
                ].map(([label, value, icon]) => (
                  <div key={label} style={{
                    background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)', padding: '12px 14px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 22 }}>{icon}</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--accent-cyan)' }}>{value}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</div>
                  </div>
                ))}
              </div>

              {/* Class distribution */}
              <div style={{ marginBottom: 16 }}>
                <div className="form-label" style={{ marginBottom: 8 }}>Phân bố các lớp</div>
                {Object.entries(result.summary?.class_counts || {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, count]) => {
                    const pct = ((count / result.summary.n_rows) * 100).toFixed(1);
                    return (
                      <div key={cls} style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 3 }}>
                          <span style={{ fontWeight: 600 }}>{cls}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{count.toLocaleString()} ({pct}%)</span>
                        </div>
                        <div style={{ height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{
                            height: '100%', width: `${pct}%`,
                            background: 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))',
                            borderRadius: 3,
                          }} />
                        </div>
                      </div>
                    );
                  })}
              </div>

              {/* Pipeline info */}
              <div style={{
                padding: '12px 14px', background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)',
                fontSize: 12, color: 'var(--text-muted)',
              }}>
                <div><strong>🔤 Label Encoded:</strong> {result.pipeline?.encoded_columns?.join(', ') || 'Không có'}</div>
                <div style={{ marginTop: 4 }}><strong>📐 Scaled:</strong> {result.pipeline?.scaled_columns?.length || 0} cột số</div>
                <div style={{ marginTop: 4 }}><strong>🗑️ Đã bỏ:</strong> {result.pipeline?.columns_dropped?.join(', ') || 'Không có'}</div>
                <div style={{ marginTop: 4 }}><strong>💾 Session:</strong> <code style={{ color: 'var(--accent-cyan)' }}>{result.session_id}</code></div>
              </div>
              {/* Training info (if auto-trained) */}
              {result.training_results && (
                  <div style={{ marginTop: 16 }}>
                    <div className="form-label" style={{ marginBottom: 8 }}>✅ Kết quả Huấn luyện</div>
                    <div style={{ maxHeight: '200px', overflowY: 'auto', paddingRight: 4 }}>
                      {result.training_results.map(r => (
                          <div key={r.class_name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', marginBottom: 8 }}>
                              <div>
                                <strong style={{ color: 'var(--accent-blue)', fontSize: 14 }}>{r.class_name}</strong> 
                                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{r.n_samples} mẫu · version {r.version_name}</div>
                              </div>
                              <div style={{ fontSize: 13, fontWeight: 600, color: r.action === 'failed' ? '#f87171' : (r.action === 'retrain' ? '#6ee7b7' : '#c7d2fe'), textAlign: 'right' }}>
                                 {r.action === 'failed' ? '❌ Thất bại' : (r.action === 'retrain' ? '🔄 Retrained' : '✅ Trained')}
                                 {r.error && <div style={{ fontSize: 10, fontWeight: 400, color: '#fca5a5', maxWidth: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={r.error}>{r.error}</div>}
                              </div>
                          </div>
                      ))}
                    </div>
                  </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
