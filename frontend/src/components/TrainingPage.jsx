import { useState } from 'react';
import { uploadAPI, trainAPI } from '../services/api';

/**
 * TrainingPage – Upload dữ liệu + cấu hình tham số → Train/Retrain
 *
 * Hỗ trợ 2 luồng:
 *   Luồng A (Phase 0): Nhập session_id từ upload CSV thô → tự động load danh sách lớp
 *   Luồng B (Classic): Upload trực tiếp 3 file CSV đã xử lý
 */
export default function TrainingPage() {
  // ---- Luồng B: 3 file CSV ----
  const [files, setFiles] = useState({ samples: null, features: null, classes: null });
  const [uploadResult, setUploadResult] = useState(null);

  // ---- Luồng A: session_id từ Phase 0 ----
  const [sessionId, setSessionId] = useState('');
  const [sessionClasses, setSessionClasses] = useState(null); // { unique_classes, class_counts, saved_paths }
  const [sessionLoading, setSessionLoading] = useState(false);

  // ---- Tham số train chung ----
  const [params, setParams] = useState({
    class_name: '',
    kernel: 'rbf',
    nu: 0.05,
    gamma: 'scale',
    retrain: false,
  });

  // ---- Kết quả train ----
  const [trainResult, setTrainResult] = useState(null);
  const [trainAllResults, setTrainAllResults] = useState([]); // Kết quả train tất cả lớp
  const [loading, setLoading] = useState({ upload: false, train: false, trainAll: false });
  const [alert, setAlert] = useState({ type: '', msg: '' });

  const setAlertMsg = (type, msg) => {
    setAlert({ type, msg });
    setTimeout(() => setAlert({ type: '', msg: '' }), 6000);
  };

  // -----------------------------------------------------------------
  // LUỒNG A – Load classes từ session_id (Phase 0)
  // -----------------------------------------------------------------
  const handleLoadSession = async () => {
    if (!sessionId.trim()) {
      setAlertMsg('error', 'Vui lòng nhập session_id!');
      return;
    }
    setSessionLoading(true);
    setSessionClasses(null);
    try {
      const res = await trainAPI.getClasses(sessionId.trim());
      setSessionClasses(res);
      if (res.unique_classes?.length) {
        setParams((p) => ({ ...p, class_name: res.unique_classes[0] }));
      }
      setAlertMsg('success', `✓ Tìm thấy ${res.n_classes} lớp trong session!`);
    } catch (err) {
      setAlertMsg('error', `Không tìm thấy session: ${err.message}`);
    } finally {
      setSessionLoading(false);
    }
  };

  // Lấy đường dẫn file từ nguồn đang dùng (session hay upload result)
  const getFilePaths = () => {
    if (sessionClasses) {
      const base = `data/processed/${sessionId}`;
      return {
        samples_file : `${base}/samples.csv`,
        features_file: `${base}/features.csv`,
        classes_file : `${base}/classes.csv`,
      };
    }
    return {
      samples_file : uploadResult?.saved_paths?.samples,
      features_file: uploadResult?.saved_paths?.features,
      classes_file : uploadResult?.saved_paths?.classes,
    };
  };

  // Danh sách lớp hiện tại (từ session hay upload)
  const availableClasses = sessionClasses?.unique_classes
    || uploadResult?.unique_classes
    || [];

  // -----------------------------------------------------------------
  // LUỒNG B – Upload 3 file CSV
  // -----------------------------------------------------------------
  const handleFileChange = (field) => (e) => {
    const file = e.target.files[0];
    if (file) setFiles((prev) => ({ ...prev, [field]: file }));
  };

  const handleUpload = async () => {
    if (!files.samples || !files.features || !files.classes) {
      setAlertMsg('error', 'Vui lòng chọn đủ 3 file CSV!');
      return;
    }
    setLoading((p) => ({ ...p, upload: true }));
    setUploadResult(null);
    setSessionClasses(null); // reset session khi dùng luồng B
    try {
      const res = await uploadAPI.uploadCSV(files.samples, files.features, files.classes);
      setUploadResult(res);
      if (!params.class_name && res.unique_classes?.length) {
        setParams((p) => ({ ...p, class_name: res.unique_classes[0] }));
      }
      setAlertMsg('success', `✓ Upload thành công! ${res.n_samples} mẫu, ${res.n_features} features.`);
    } catch (err) {
      setAlertMsg('error', `Upload thất bại: ${err.message}`);
    } finally {
      setLoading((p) => ({ ...p, upload: false }));
    }
  };

  // -----------------------------------------------------------------
  // Train một lớp
  // -----------------------------------------------------------------
  const handleTrain = async () => {
    const paths = getFilePaths();
    if (!paths.features_file) {
      setAlertMsg('error', 'Hãy load session hoặc upload dữ liệu trước!');
      return;
    }
    if (!params.class_name) {
      setAlertMsg('error', 'Vui lòng chọn tên lớp để huấn luyện!');
      return;
    }
    setLoading((p) => ({ ...p, train: true }));
    setTrainResult(null);
    try {
      const res = await trainAPI.train({
        class_name   : params.class_name,
        kernel       : params.kernel,
        nu           : parseFloat(params.nu),
        gamma        : params.gamma,
        retrain      : params.retrain,
        ...paths,
      });
      setTrainResult(res);
      setAlertMsg('success', res.message);
    } catch (err) {
      setAlertMsg('error', `Huấn luyện thất bại: ${err.message}`);
    } finally {
      setLoading((p) => ({ ...p, train: false }));
    }
  };

  // -----------------------------------------------------------------
  // Train tất cả lớp (tuần tự)
  // -----------------------------------------------------------------
  const handleTrainAll = async () => {
    if (!availableClasses.length) {
      setAlertMsg('error', 'Chưa có danh sách lớp. Hãy load session hoặc upload trước!');
      return;
    }
    const paths = getFilePaths();
    if (!paths.features_file) {
      setAlertMsg('error', 'Hãy load session hoặc upload dữ liệu trước!');
      return;
    }

    setLoading((p) => ({ ...p, trainAll: true }));
    setTrainAllResults([]);
    const results = [];

    for (const cls of availableClasses) {
      const clsResult = { class_name: cls, status: 'training', message: '' };
      setTrainAllResults([...results, clsResult]);
      try {
        const res = await trainAPI.train({
          class_name   : cls,
          kernel       : params.kernel,
          nu           : parseFloat(params.nu),
          gamma        : params.gamma,
          retrain      : params.retrain,
          ...paths,
        });
        clsResult.status    = 'success';
        clsResult.message   = `v${res.version_name} – ${res.n_samples} mẫu (${(res.inlier_ratio * 100).toFixed(1)}%)`;
        clsResult.version   = res.version_name;
      } catch (err) {
        clsResult.status  = 'error';
        clsResult.message = err.message;
      }
      results.push(clsResult);
      setTrainAllResults([...results]);
    }

    const success = results.filter((r) => r.status === 'success').length;
    setAlertMsg(
      success === results.length ? 'success' : 'error',
      `Train all hoàn tất: ${success}/${results.length} lớp thành công.`
    );
    setLoading((p) => ({ ...p, trainAll: false }));
  };

  const UploadZone = ({ field, label, icon }) => (
    <div
      className={`upload-zone ${files[field] ? 'uploaded' : ''}`}
      onClick={() => document.getElementById(`input-${field}`).click()}
    >
      <input id={`input-${field}`} type="file" accept=".csv" onChange={handleFileChange(field)} />
      <div className="upload-icon">{files[field] ? '✅' : icon}</div>
      {files[field] ? (
        <div className="filename">📄 {files[field].name}</div>
      ) : (
        <>
          <div className="upload-text">{label}</div>
          <div className="upload-hint">Kéo thả hoặc click để chọn .csv</div>
        </>
      )}
    </div>
  );

  const dataReady = !!(sessionClasses || uploadResult);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🎓 Huấn luyện Model</h1>
        <p className="page-subtitle">Upload dữ liệu và cấu hình tham số để huấn luyện mOC-iSVM</p>
      </div>

      {alert.msg && <div className={`alert alert-${alert.type}`}>{alert.msg}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* ====== Bước 1: Nguồn dữ liệu ====== */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">📁 Bước 1: Chọn nguồn dữ liệu</div>
          </div>

          {/* --- Luồng A: session_id từ Phase 0 --- */}
          <div style={{ marginBottom: 20 }}>
            <div className="form-label" style={{ fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: 12 }}>
              🔗 Luồng A – Từ session Phase 0 (Recommended)
            </div>
            <div className="form-group">
              <label className="form-label">Session ID (từ Upload CSV Thô)</label>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="form-control"
                  placeholder="ví dụ: abc123def456"
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                  style={{ flex: 1 }}
                />
                <button
                  className="btn btn-outline"
                  onClick={handleLoadSession}
                  disabled={sessionLoading || !sessionId.trim()}
                  style={{ whiteSpace: 'nowrap', padding: '0 14px' }}
                >
                  {sessionLoading ? <><span className="spinner" /> </> : '📂 Load'}
                </button>
              </div>
            </div>

            {sessionClasses && (
              <div className="alert alert-success" style={{ marginTop: 8, fontSize: 12 }}>
                <strong>✓ Session hợp lệ</strong>&nbsp;–&nbsp;
                {sessionClasses.n_classes} lớp: {sessionClasses.unique_classes.join(', ')}
              </div>
            )}
          </div>

          <div style={{
            textAlign: 'center', color: 'var(--text-muted)',
            fontSize: 12, margin: '0 0 16px', display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <div style={{ flex: 1, height: 1, background: 'var(--border-color)' }} />
            HOẶC
            <div style={{ flex: 1, height: 1, background: 'var(--border-color)' }} />
          </div>

          {/* --- Luồng B: Upload 3 file CSV --- */}
          <div>
            <div className="form-label" style={{ fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 12 }}>
              📤 Luồng B – Upload 3 file CSV trực tiếp
            </div>

            <div style={{ display: 'grid', gap: 10, marginBottom: 16 }}>
              <div>
                <div className="form-label" style={{ fontSize: 11 }}>📊 features.csv – Ma trận số đã scale</div>
                <UploadZone field="features" label="features.csv" icon="📊" />
              </div>
              <div>
                <div className="form-label" style={{ fontSize: 11 }}>🎯 classes.csv – Nhãn lớp</div>
                <UploadZone field="classes" label="classes.csv" icon="🎯" />
              </div>
              <div>
                <div className="form-label" style={{ fontSize: 11 }}>🆔 samples.csv – Định danh mẫu</div>
                <UploadZone field="samples" label="samples.csv" icon="🆔" />
              </div>
            </div>

            <button
              className="btn btn-outline"
              style={{ width: '100%' }}
              onClick={handleUpload}
              disabled={loading.upload || !files.samples || !files.features || !files.classes}
            >
              {loading.upload ? <><span className="spinner" /> Đang upload...</> : '⬆️ Validate & Upload'}
            </button>

            {uploadResult && (
              <div className="alert alert-success" style={{ marginTop: 12, fontSize: 12 }}>
                <strong>✓ Dữ liệu hợp lệ</strong>&nbsp;–&nbsp;
                {uploadResult.n_samples} mẫu · {uploadResult.n_features} features · {' '}
                {uploadResult.unique_classes?.join(', ')}
              </div>
            )}
          </div>
        </div>

        {/* ====== Bước 2: Tham số & Train ====== */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">⚙️ Bước 2: Cấu hình & Train</div>
          </div>

          <div className="form-group">
            <label className="form-label">Tên lớp (class_name) *</label>
            {availableClasses.length > 0 ? (
              <select
                className="form-control"
                value={params.class_name}
                onChange={(e) => setParams((p) => ({ ...p, class_name: e.target.value }))}
              >
                <option value="">-- Chọn lớp --</option>
                {availableClasses.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            ) : (
              <input
                className="form-control"
                placeholder="ví dụ: successful"
                value={params.class_name}
                onChange={(e) => setParams((p) => ({ ...p, class_name: e.target.value }))}
              />
            )}
          </div>

          <div className="form-group">
            <label className="form-label">Kernel</label>
            <select
              className="form-control"
              value={params.kernel}
              onChange={(e) => setParams((p) => ({ ...p, kernel: e.target.value }))}
            >
              <option value="rbf">RBF (Radial Basis Function)</option>
              <option value="linear">Linear</option>
              <option value="poly">Polynomial</option>
              <option value="sigmoid">Sigmoid</option>
            </select>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div className="form-group">
              <label className="form-label">Nu (0–1)</label>
              <input
                type="number"
                className="form-control"
                min="0.01" max="1" step="0.01"
                value={params.nu}
                onChange={(e) => setParams((p) => ({ ...p, nu: e.target.value }))}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Gamma</label>
              <select
                className="form-control"
                value={params.gamma}
                onChange={(e) => setParams((p) => ({ ...p, gamma: e.target.value }))}
              >
                <option value="scale">scale (tự động)</option>
                <option value="auto">auto</option>
                <option value="0.01">0.01</option>
                <option value="0.1">0.1</option>
                <option value="1.0">1.0</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={params.retrain}
                onChange={(e) => setParams((p) => ({ ...p, retrain: e.target.checked }))}
                style={{ width: 16, height: 16 }}
              />
              <span className="form-label" style={{ marginBottom: 0 }}>
                🔄 Incremental Retrain (gộp dữ liệu cũ + mới, tăng phiên bản)
              </span>
            </label>
          </div>

          {/* Nút Train */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <button
              className="btn btn-primary"
              onClick={handleTrain}
              disabled={loading.train || loading.trainAll || !dataReady || !params.class_name}
            >
              {loading.train ? <><span className="spinner" /> Đang train...</> : '🚀 Train lớp này'}
            </button>

            <button
              className="btn btn-outline"
              onClick={handleTrainAll}
              disabled={loading.train || loading.trainAll || !dataReady || availableClasses.length === 0}
              style={{ borderColor: 'var(--accent-purple)', color: 'var(--accent-purple)' }}
            >
              {loading.trainAll
                ? <><span className="spinner" /> Đang train tất cả...</>
                : `🏋️ Train tất cả (${availableClasses.length} lớp)`}
            </button>
          </div>

          {/* Kết quả train một lớp */}
          {trainResult && (
            <div style={{ marginTop: 20 }}>
              <div className="alert alert-success">{trainResult.message}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {[
                  ['Lớp', trainResult.class_name],
                  ['Phiên bản', trainResult.version_name],
                  ['Số mẫu', trainResult.n_samples?.toLocaleString()],
                  ['Inlier ratio', `${(trainResult.inlier_ratio * 100).toFixed(1)}%`],
                ].map(([label, value]) => (
                  <div key={label} style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '10px 12px',
                  }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--accent-cyan)' }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Kết quả train tất cả lớp */}
          {trainAllResults.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div className="form-label" style={{ marginBottom: 10 }}>📊 Tiến trình Train All</div>
              <div style={{ display: 'grid', gap: 6 }}>
                {trainAllResults.map((r) => (
                  <div key={r.class_name} style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    padding: '8px 12px',
                    background: r.status === 'success'
                      ? 'rgba(52,211,153,0.08)'
                      : r.status === 'error'
                      ? 'rgba(248,113,113,0.08)'
                      : 'rgba(255,255,255,0.03)',
                    borderRadius: 'var(--radius-sm)',
                    borderLeft: `3px solid ${r.status === 'success' ? 'var(--accent-green)' : r.status === 'error' ? '#f87171' : 'var(--border-color)'}`,
                  }}>
                    <span style={{ fontSize: 14 }}>
                      {r.status === 'success' ? '✅' : r.status === 'error' ? '❌' : '⏳'}
                    </span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)', minWidth: 100 }}>
                      {r.class_name}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', flex: 1 }}>
                      {r.status === 'training' ? 'Đang train...' : r.message}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
