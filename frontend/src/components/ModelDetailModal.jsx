import { useState, useEffect } from 'react';
import { modelsAPI } from '../services/api';

/**
 * ModelDetailModal – Modal xem chi tiết dữ liệu bên trong một model .pkl
 */
export default function ModelDetailModal({ className, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  const rowsPerPage = 20;

  useEffect(() => {
    if (!className) return;
    setLoading(true);
    setError('');
    modelsAPI.detail(className, 500)
      .then(setData)
      .catch((err) => setError(err.message || 'Lỗi tải dữ liệu model'))
      .finally(() => setLoading(false));
  }, [className]);

  if (!className) return null;

  const totalPages = data ? Math.ceil(data.support_vectors_data.length / rowsPerPage) : 0;
  const pagedData = data ? data.support_vectors_data.slice(page * rowsPerPage, (page + 1) * rowsPerPage) : [];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div>
            <h2 className="modal-title">🔍 Chi tiết Model</h2>
            <p className="modal-subtitle">{className}</p>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {loading ? (
            <div className="modal-loading">
              <div className="spinner" />
              <p>Đang tải dữ liệu model...</p>
            </div>
          ) : error ? (
            <div className="alert alert-error">⚠️ {error}</div>
          ) : data ? (
            <>
              {/* Metadata Cards */}
              <div className="model-meta-grid">
                <div className="model-meta-card">
                  <div className="meta-icon">📛</div>
                  <div className="meta-label">Version</div>
                  <div className="meta-value">{data.version_name}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">⚙️</div>
                  <div className="meta-label">Kernel</div>
                  <div className="meta-value">{data.kernel}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">🎯</div>
                  <div className="meta-label">Nu</div>
                  <div className="meta-value">{data.nu}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">📐</div>
                  <div className="meta-label">Gamma</div>
                  <div className="meta-value">{data.gamma}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">📊</div>
                  <div className="meta-label">Support Vectors</div>
                  <div className="meta-value">{data.n_samples.toLocaleString()}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">🧬</div>
                  <div className="meta-label">Features</div>
                  <div className="meta-value">{data.n_features}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">🎯</div>
                  <div className="meta-label">Support Vectors</div>
                  <div className="meta-value">{data.support_vectors.toLocaleString()}</div>
                </div>
                <div className="model-meta-card">
                  <div className="meta-icon">{data.is_trained ? '✅' : '❌'}</div>
                  <div className="meta-label">Trạng thái</div>
                  <div className="meta-value">{data.is_trained ? 'Đã train' : 'Chưa train'}</div>
                </div>
              </div>

              {/* Training Data Table */}
              <div className="card" style={{ marginTop: 20 }}>
                <div className="card-header">
                  <div className="card-title">
                    📋 Support Vectors (Dữ liệu biên giới)
                    <span style={{ fontSize: 12, fontWeight: 400, marginLeft: 8, color: 'var(--text-muted)' }}>
                      (Hiển thị {data.showing_rows}/{data.total_rows} dòng)
                    </span>
                  </div>
                </div>
                <div className="table-wrapper" style={{ maxHeight: 400, overflow: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th style={{ width: 50 }}>#</th>
                        {Array.from({ length: data.n_features }, (_, i) => (
                          <th key={i}>F{i + 1}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {pagedData.map((row, idx) => (
                        <tr key={idx}>
                          <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                            {page * rowsPerPage + idx + 1}
                          </td>
                          {row.map((val, j) => (
                            <td key={j} style={{ fontFamily: 'monospace', fontSize: 12 }}>
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="modal-pagination">
                    <button
                      className="btn btn-outline btn-sm"
                      disabled={page === 0}
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                    >
                      ← Trước
                    </button>
                    <span className="pagination-info">
                      Trang {page + 1} / {totalPages}
                    </span>
                    <button
                      className="btn btn-outline btn-sm"
                      disabled={page >= totalPages - 1}
                      onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                    >
                      Sau →
                    </button>
                  </div>
                )}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
