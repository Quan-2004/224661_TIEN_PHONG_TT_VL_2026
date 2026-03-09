import { useState, useEffect } from 'react';
import { modelsAPI, trainAPI } from '../services/api';
import ModelDetailModal from './ModelDetailModal';

/**
 * Dashboard – Hiển thị trạng thái tất cả model từ global_manifest.xml
 */
export default function Dashboard() {
  const [modelsData, setModelsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const models = await modelsAPI.list();
      setModelsData(models);
    } catch (err) {
      setError('Không thể tải dữ liệu. Kiểm tra kết nối tới Backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const getVersionColor = (version) => {
    const num = parseInt(version?.split('-').pop() || '1');
    if (num >= 5) return 'badge-purple';
    if (num >= 3) return 'badge-blue';
    return 'badge-green';
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    try {
      return new Date(dateStr).toLocaleString('vi-VN', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch { return dateStr; }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">📊 Dashboard</h1>
        <p className="page-subtitle">Tổng quan hệ thống mOC-iSVM – Modified One-Class Incremental SVM</p>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-icon blue">🤖</div>
          <div className="stat-info">
            <div className="stat-value">{loading ? '…' : (modelsData?.total ?? 0)}</div>
            <div className="stat-label">Models đang hoạt động</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon purple">📦</div>
          <div className="stat-info">
            <div className="stat-value">
              {loading ? '…' : (modelsData?.models?.reduce((a, m) => a + parseInt(m.metadata?.n_samples || 0), 0) ?? 0)}
            </div>
            <div className="stat-label">Tổng mẫu huấn luyện</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon orange">🕐</div>
          <div className="stat-info">
            <div className="stat-value">
              {loading ? '…' : (modelsData?.last_updated !== 'N/A' ? formatDate(modelsData?.last_updated).split(',')[0] : 'N/A')}
            </div>
            <div className="stat-label">Cập nhật lần cuối</div>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}

      {/* Models Table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title">🗃️ Danh sách Model</div>
          <button className="btn btn-outline" style={{ fontSize: 13, padding: '6px 14px' }} onClick={fetchData}>
            🔄 Làm mới
          </button>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            <div className="spinner" style={{ margin: '0 auto 12px' }} />
            <div>Đang tải dữ liệu...</div>
          </div>
        ) : !modelsData?.models?.length ? (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🤖</div>
            <div>Chưa có model nào. Hãy vào trang <strong>Training</strong> để huấn luyện.</div>
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Tên lớp</th>
                  <th>Phiên bản</th>
                  <th>Kernel</th>
                  <th>Nu / Gamma</th>
                  <th>Support Vectors</th>
                  <th>Ngày train</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.models.map((model) => (
                  <tr key={model.class_name} className="clickable-row" onClick={() => setSelectedModel(model.class_name)} title="Click để xem chi tiết">
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                      🔍 {model.class_name}
                    </td>
                    <td>
                      <span className={`badge ${getVersionColor(model.version)}`}>
                        {model.version}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-blue">{model.metadata?.kernel || 'rbf'}</span>
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {model.metadata?.nu} / {model.metadata?.gamma}
                    </td>
                    <td style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>
                      {model.metadata?.n_samples ? Number(model.metadata.n_samples).toLocaleString() : '—'}
                    </td>
                    <td style={{ fontSize: 12 }}>{formatDate(model.metadata?.trained_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Model Detail Modal */}
      {selectedModel && (
        <ModelDetailModal
          className={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}
    </div>
  );
}
