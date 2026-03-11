/**
 * API Service Layer
 * ==================
 * Tất cả các lời gọi đến Backend FastAPI được tổng hợp ở đây.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ---------- Base fetch wrapper ----------

async function apiFetch(path, options = {}) {
  const headers = { ...options.headers };

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    const message = errorData.detail
      ? typeof errorData.detail === 'string'
        ? errorData.detail
        : JSON.stringify(errorData.detail)
      : `HTTP ${res.status}`;
    throw new Error(message);
  }

  return res.json();
}

// ---------- Models ----------

export const modelsAPI = {
  list:   ()                     => apiFetch('/models'),
  detail: (className, limit=100) => apiFetch(`/models/${encodeURIComponent(className)}?limit=${limit}`),
};

// ---------- Upload / Auto-Train ----------

export const uploadAPI = {
  /** Preview cột CSV để auto-detect */
  previewCSV: (file) => {
    const form = new FormData();
    form.append('file', file);
    return apiFetch('/upload-raw/preview', { method: 'POST', body: form });
  },

  /**
   * Phase 0 + 1/2 – Upload 1 CSV thô → GlobalScaler fit → huấn luyện tất cả class
   *
   * Lưu ý: Backend tự fit GlobalScaler trên toàn bộ X trước khi chia class.
   * Không cần truyền tham số scale nữa.
   *
   * config: { class_column, id_columns, drop_columns, kernel, nu, gamma,
   *           version_name, age_threshold, error_threshold, retrain }
   */
  autoTrain: (file, config = {}) => {
    const form = new FormData();
    form.append('file',            file);
    form.append('class_column',    config.class_column    || 'state');
    form.append('id_columns',      config.id_columns      || '[]');
    form.append('drop_columns',    config.drop_columns    || '[]');
    form.append('kernel',          config.kernel          || 'rbf');
    form.append('nu',              config.nu              ?? 0.05);
    form.append('gamma',           config.gamma           || 'scale');
    form.append('version_name',    config.version_name    || 'v1.0');
    form.append('age_threshold',   config.age_threshold   ?? 5);
    form.append('error_threshold', config.error_threshold ?? 0.5);
    form.append('retrain',         config.retrain !== false ? 'true' : 'false');
    return apiFetch('/auto-train', { method: 'POST', body: form });
  },
};

// ---------- Training ----------

export const trainAPI = {
  train:      (params)    => apiFetch('/train', { method: 'POST', body: JSON.stringify(params) }),
  getClasses: (sessionId) => apiFetch(`/train/classes?session_id=${sessionId}`),
};

// ---------- Predict ----------

export const predictAPI = {
  /** Thông tin model active + trạng thái GlobalScaler */
  info: () => apiFetch('/predict/info'),

  /**
   * Dự đoán từ mảng dữ liệu thô (chưa scale).
   * Backend tự load GlobalScaler và transform.
   */
  predict: (payload) =>
    apiFetch('/predict', { method: 'POST', body: JSON.stringify(payload) }),

  /**
   * Phase 3 – Upload test.csv → scale → predict → báo cáo
   * config: { class_column, min_margin, return_csv }
   */
  predictCSV: (file, config = {}) => {
    const form = new FormData();
    form.append('file',         file);
    form.append('class_column', config.class_column || '');
    form.append('min_margin',   config.min_margin   ?? 0.0);
    form.append('return_csv',   config.return_csv   ? 'true' : 'false');
    return apiFetch('/predict/csv', { method: 'POST', body: form });
  },

  reload: () => apiFetch('/predict/reload', { method: 'POST' }),
};
