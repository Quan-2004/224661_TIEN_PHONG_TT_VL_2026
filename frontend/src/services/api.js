/**
 * API Service Layer
 * ==================
 * Tất cả các lời gọi đến Backend FastAPI được tổng hợp ở đây.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ---------- Base fetch wrapper ----------

async function apiFetch(path, options = {}) {
  const headers = {
    ...options.headers,
  };

  // Chỉ thêm Content-Type nếu không phải FormData
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

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
  /** Lấy danh sách tất cả model từ XML */
  list: () => apiFetch('/models'),

  /** Xem chi tiết model + training data */
  detail: (className, limit = 100) =>
    apiFetch(`/models/${encodeURIComponent(className)}?limit=${limit}`),
};

// ---------- Upload ----------

export const uploadAPI = {
  /**
   * Preview cột CSV – gọi trước khi xử lý để auto-detect cột
   * @param {File} file
   */
  previewCSV: (file) => {
    const form = new FormData();
    form.append('file', file);
    return apiFetch('/upload-raw/preview', { method: 'POST', body: form });
  },

  /**
   * Upload 3 file CSV đã xử lý
   */
  uploadCSV: (samplesFile, featuresFile, classesFile) => {
    const form = new FormData();
    form.append('samples',  samplesFile);
    form.append('features', featuresFile);
    form.append('classes',  classesFile);

    return apiFetch('/upload', { method: 'POST', body: form });
  },

  /**
   * Phase 0 – Upload 1 file CSV thô → tiền xử lý tự động
   * @param {File}   file   – File CSV thô
   * @param {Object} config – { class_column, id_columns, drop_columns, scale }
   */
  uploadRaw: (file, config = {}) => {
    const form = new FormData();
    form.append('file',         file);
    form.append('class_column', config.class_column   || 'state');
    form.append('id_columns',   config.id_columns     || '[]');
    form.append('drop_columns', config.drop_columns   || '[]');
    form.append('scale',        config.scale !== false ? 'true' : 'false');

    return apiFetch('/upload-raw', { method: 'POST', body: form });
  },
};

// ---------- Training ----------

export const trainAPI = {
  /**
   * Kích hoạt training workflow
   * @param {Object} params
   */
  train: (params) =>
    apiFetch('/train', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  /** Lịch sử huấn luyện */
  history: (limit = 20) => apiFetch(`/train/history?limit=${limit}`),

  /**
   * Lấy danh sách lớp từ session đã upload CSV thô (Phase 0)
   * @param {string} sessionId
   */
  getClasses: (sessionId) => apiFetch(`/train/classes?session_id=${sessionId}`),
};

// ---------- Predict ----------

export const predictAPI = {
  /** Lấy thông tin các lớp đang active */
  info: () => apiFetch('/predict/info'),

  /** Chạy suy luận (OvR) với tuỳ chọn trả về plot data */
  predict: (payload) => 
    apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
};
