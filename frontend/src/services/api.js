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
  /**
   * Phase 0 + 1 – Upload 1 file CSV thô → tiền xử lý tự động + huấn luyện tất cả các class
   * @param {File}   file   – File CSV thô
   * @param {Object} config – Lựa chọn xử lý và SVM params
   */
  autoTrain: (file, config = {}) => {
    const form = new FormData();
    form.append('file',         file);
    form.append('class_column', config.class_column   || 'state');
    form.append('id_columns',   config.id_columns     || '[]');
    form.append('drop_columns', config.drop_columns   || '[]');
    form.append('scale',        config.scale !== false ? 'true' : 'false');
    
    // Training params
    form.append('kernel',       config.kernel       || 'rbf');
    form.append('nu',           config.nu           || 0.1);
    form.append('gamma',        config.gamma        || 'scale');
    form.append('version_name', config.version_name || 'v1.0');

    return apiFetch('/auto-train', { method: 'POST', body: form });
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
