import React, { useState, useEffect } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { predictAPI } from '../services/api';

/**
 * Màn hình Kiểm thử Inference (OvR)
 * Cho phép Mock dữ liệu hoặc test nhanh.
 */
export default function InferenceTestPage() {
  const [activeClasses, setActiveClasses] = useState([]);
  const [modelFeatures, setModelFeatures] = useState(0);
  const [testData, setTestData] = useState([]);
  
  const [results, setResults] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 1. Load active classes info on mount
  const loadInfo = async () => {
    try {
      const res = await predictAPI.info();
      setActiveClasses(res.active_classes || []);
      setModelFeatures(res.n_features || 10);
    } catch (err) {
      console.error(err);
      setError("Không thể tải thông tin mô hình.");
    }
  };

  useEffect(() => {
    loadInfo();
  }, []);

  // 2. Generate Synthetic Data
  const generateSyntheticData = () => {
    if (activeClasses.length === 0) {
      setError("Chưa có class nào được train để tạo dữ liệu giả lập.");
      return;
    }
    
    // Yêu cầu data bằng việc tạo random
    // Ở frontend ta sẽ giả lập random quanh 0 cho noise, và random quanh 1 cho các class
    // Tốt hơn: Nên call một API backend /predict/generate_mock (Nhưng vì ko tạo api backend ta tự fake ở frontend)
    const n_samples_per_class = 5;
    const n_noise = 10;
    
    let simulatedData = [];
    
    // Fake clusters cho mỗi class
    activeClasses.forEach((cls, clsIndex) => {
       for(let i=0; i<n_samples_per_class; i++) {
           let record = { true_class: cls, values: [] };
           for(let f=0; f<modelFeatures; f++) {
               // Tâm ngẫu nhiên dồn vào một góc cho dễ nhìn
               let center = (clsIndex + 1) * 2;
               let val = center + (Math.random() * 2 - 1);
               record.values.push(val);
           }
           simulatedData.push(record);
       }
    });
    
    // Fake noise
    for(let i=0; i<n_noise; i++) {
         let record = { true_class: 'unknown', values: [] };
         for(let f=0; f<modelFeatures; f++) {
              record.values.push((Math.random() - 0.5) * 20); // Noise phân bố rộng hơn
         }
         simulatedData.push(record);
    }
    
    setTestData(simulatedData);
    setResults(null);
    setPlotData(null);
    setError(null);
  };

  // 3. Run Inference
  const runInference = async () => {
    if (testData.length === 0) return;
    
    setLoading(true);
    setError(null);
    try {
      const payload = {
         data: testData.map(d => d.values),
         return_scores: true,
         return_plot_data: true,
         true_labels: testData.map(d => d.true_class)
      };
      
      const res = await predictAPI.predict(payload);
      setResults(res.results);
      setPlotData(res.plot_data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Lỗi khi chạy dự đoán.");
    } finally {
      setLoading(false);
    }
  };
  
  // Custom Scatter Tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="custom-tooltip" style={{ backgroundColor: '#fff', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}>
          {data.is_test_point ? (
            <>
              <p><strong>Test Sample IDX:</strong> {data.idx}</p>
              <p><strong>True Class:</strong> {data.true_class}</p>
              <p><strong>Predicted:</strong> <span style={{color: data.is_unknown ? 'black' : 'var(--accent-blue)', fontWeight: 'bold'}}>{data.predicted_class}</span></p>
            </>
          ) : (
             <p><strong>Support Vector</strong> ({data.class})</p>
          )}
        </div>
      );
    }
    return null;
  };

  // Format Scatter series
  let scatterSeries = [];
  const CATEGORY_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  if (plotData) {
     // 1. Group support vectors
     const svByClass = {};
     (plotData.support_vectors || []).forEach(sv => {
         if (!svByClass[sv.class]) svByClass[sv.class] = [];
         svByClass[sv.class].push(sv);
     });
     
     // Thêm series SV
     Object.keys(svByClass).forEach((cls, idx) => {
         scatterSeries.push({
            key: `sv-${cls}`,
            name: `SV - ${cls}`,
            data: svByClass[cls],
            fill: CATEGORY_COLORS[idx % CATEGORY_COLORS.length],
            shape: 'circle',
            opacity: 0.3
         });
     });
     
     // 2. Thêm Test points
     const testPoints = (plotData.test_points || []).map(tp => ({...tp, is_test_point: true}));
     
     // Tách test points thành unknown và predicted có class để tô màu
     const tpUnknown = testPoints.filter(tp => tp.is_unknown);
     const tpClassified = testPoints.filter(tp => !tp.is_unknown);
     
     if (tpUnknown.length > 0) {
        scatterSeries.push({
           key: `test-unknown`,
           name: `Predict: unknown`,
           data: tpUnknown,
           fill: '#000000',
           shape: 'cross',
           opacity: 1
        });
     }
     
     // Nhóm classified theo predicted class
     const tpByPred = {};
     tpClassified.forEach(tp => {
        if(!tpByPred[tp.predicted_class]) tpByPred[tp.predicted_class] = [];
        tpByPred[tp.predicted_class].push(tp);
     });
     
     Object.keys(tpByPred).forEach((cls) => {
         // Tìm màu khớp với màu SV
         let colorIdx = activeClasses.indexOf(cls);
         let color = colorIdx >= 0 ? CATEGORY_COLORS[colorIdx % CATEGORY_COLORS.length] : '#6b7280';
         
         scatterSeries.push({
            key: `test-pred-${cls}`,
            name: `Predict: ${cls}`,
            data: tpByPred[cls],
            fill: color,
            shape: 'cross',
            opacity: 1
         });
     });
  }

  return (
    <div className="page-container" style={{ padding: '2rem' }}>
      <header className="page-header" style={{ marginBottom: '2rem' }}>
        <h2>Kiểm thử Inference (OvR)</h2>
        <p>Sinh dữ liệu kiểm thử giả lập, đưa qua mô hình và xem hình ảnh Scatter Plot thông qua bộ giảm chiều PCA.</p>
      </header>

      {error && <div className="error-banner" style={{marginBottom: '1rem', padding: '1rem', background: '#fee2e2', color: '#dc2626', borderRadius: '8px'}}>{error}</div>}

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
        <button className="btn btn-secondary" onClick={generateSyntheticData}>
          1. Sinh dữ liệu Test Mock
        </button>
        <button 
           className="btn btn-primary" 
           onClick={runInference}
           disabled={testData.length === 0 || loading}
        >
          {loading ? 'Đang phân tích...' : '2. Chạy Prediction'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Left Column: Result Table */}
        <div className="card" style={{ padding: '1.5rem', background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}>
          <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Bảng Dự đoán</h3>
          
          {!results && testData.length > 0 && (
             <p style={{color: '#6b7280'}}>Đã sinh {testData.length} mẫu dữ liệu. Nhấn Chạy Prediction để xem kết quả.</p>
          )}
          
          {results && (
            <div style={{ overflowX: 'auto', maxHeight: '500px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                    <th style={{ padding: '0.75rem 0.5rem' }}>ID</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Dữ liệu Sinh</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Dự đoán</th>
                    <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Max Score</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, idx) => {
                    // Find actual test point class for this idx
                    const trueClass = testData[idx]?.true_class || 'N/A';
                    const isCorrect = trueClass === r.predicted_class;
                    
                    return (
                      <tr key={idx} style={{ borderBottom: '1px solid #e5e7eb' }}>
                        <td style={{ padding: '0.75rem 0.5rem' }}>#{idx}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{trueClass}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          <span style={{ 
                             display: 'inline-block', 
                             padding: '2px 8px', 
                             borderRadius: '9999px',
                             fontSize: '0.8rem',
                             fontWeight: 600,
                             backgroundColor: r.predicted_class === 'unknown' ? '#f3f4f6' : 'rgba(59, 130, 246, 0.1)',
                             color: r.predicted_class === 'unknown' ? '#374151' : 'var(--accent-blue)',
                             border: `1px solid ${isCorrect ? '#10b981' : '#f43f5e'}`
                          }}>
                            {r.predicted_class}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>
                          {r.confidence.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#4b5563' }}>
                * Viền <span style={{color: '#10b981'}}>xanh lá</span>: Đúng với dữ liệu sinh mock. Viền <span style={{color: '#f43f5e'}}>đỏ</span>: Sai. Thường test mock random sẽ bị sai lệch.
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Scatter Plot */}
        <div className="card" style={{ padding: '1.5rem', background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}>
          <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Scatter Plot (PCA 2D)</h3>
          
          {loading && <div className="loading-spinner" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Đang phân tích PCA...</div>}
          
          {!loading && !plotData && (
             <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f9fafb', border: '1px dashed #d1d5db', borderRadius: '8px', color: '#9ca3af' }}>
                Chưa có dữ liệu biểu đồ
             </div>
          )}
          
          {!loading && plotData && (
             <div style={{ height: '400px', width: '100%' }}>
               <ResponsiveContainer width="100%" height="100%">
                 <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                   <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                   <XAxis type="number" dataKey="x" name="PCA 1" tick={{fontSize: 12}} />
                   <YAxis type="number" dataKey="y" name="PCA 2" tick={{fontSize: 12}} />
                   <Tooltip content={<CustomTooltip />} />
                   <Legend wrapperStyle={{ fontSize: '12px' }} />
                   
                   {scatterSeries.map((series) => (
                      <Scatter 
                        key={series.key} 
                        name={series.name} 
                        data={series.data} 
                        fill={series.fill} 
                        shape={series.shape}
                      >
                         {series.data.map((entry, idx) => (
                           <Cell key={idx} fill={series.fill} opacity={series.opacity} strokeWidth={series.shape === 'cross' ? 2 : 1} stroke={series.shape === 'cross' ? series.fill : 'none'}/>
                         ))}
                      </Scatter>
                   ))}
                 </ScatterChart>
               </ResponsiveContainer>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}
