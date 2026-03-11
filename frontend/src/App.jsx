import { useState } from 'react';
import './index.css';
import Dashboard from './components/Dashboard';
import TrainingPage from './components/TrainingPage';
import UploadRawPage from './components/UploadRawPage';
import SupportVectorPlot from './components/SupportVectorPlot';
import TestCSVPage from './components/TestCSVPage';

/**
 * App – Root component mOC-iSVM Frontend
 * Kiến trúc: Global Scaler + SV Pruning + Euclidean Tie-break
 */
export default function App() {
  const [page, setPage] = useState('dashboard');

  const navItems = [
    { id: 'dashboard',  label: 'Dashboard',       icon: '📊' },
    { id: 'upload-raw', label: 'Train CSV',        icon: '🎓', badge: 'Pha 0→2' },
    { id: 'test',       label: 'Test CSV',         icon: '🧪', badge: 'Pha 3' },
    { id: 'sv-plot',    label: 'SV Map',           icon: '📌', badge: 'Debug' },
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>mOC-iSVM</h1>
          <span>MLOps Dashboard</span>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span className="icon">{item.icon}</span>
              {item.label}
              {item.badge && (
                <span style={{
                  marginLeft: 'auto', fontSize: 9, fontWeight: 700,
                  background: 'var(--accent-blue)', color: '#fff',
                  borderRadius: 4, padding: '2px 5px', letterSpacing: '0.05em',
                }}>{item.badge}</span>
              )}
            </button>
          ))}
        </nav>

        {/* Architecture Info */}
        <div style={{
          marginTop: 'auto',
          padding: '12px 16px',
          borderTop: '1px solid var(--border-color)',
          fontSize: 11,
          color: 'var(--text-muted)',
          lineHeight: 1.6,
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4, color: 'var(--text-secondary)' }}>
            Kiến trúc
          </div>
          <div>🌐 Global Scaler</div>
          <div>✂️ SV Age + Error Pruning</div>
          <div>📐 Euclidean Tie-break</div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {page === 'dashboard'  && <Dashboard />}
        {page === 'upload-raw' && <UploadRawPage />}
        {page === 'test'       && <TestCSVPage />}
        {page === 'sv-plot'    && <SupportVectorPlot />}
      </main>
    </div>
  );
}
