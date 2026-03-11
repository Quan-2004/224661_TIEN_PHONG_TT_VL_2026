import { useState } from 'react';
import './index.css';
import Dashboard from './components/Dashboard';
import TrainingPage from './components/TrainingPage';
import UploadRawPage from './components/UploadRawPage';
import TestCsvPage from './components/TestCsvPage';

/**
 * App – Root component mOC-iSVM Frontend
 * Không yêu cầu đăng nhập – truy cập thẳng ứng dụng.
 */
export default function App() {
  const [page, setPage] = useState('dashboard');

  const navItems = [
    { id: 'dashboard',  label: 'Dashboard',          icon: '📊' },
    { id: 'upload-raw', label: 'Upload CSV Thô',     icon: '📥', badge: 'Phase 0' },
    { id: 'train',      label: 'Huấn luyện',         icon: '🎓' },
    { id: 'test-csv',   label: 'Kiểm thử CSV',       icon: '🔬', badge: 'Mới' },
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
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {page === 'dashboard'  && <Dashboard />}
        {page === 'upload-raw' && <UploadRawPage />}
        {page === 'train'      && <TrainingPage />}
        {page === 'test-csv'   && <TestCsvPage />}
      </main>
    </div>
  );
}
