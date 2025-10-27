import './app.module.css';
import ChatInterface from './components/ChatInterface';
import logo from '../image.png';

export function App() {
  return (
    <div className="app">
      <div className="app-header" style={{ position: 'fixed', top: 0, right: 0, padding: '1rem', zIndex: 1000 }}>
        <img src={logo} alt="Logo" className="app-logo" style={{ height: '60px', width: 'auto' }} />
      </div>
      <ChatInterface />
    </div>
  );
}

export default App;