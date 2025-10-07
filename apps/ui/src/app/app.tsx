import './app.module.css';
import VoiceChatbot from './components/VoiceChatbot';

export function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>CDCP AI</h1>
        <p>Fine-tuned models for CDCP analysis</p>
      </header>
      <main className="app-main">
        <VoiceChatbot />
      </main>
    </div>
  );
}

export default App;