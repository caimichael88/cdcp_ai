import './ChatInterface.css';
import { useState, useRef, useCallback } from 'react';
import axios from 'axios';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const ChatInterface = () => {
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunkRef = useRef<Blob[]>([]);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  const API_PY_DOMAIN = (import.meta as any).env?.VITE_API_PY_DOMAIN || 'http://localhost:8001';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsProcessing(true);
    setError(null);

    try {
      const response = await axios.post(`${API_PY_DOMAIN}/query`, {
        query: inputText,
      });

      // Extract answer from the response
      const answerText = response.data?.answer || 'No response available';

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: answerText,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);

      // If audio is included, play it
      if (response.data?.audio_base64) {
        stopAudio();

        const audioData = atob(response.data.audio_base64);
        const audioArray = new Uint8Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
          audioArray[i] = audioData.charCodeAt(i);
        }
        const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioPlayerRef.current = audio;

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          setIsPlayingAudio(false);
        };

        setIsPlayingAudio(true);
        audio.play().catch((err) => {
          console.log('Error playing audio:', err);
          setIsPlayingAudio(false);
        });
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to get response. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunkRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunkRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunkRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone. Please ensure microphone permissions are granted.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  }, [isRecording]);

  const stopAudio = useCallback(() => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
      setIsPlayingAudio(false);
    }
  }, []);

  const processAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData();
      const uniqueFilename = `recording_${Date.now()}_${Math.random().toString(36).substring(2, 11)}.wav`;
      formData.append('file', audioBlob, uniqueFilename);

      const voiceResponse = await axios.post(`${API_PY_DOMAIN}/voice/voice-call`, formData);
      const response_data = voiceResponse.data;

      const userMessageObj: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: response_data.transcription || 'Voice message (transcription not available)',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessageObj]);

      const aiMessageObj: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response_data.response_text || 'AI response (text not available)',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessageObj]);

      if (response_data.audio_base64) {
        stopAudio();

        const audioData = atob(response_data.audio_base64);
        const audioArray = new Uint8Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
          audioArray[i] = audioData.charCodeAt(i);
        }
        const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioPlayerRef.current = audio;

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          setIsPlayingAudio(false);
        };

        setIsPlayingAudio(true);
        audio.play().catch((err) => {
          console.log('Error playing audio:', err);
          setError('Failed to play audio response');
          setIsPlayingAudio(false);
        });
      }
    } catch (err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="chat-interface">
      {messages.length === 0 ? (
        <div className="welcome-screen">
          <h1 className="welcome-title">What can I help with your CDCP question?</h1>
          <div className="input-container">
            <form onSubmit={handleSubmit}>
              <div className="input-wrapper">
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Ask anything"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isProcessing || isRecording}
                />
                <button
                  type="button"
                  className={`mic-button ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isProcessing}
                  title={isRecording ? 'Stop recording' : 'Start recording'}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" strokeWidth="2"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
                {isPlayingAudio && (
                  <button
                    type="button"
                    className="stop-audio-inline"
                    onClick={stopAudio}
                    title="Stop audio"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2"/>
                    </svg>
                  </button>
                )}
              </div>
            </form>
            {(isRecording || isProcessing || isPlayingAudio) && (
              <div className="status-indicator">
                {isRecording && <p className="status-text recording">🎙️ Recording... Click microphone to stop</p>}
                {isProcessing && <p className="status-text processing">⏳ Processing your request...</p>}
                {isPlayingAudio && <p className="status-text playing">🔊 Playing audio response...</p>}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="conversation-view">
          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.type}`}>
                <div className="message-content">{message.content}</div>
                <div className="message-time">{message.timestamp.toLocaleTimeString()}</div>
              </div>
            ))}
          </div>
          <div className="input-container-bottom">
            <form onSubmit={handleSubmit}>
              <div className="input-wrapper">
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Ask anything"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isProcessing || isRecording}
                />
                <button
                  type="button"
                  className={`mic-button ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isProcessing}
                  title={isRecording ? 'Stop recording' : 'Start recording'}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" strokeWidth="2"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
                {isPlayingAudio && (
                  <button
                    type="button"
                    className="stop-audio-inline"
                    onClick={stopAudio}
                    title="Stop audio"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2"/>
                    </svg>
                  </button>
                )}
              </div>
            </form>
            {(isRecording || isProcessing || isPlayingAudio) && (
              <div className="status-indicator">
                {isRecording && <p className="status-text recording">🎙️ Recording... Click microphone to stop</p>}
                {isProcessing && <p className="status-text processing">⏳ Processing your request...</p>}
                {isPlayingAudio && <p className="status-text playing">🔊 Playing audio response...</p>}
              </div>
            )}
          </div>
        </div>
      )}
      {error && <div className="error-notification">{error}</div>}
    </div>
  );
};

export default ChatInterface;
