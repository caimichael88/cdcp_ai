import './VoiceChatbot.css';
import React, { useState, useRef, useCallback} from 'react'
import axios from 'axios';
import { time } from 'console';


interface Message{
  id: string,
  type: 'user' | 'assistant',
  content: string,
  timestamp: Date;
}

const VoiceChatbot = () => {

  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([]);

  const mediaRecorderRef = useRef<MediaRecorder | null> (null);
  const audioChunkRef = useRef<Blob[]>([]);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  const API_PY_DOMAIN = (import.meta as any).env?.VITE_API_PY_DOMAIN || 'http://localhost:8001'; 

  const startRecording = useCallback(async () => {
    try{
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const medicaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = medicaRecorder; 
      audioChunkRef.current = [];

      medicaRecorder.ondataavailable = (event: BlobEvent) => {
        if(event.data.size > 0) {
          audioChunkRef.current.push(event.data);
        }
      };

      medicaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunkRef.current, { type: 'audio/wav'});
        await processAudio(audioBlob);

        stream.getTracks().forEach(track => track.stop());
      };

      medicaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch(err){
      console.error('Error starting recording: ', err);
      setError("Failed to access microphone. Please ensure microphone permissions are granted.");
    }
  }, []);

  const stopRecording = useCallback(()=> {
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
    try{
      const formData = new FormData();
      const uniqueFilename = `recording_${Date.now()}_${Math.random().toString(36).substring(2, 11)}.wav`;
      formData.append('file', audioBlob, uniqueFilename);

      //send audio to /voice/agent-with-text endpoint for LangGraph processing with text response

      const voiceResponse = await axios.post(`${API_PY_DOMAIN}/voice/agent-with-text`, formData);
      const response_data = voiceResponse.data;

      //Add user message with actual transcription
      const userMessageObj: Message = {
        id: Date.now().toString(),
        type: "user",
        content: response_data.transcription || "Voice message (transcription not available)",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessageObj]);

      //add AI response message with actual response text
      const aiMessageObj: Message = {
        id: (Date.now() +1).toString(),
        type: "assistant",
        content: response_data.response_text || "AI response (text not available)",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessageObj])

      //play the AI response audio data from base64
      if(response_data.audio_base64) {
        // Stop any currently playing audio
        stopAudio();

        const audioData = atob(response_data.audio_base64);
        const audioArray = new Uint8Array(audioData.length);
        for( let i=0; i< audioData.length; i++) {
          audioArray[i] = audioData.charCodeAt(i);
        }
        const audioBlob = new Blob([audioArray], {type: 'audio/wav'});
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioPlayerRef.current = audio;

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          setIsPlayingAudio(false);
        }

        setIsPlayingAudio(true);
        audio.play().catch(err => {
          console.log('Error playing audio: ', err);
          setError('Failed to play audio response');
          setIsPlayingAudio(false);
        })
      }

    }catch(err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    }finally {
      setIsProcessing(false);
    }
  };

  const clearConversation = () => {
    stopAudio();
    setMessages([]);
    setError(null);
  }

  return (
    <div className="voice-chatbot">
      <div className="chatbot-header">
        <h1>AI Research Assistant</h1>
        <p>Click the microphone to start a voice conversation</p>
      </div>
      <div className="conversation-display">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Welcome! Click the microphone button to start your conversation.</p>
          </div>
        ) : (
          <div className="conversation-history">
            {messages.map((message) => (
              <div key={message.id} className="conversation-turn">
                <div className={`${message.type}-message`}>
                  <strong>{message.type === 'user' ? 'You:' : 'Assistant:'}</strong> {message.content}
                </div>
                <div className="timestamp">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="controls">
        <button
          className={`record-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <span>Processing...</span>
          ) : isRecording ? (
            <span>🔴 Stop Talking</span>
          ) : (
            <span><img src='../assets/mic16.png'/> Start Talking</span>
          )}
        </button>

        {isPlayingAudio && (
          <button
            className="stop-audio-button"
            onClick={stopAudio}
          >
            ⏹️ Stop Audio
          </button>
        )}

        {messages.length > 0 && (
          <button
            className="clear-button"
            onClick={clearConversation}
            disabled={isRecording || isProcessing}
          >
            Clear Chat
          </button>
        )}
      </div>

      <div className="status">
        {isRecording && <p>🎙️ Recording... Click "Stop Recording" when done</p>}
        {isProcessing && <p>⏳ Processing your request...</p>}
        {isPlayingAudio && <p>🔊 Playing audio response...</p>}
        {!isRecording && !isProcessing && !isPlayingAudio && <p>💬 Ready for conversation</p>}
      </div>
    </div>
  );
}

export default VoiceChatbot;