"""
Ports / interfaces for the TTS pipeline

- **TTSEngine**: top-level service interface the router depends on (single entrypoint).
- **NeuralTextToMel**: the *Tacotron2-style* text→mel component we specifically mean
  when we say "Text2Mel" in this project.
- **NeuralVocoder**: the mel→waveform component (HiFi-GAN/WaveGlow).
- **SystemTTSEngine**: optional interface for OS/system TTS (e.g., pyttsx3) so we
  can swap engines behind the same `TTSEngine` contract.
- **CoquiTTSEngine

"""