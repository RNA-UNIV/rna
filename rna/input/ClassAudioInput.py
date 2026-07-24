from IPython.display import HTML, Audio, display
from google.colab.output import eval_js
from base64 import b64decode
import tempfile
import numpy as np
from scipy.io import wavfile

audio_html = """
<style>
  .audio-panel {
    border: 2px dashed #ccc;
    padding: 12px;
    border-radius: 6px;
    background-color: #f9f9f9;
    font-family: Arial, sans-serif;
    max-width: 100%;
    min-width: 600px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .audio-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 0 4px;
  }

  .audio-status {
    font-size: 0.85rem;
    color: #666;
    font-weight: bold;
    white-space: nowrap;
    min-width: 120px;
  }

  .audio-status.timer {
    color: #d63420;
  }

  .audio-status.recording {
    color: #d63420;
    animation: pulse-text 1s infinite;
  }

  @keyframes pulse-text {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .audio-properties {
    display: flex;
    gap: 16px;
    font-size: 0.75rem;
    color: #888;
    align-items: center;
    flex-wrap: wrap;
  }

  .audio-properties span {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .property-value {
    color: #444;
    font-weight: 600;
  }

  .progress-indicator {
    font-size: 0.8rem;
    font-weight: bold;
    color: #28a745;
    min-width: 40px;
    text-align: right;
  }

  .waveform-container {
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
    padding: 4px;
    width: 100%;
    position: relative;
  }

  .waveform-canvas {
    width: 100%;
    height: 80px;
    display: block;
    cursor: pointer;
    border-radius: 2px;
  }

  /* Línea de tiempo superpuesta */
  .playhead {
    position: absolute;
    top: 4px;
    bottom: 4px;
    width: 2px;
    background-color: #ff0000;
    pointer-events: none;
    display: none;
    z-index: 10;
    transition: left 0.05s linear;
  }

  .playhead::before {
    content: '▲';
    position: absolute;
    top: -12px;
    left: 50%;
    transform: translateX(-50%);
    color: #ff0000;
    font-size: 10px;
  }

  .controls-row {
    display: flex;
    gap: 6px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: nowrap;
  }

  .controls-left {
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: nowrap;
  }

  .controls-right {
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: nowrap;
  }

  .audio-btn {
    padding: 4px 10px;
    font-size: 0.75rem;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.2s;
    height: 28px;
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }

  .audio-btn:hover:enabled {
    opacity: 0.8;
    transform: scale(0.98);
  }

  .audio-btn:disabled {
    cursor: not-allowed;
    opacity: 0.4;
  }

  .btn-record {
    background-color: #e05a3a;
    color: white;
    min-width: 70px;
  }

  .btn-record.recording {
    background-color: #d63420;
    animation: pulse-btn 1s infinite;
  }

  @keyframes pulse-btn {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .btn-play {
    background-color: #1fa3ec;
    color: white;
    min-width: 60px;
  }

  .btn-accept {
    background-color: #28a745;
    color: white;
    min-width: 50px;
  }

  .btn-cancel {
    background-color: #6c757d;
    color: white;
    min-width: 50px;
  }

  .btn-export {
    background-color: #6f42c1;
    color: white;
    min-width: 50px;
  }

  .volume-control {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .volume-slider {
    width: 60px;
    height: 4px;
    cursor: pointer;
  }

  .volume-label {
    font-size: 0.7rem;
    color: #666;
    min-width: 28px;
    text-align: center;
  }

  .error {
    color: #d63420;
  }

  .success {
    color: #28a745;
  }

  @media (max-width: 768px) {
    .audio-panel {
      min-width: auto;
      padding: 8px;
    }

    .audio-header {
      flex-wrap: wrap;
    }

    .audio-properties {
      gap: 8px;
      font-size: 0.7rem;
    }

    .controls-row {
      flex-wrap: wrap;
    }

    .audio-btn {
      font-size: 0.7rem;
      padding: 3px 8px;
      height: 24px;
    }

    .volume-slider {
      width: 40px;
    }
  }
</style>

<div class="audio-panel">
  <!-- FILA 1: Estado + Propiedades -->
  <div class="audio-header">
    <div class="audio-status" id="status">✅ Listo</div>
    <div class="audio-properties">
      <span>📊 <span class="property-value" id="sample-rate">44.1</span>kHz</span>
      <span>📁 <span class="property-value" id="file-size">0</span>KB</span>
      <span>🎚️ <span class="property-value" id="peak-level">-∞</span>dB</span>
      <span>⏱️ <span class="property-value" id="timer-display">00:00</span></span>
      <span class="progress-indicator" id="progress">0%</span>
    </div>
  </div>

  <!-- FILA 2: Waveform con línea de tiempo -->
  <div class="waveform-container">
    <canvas id="waveform-canvas" class="waveform-canvas"></canvas>
    <div class="playhead" id="playhead"></div>
  </div>

  <!-- FILA 3: Controles -->
  <div class="controls-row">
    <div class="controls-left">
      <button class="audio-btn btn-record" id="btn-record">🎤 Grabar</button>
      <button class="audio-btn btn-play" id="btn-play" disabled>▶ Reproducir</button>
      <button class="audio-btn btn-accept" id="btn-accept" disabled>✓</button>
      <button class="audio-btn btn-cancel" id="btn-cancel">✕</button>
    </div>
    <div class="controls-right">
      <div class="volume-control">
        <span style="font-size:0.75rem;">🎚️ Nivel</span>
        <input type="range" id="volume-slider" class="volume-slider" min="0" max="100" value="100">
        <span class="volume-label" id="volume-label">80%</span>
      </div>
      <button class="audio-btn btn-export" id="btn-export" disabled>💾</button>
    </div>
  </div>
</div>

<script>
(async function() {
  const statusEl = document.getElementById('status');
  const btnRecord = document.getElementById('btn-record');
  const btnPlay = document.getElementById('btn-play');
  const btnAccept = document.getElementById('btn-accept');
  const btnCancel = document.getElementById('btn-cancel');
  const btnExport = document.getElementById('btn-export');
  const waveformCanvas = document.getElementById('waveform-canvas');
  const volumeSlider = document.getElementById('volume-slider');
  const volumeLabel = document.getElementById('volume-label');
  const timerDisplay = document.getElementById('timer-display');
  const progressDisplay = document.getElementById('progress');
  const sampleRateDisplay = document.getElementById('sample-rate');
  const fileSizeDisplay = document.getElementById('file-size');
  const peakLevelDisplay = document.getElementById('peak-level');
  const playhead = document.getElementById('playhead');

  const waveCtx = waveformCanvas.getContext('2d');

  function resizeCanvas() {
    waveformCanvas.width = waveformCanvas.offsetWidth || 800;
    waveformCanvas.height = 80;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  let mediaRecorder;
  let audioContext;
  let analyser;
  let micStream;
  let audioChunks = [];
  let audioBlob = null;
  let audioUrl = null;
  let audioElement = null;
  let startTime = 0;
  let timerInterval = null;
  let animationId = null;
  let dataArray = null;
  let isPlayingBack = false;
  let cancelled = false;
  let recordingStartTime = 0;
  let peakLevel = -Infinity;

  // ===== Datos permanentes del audio =====
  let originalAudioBuffer = null;

  // Waveform (min/max)
  let waveformData = [];
  let waveformSamples = 400;

  // Nivel final elegido por el usuario (0..1)
  let outputLevel = 0.80;

  // Ganancia necesaria para normalizar el audio
  let normalizationGain = 1.0;

  // Pico original del audio
  let originalPeak = 0;

  function updateStatus(msg, className = '') {
    statusEl.textContent = msg;
    statusEl.className = 'audio-status ' + className;
  }

  function formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  function updateTimer(ms) {
    timerDisplay.textContent = formatTime(ms);
  }

  function updateProgress(percent) {
    progressDisplay.textContent = Math.round(percent) + '%';
  }

  function updatePeakLevel(level) {
    if (level > peakLevel) peakLevel = level;
    const db = peakLevel > 0 ? 20 * Math.log10(peakLevel) : -Infinity;
    peakLevelDisplay.textContent = db > -Infinity ? db.toFixed(1) : '-∞';
  }

  function updateFileSize(bytes) {
    if (bytes < 1024) {
      fileSizeDisplay.textContent = bytes + 'B';
    } else if (bytes < 1024 * 1024) {
      fileSizeDisplay.textContent = (bytes / 1024).toFixed(1) + 'KB';
    } else {
      fileSizeDisplay.textContent = (bytes / (1024 * 1024)).toFixed(2) + 'MB';
    }
  }

  function startTimer() {
    recordingStartTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = Date.now() - recordingStartTime;
      updateTimer(elapsed);
      const maxDuration = 60000;
      const progress = Math.min((elapsed / maxDuration) * 100, 100);
      updateProgress(progress);
      updateStatus('🔴 Grabando...', 'recording');
    }, 100);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function drawWaveform(cursorPosition = null) {
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    
    // Fondo
    waveCtx.clearRect(0, 0, w, h);
    waveCtx.fillStyle = "#fafafa";
    waveCtx.fillRect(0, 0, w, h);
    
    // Línea central (silencio)
    const centerY = h / 2;
    
    waveCtx.strokeStyle = "#dddddd";
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(0, centerY);
    waveCtx.lineTo(w, centerY);
    waveCtx.stroke();
    
    // Verificar si hay datos de waveform
    if (!waveformData || waveformData.length === 0) {
        waveCtx.fillStyle = "#888";
        waveCtx.font = "13px Arial";
        waveCtx.textAlign = "center";
        waveCtx.textBaseline = "middle";
        waveCtx.fillText("🎤 Esperando audio...", w / 2, centerY);
        return;
    }
    
    const dx = w / waveformData.length;
    
    // Dibujar waveform
    for (let i = 0; i < waveformData.length; i++) {
        const sample = waveformData[i];
        const min = sample.min * normalizationGain * outputLevel;
        const max = sample.max * normalizationGain * outputLevel;
        const yMin = Math.min(centerY - min * (centerY - 2), h - 2);
        const yMax = Math.max(centerY - max * (centerY - 2), 2);
        const x = Math.round(i * dx);
        
        waveCtx.strokeStyle = "#1f8ef1";
        waveCtx.lineWidth = 1;
        waveCtx.beginPath();
        waveCtx.moveTo(x, yMax);
        waveCtx.lineTo(x, yMin);
        waveCtx.stroke();
    }
    
    // Dibujar cursor de reproducción si se proporciona posición
    if (cursorPosition !== null && cursorPosition >= 0 && cursorPosition <= 1) {
        const x = cursorPosition * w;
        
        // Actualizar playhead HTML
        if (playhead) {
            playhead.style.display = 'block';
            playhead.style.left = (cursorPosition * 100) + '%';
        }
        
        // Dibujar línea roja en el canvas
        waveCtx.strokeStyle = "#ff3030";
        waveCtx.lineWidth = 2;
        waveCtx.beginPath();
        waveCtx.moveTo(x, 0);
        waveCtx.lineTo(x, h);
        waveCtx.stroke();
        
        // Dibujar triángulo en la parte superior
        waveCtx.fillStyle = "#ff3030";
        waveCtx.beginPath();
        waveCtx.moveTo(x, 0);
        waveCtx.lineTo(x - 6, 10);
        waveCtx.lineTo(x + 6, 10);
        waveCtx.closePath();
        waveCtx.fill();
    } else {
        // Ocultar playhead si no hay posición
        if (playhead) {
            playhead.style.display = 'none';
        }
    }
  }

  function generateFullWaveform(audioBuffer) {

      if (!audioBuffer)
          return;

      originalAudioBuffer = audioBuffer;

      const channelData = audioBuffer.getChannelData(0);
      const totalSamples = channelData.length;

      waveformData = [];
      originalPeak = 0;

      if (totalSamples === 0) {
          normalizationGain = 1.0;
          return;
      }

      const samplesPerBlock = Math.ceil(totalSamples / waveformSamples);

      for (let block = 0; block < waveformSamples; block++) {

          const start = block * samplesPerBlock;

          if (start >= totalSamples)
              break;

          const end = Math.min(start + samplesPerBlock, totalSamples);

          let min = Number.POSITIVE_INFINITY;
          let max = Number.NEGATIVE_INFINITY;
          let sumSquares = 0;

          for (let i = start; i < end; i++) {

              const s = channelData[i];

              if (s < min) min = s;
              if (s > max) max = s;

              const a = Math.abs(s);
              if (a > originalPeak)
                  originalPeak = a;

              sumSquares += s * s;
          }

          waveformData.push({
              min: min,
              max: max,
              rms: Math.sqrt(sumSquares / (end - start))
          });
      }

      normalizationGain =
          originalPeak > 0 ? 1.0 / originalPeak : 1.0;
  }

  function animateWaveform() {
    if (!analyser || mediaRecorder.state !== 'recording') {
        return;
    }
    
    if (!dataArray) {
        dataArray = new Uint8Array(analyser.frequencyBinCount);
    }
    
    analyser.getByteTimeDomainData(dataArray);
    
    // Suavizado: promediar muestras para movimiento más lento
    const smoothData = new Uint8Array(200);
    const step = Math.floor(dataArray.length / smoothData.length);
    
    for (let i = 0; i < smoothData.length; i++) {
        let sum = 0;
        const start = i * step;
        const end = Math.min(start + step, dataArray.length);
        for (let j = start; j < end; j++) {
            sum += dataArray[j];
        }
        smoothData[i] = sum / (end - start);
    }
    
    // Dibujar waveform con los datos en vivo (sin cursor de reproducción)
    drawWaveformFromData(smoothData);
    
    // Actualizar nivel de pico
    let maxVal = 0;
    for (let i = 0; i < smoothData.length; i++) {
        const v = Math.abs((smoothData[i] - 128) / 128.0);
        if (v > maxVal) maxVal = v;
    }
    updatePeakLevel(maxVal);
    
    if (mediaRecorder.state === 'recording') {
        setTimeout(() => {
            animationId = requestAnimationFrame(animateWaveform);
        }, 66); // ~15 fps
    }
  }

  // Función auxiliar para dibujar waveform desde datos en vivo
  function drawWaveformFromData(data) {
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    
    waveCtx.clearRect(0, 0, w, h);
    waveCtx.fillStyle = "#fafafa";
    waveCtx.fillRect(0, 0, w, h);
    
    const centerY = h / 2;
    const dx = w / data.length;
    
    waveCtx.strokeStyle = "#1f8ef1";
    waveCtx.lineWidth = 1;
    
    for (let i = 0; i < data.length; i++) {
        const normalized = (data[i] - 128) / 128.0;
        const y = centerY - normalized * (centerY - 2);
        
        const x = Math.round(i * dx);
        waveCtx.beginPath();
        waveCtx.moveTo(x, centerY);
        waveCtx.lineTo(x, y);
        waveCtx.stroke();
    }
  }

  function animatePlayback() {
    if (!audioElement || audioElement.paused) {
        return;
    }
    
    const duration = audioElement.duration;
    if (!duration || duration <= 0) {
        return;
    }
    
    const currentTime = audioElement.currentTime;
    const position = currentTime / duration;
    
    // Actualizar tiempo mostrado
    updateTimer(currentTime * 1000);
    
    // Actualizar barra de progreso
    updateProgress(position * 100);
    
    // Redibujar waveform con cursor
    drawWaveform(position);
    
    // Continuar animación
    if (!audioElement.paused) {
        requestAnimationFrame(animatePlayback);
    }
  }

  function updateVolumeLabel() {

      outputLevel = volumeSlider.value / 100.0;

      volumeLabel.textContent =
          Math.round(outputLevel * 100) + "%";

      // Redibujar el waveform con el nuevo nivel
      if (audioElement &&
          !audioElement.paused &&
          audioElement.duration > 0) {

          drawWaveform(
              audioElement.currentTime /
              audioElement.duration
          );
      }
      else {

          drawWaveform();
      }

      // Regenerar el WAV que se devolverá a Python
      if (originalAudioBuffer) {

          audioBlob = audioBufferToWav(originalAudioBuffer);

          if (audioUrl)
              URL.revokeObjectURL(audioUrl);

          audioUrl = URL.createObjectURL(audioBlob);

          audioElement = new Audio(audioUrl);

          updateFileSize(audioBlob.size);
      }
  }

  volumeSlider.oninput = updateVolumeLabel;

  // Función para convertir AudioBuffer a WAV
  function audioBufferToWav(audioBuffer) {

      const numChannels = audioBuffer.numberOfChannels;
      const sampleRate = audioBuffer.sampleRate;
      const numFrames = audioBuffer.length;

      const bytesPerSample = 2;
      const blockAlign = numChannels * bytesPerSample;
      const dataSize = numFrames * blockAlign;

      const buffer = new ArrayBuffer(44 + dataSize);
      const view = new DataView(buffer);

      function writeString(offset, string) {
          for (let i = 0; i < string.length; i++)
              view.setUint8(offset + i, string.charCodeAt(i));
      }

      // =====================
      // Cabecera WAV
      // =====================

      writeString(0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(8, "WAVE");

      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * blockAlign, true);
      view.setUint16(32, blockAlign, true);
      view.setUint16(34, 16, true);

      writeString(36, "data");
      view.setUint32(40, dataSize, true);

      // =====================
      // Audio
      // =====================

      const effectiveGain =
          normalizationGain * outputLevel;

      let offset = 44;

      for (let i = 0; i < numFrames; i++) {

          for (let ch = 0; ch < numChannels; ch++) {

              let sample =
                  audioBuffer
                      .getChannelData(ch)[i] *
                  effectiveGain;

              // limitar por seguridad
              sample = Math.max(-1, Math.min(1, sample));

              const pcm =
                  sample < 0
                      ? sample * 32768
                      : sample * 32767;

              view.setInt16(offset, pcm, true);

              offset += 2;
          }
      }

      return new Blob(
          [buffer],
          { type: "audio/wav" }
      );
  }

  let resolveAudio, rejectAudio;
  window.audioPromise = new Promise((resolve, reject) => {
    resolveAudio = resolve;
    rejectAudio = reject;
  });

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false
      } 
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512; // Más muestras para mejor resolución

    const source = audioContext.createMediaStreamSource(micStream);
    source.connect(analyser);

    mediaRecorder = new MediaRecorder(micStream);

    btnRecord.onclick = async () => {
      if (mediaRecorder.state === 'inactive') {
        cancelled = false;
        audioChunks = [];
        audioBlob = null;
        audioUrl = null;
        peakLevel = -Infinity;
        fullWaveformData = null;
        playhead.style.display = 'none';

        mediaRecorder.start(100);
        btnRecord.classList.add('recording');
        btnRecord.textContent = '⏹ Parar';
        btnPlay.disabled = true;
        btnAccept.disabled = true;
        btnExport.disabled = true;
        btnCancel.disabled = false;
        startTimer();
        animateWaveform();
        updateStatus('🔴 Grabando...', 'recording');
      } else {
        mediaRecorder.stop();
        btnRecord.classList.remove('recording');
        btnRecord.textContent = '🎤 Grabar';
        stopTimer();
        if (animationId) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
      }
    };

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      if (!cancelled) {
         drawWaveform(null); // Asegurar que no hay cursor
      }
      
      audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

      const reader = new FileReader();
      reader.onload = (e) => {
        const arrayBuffer = e.target.result;
        audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
          audioBlob = audioBufferToWav(audioBuffer);
          audioUrl = URL.createObjectURL(audioBlob);
          btnPlay.disabled = false;
          btnAccept.disabled = false;
          btnExport.disabled = false;

          // Actualizar propiedades
          sampleRateDisplay.textContent = (audioBuffer.sampleRate / 1000).toFixed(1);
          updateFileSize(audioBlob.size);
          updateStatus('✅ Grabación lista', 'success');
          updateProgress(100);

          // GENERAR WAVEFORM COMPLETO
          originalAudioBuffer = audioBuffer;

          generateFullWaveform(audioBuffer);

          drawWaveform();

          // Actualizar timer con duración total
          const durationMs = audioBuffer.length / audioBuffer.sampleRate * 1000;
          updateTimer(durationMs);

        }, (error) => {
          updateStatus('❌ Error procesando audio', 'error');
          console.error('Decode error:', error);
        });
      };
      reader.readAsArrayBuffer(audioBlob);
    };

    btnPlay.onclick = () => {
      if (audioUrl) {
        if (audioElement && isPlayingBack) {
            audioElement.pause();
            isPlayingBack = false;
            btnPlay.textContent = '▶ Reproducir';
            drawWaveform(null); // Pasar null para ocultar cursor
            updateStatus('✅ Grabación lista', 'success');
            return;
        }
        
        audioElement = new Audio(audioUrl);
        audioElement.volume = volumeSlider.value / 100;
        audioElement.crossOrigin = "anonymous";
        
        isPlayingBack = true;
        btnPlay.textContent = '⏸ Pausar';
        
        audioElement.play().catch(err => {
            console.error('Error al reproducir:', err);
            updateStatus('❌ Error reproduciendo', 'error');
            isPlayingBack = false;
            btnPlay.textContent = '▶ Reproducir';
            drawWaveform(null);
        });
        
        updateStatus('▶️ Reproduciendo...', 'timer');
        
        // Iniciar animación de reproducción
        animatePlayback();
        
        audioElement.onended = () => {
            isPlayingBack = false;
            btnPlay.textContent = '▶ Reproducir';
            drawWaveform(null);
            updateStatus('✅ Grabación lista', 'success');
            updateTimer(audioElement.duration * 1000);
        };
        
        audioElement.onerror = (error) => {
            console.error('Error de audio:', error);
            updateStatus('❌ Error reproduciendo', 'error');
            isPlayingBack = false;
            btnPlay.textContent = '▶ Reproducir';
            drawWaveform(null);
        };
      }
    };

    btnCancel.onclick = () => {
      cancelled = true;
      btnRecord.disabled = true;
      btnPlay.disabled = true;
      btnAccept.disabled = true;
      btnExport.disabled = true;
      btnCancel.disabled = true;

      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        btnRecord.classList.remove('recording');
        btnRecord.textContent = '🎤 Grabar';
        stopTimer();
        if (animationId) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
        return;
      }

      if (audioElement && isPlayingBack) {
        audioElement.pause();
        isPlayingBack = false;
      }

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      audioChunks = [];
      audioBlob = null;
      audioUrl = null;
      audioElement = null;

      waveformData = [];
      originalAudioBuffer = null;

      playhead.style.display = 'none';

      btnPlay.disabled = true;
      btnAccept.disabled = true;
      btnExport.disabled = true;
      btnPlay.textContent = '▶ Reproducir';
      btnRecord.disabled = false;
      btnCancel.disabled = false;

      drawWaveform();
      updateStatus('❌ Cancelado', 'error');
      updateTimer(0);
      updateProgress(0);
      peakLevelDisplay.textContent = '-∞';
      fileSizeDisplay.textContent = '0KB';

      resolveAudio(null);
    };

    btnExport.onclick = () => {
      if (audioBlob) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(audioBlob);
        link.download = `recording_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.wav`;
        link.click();
        updateStatus('💾 Archivo descargado', 'success');
      }
    };

    btnAccept.onclick = async () => {
      if (audioBlob) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const base64 = e.target.result.split(',')[1];
          window.audioData = base64;
          window.audioReady = true;
          resolveAudio(base64);
          btnRecord.disabled = true;
          btnPlay.disabled = true;
          btnAccept.disabled = true;
          btnCancel.disabled = true;
          btnExport.disabled = true;
          updateStatus('✅ Audio aceptado ✓', 'success');
        };
        reader.readAsDataURL(audioBlob);
      }
    };

  } catch (error) {
    updateStatus('❌ Error: Sin acceso al micrófono', 'error');
    btnRecord.disabled = true;
    rejectAudio(error);
    console.error('Error:', error);
  }
})();
</script>
"""


class AudioInput(object):
    """
    Panel interactivo para grabar audio en Google Colab.
    Versión con waveform completo y línea de tiempo.
    """

    def record(self, duration=None, sample_rate=16000):
        """
        Graba audio con interfaz interactiva.

        Returns:
            tuple: (audio_array, sample_rate)
        """
        display(HTML(audio_html))
        base64_audio = eval_js("audioPromise")

        if base64_audio is None:
            return None, None

        audio_binary = b64decode(base64_audio)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_binary)
        temp_file.close()

        try:
            sr, audio_data = wavfile.read(temp_file.name)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            return audio_data, sr
        finally:
            import os
            os.unlink(temp_file.name)

    def record_to_file(self, filename='recording.wav', duration=None, sample_rate=16000):
        """
        Graba audio y lo guarda en un archivo.
        """
        display(HTML(audio_html))
        base64_audio = eval_js("audioPromise")

        if base64_audio is None:
            return None, None

        audio_binary = b64decode(base64_audio)
        output_dir = tempfile.mkdtemp()
        output_path = f"{output_dir}/{filename}"

        with open(output_path, 'wb') as f:
            f.write(audio_binary)

        sr, _ = wavfile.read(output_path)
        return output_path, sr

    def record_with_preview(self, duration=None, sample_rate=16000):
        """
        Graba audio y retorna el array de audio y sample_rate.
        """
        return self.record(duration=duration, sample_rate=sample_rate)