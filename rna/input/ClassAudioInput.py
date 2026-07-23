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
    padding: 16px;
    border-radius: 4px;
    background-color: #f9f9f9;
    font-family: Arial, sans-serif;
    max-width: 500px;
  }

  .audio-status {
    text-align: center;
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 12px;
    min-height: 20px;
    font-weight: bold;
  }

  .waveform-container {
    margin-bottom: 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
    padding: 8px;
  }

  .waveform-canvas {
    width: 100%;
    height: 100px;
    display: block;
    cursor: pointer;
  }

  .controls-row {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 12px;
  }

  .volume-control {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .volume-slider {
    width: 100px;
    height: 6px;
    cursor: pointer;
  }

  .volume-label {
    font-size: 0.8rem;
    color: #666;
    min-width: 30px;
  }

  .audio-controls {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
  }

  .audio-btn {
    padding: 6px 12px;
    font-size: 0.85rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    transition: opacity 0.2s;
    height: 32px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .audio-btn:hover:enabled {
    opacity: 0.8;
  }

  .audio-btn:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .btn-record {
    background-color: #e05a3a;
    color: white;
  }

  .btn-record.recording {
    background-color: #d63420;
    animation: pulse 1s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .btn-play {
    background-color: #1fa3ec;
    color: white;
  }

  .btn-accept {
    background-color: #28a745;
    color: white;
  }

  .btn-cancel {
    background-color: #6c757d;
    color: white;
  }

  .timer {
    font-weight: bold;
    color: #d63420;
  }

  .error {
    color: #d63420;
  }
</style>

<div class="audio-panel">
  <div class="audio-status" id="status">Listo para grabar</div>

  <div class="waveform-container">
    <canvas id="waveform-canvas" class="waveform-canvas"></canvas>
  </div>

  <div class="controls-row">
    <div class="volume-control">
      <span>🔊</span>
      <input type="range" id="volume-slider" class="volume-slider" min="0" max="100" value="80">
      <span class="volume-label" id="volume-label">80%</span>
    </div>
  </div>

  <div class="audio-controls">
    <button class="audio-btn btn-record" id="btn-record">🎤 Grabar</button>
    <button class="audio-btn btn-play" id="btn-play" disabled>▶ Reproducir</button>
    <button class="audio-btn btn-accept" id="btn-accept" disabled>✓ Aceptar</button>
    <button class="audio-btn btn-cancel" id="btn-cancel">✕ Cancelar</button>
  </div>
</div>

<script>
(async function() {
  const statusEl = document.getElementById('status');
  const btnRecord = document.getElementById('btn-record');
  const btnPlay = document.getElementById('btn-play');
  const btnAccept = document.getElementById('btn-accept');
  const btnCancel = document.getElementById('btn-cancel');
  const waveformCanvas = document.getElementById('waveform-canvas');
  const volumeSlider = document.getElementById('volume-slider');
  const volumeLabel = document.getElementById('volume-label');

  const waveCtx = waveformCanvas.getContext('2d');
  waveformCanvas.width = waveformCanvas.offsetWidth;
  waveformCanvas.height = 100;

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

  function updateStatus(msg, className = '') {
    statusEl.textContent = msg;
    statusEl.className = 'audio-status ' + className;
  }

  function formatTime(ms) {
    const seconds = Math.floor(ms / 1000);
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      updateStatus(`Grabando... ${formatTime(elapsed)}`, 'timer');
    }, 100);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function drawWaveform(data = null) {
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;

    // Limpiar canvas
    waveCtx.fillStyle = '#f9f9f9';
    waveCtx.fillRect(0, 0, width, height);

    // Línea de referencia
    waveCtx.strokeStyle = '#ddd';
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(0, height / 2);
    waveCtx.lineTo(width, height / 2);
    waveCtx.stroke();

    if (!data || data.length === 0) return;

    // Dibujar forma de onda
    waveCtx.strokeStyle = '#1fa3ec';
    waveCtx.lineWidth = 2;
    waveCtx.beginPath();

    const sliceWidth = width / data.length;
    let x = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i] / 128.0;
      const y = (v * height) / 2;

      if (i === 0) {
        waveCtx.moveTo(x, y);
      } else {
        waveCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    waveCtx.lineTo(width, height / 2);
    waveCtx.stroke();
  }

  function animateWaveform() {
    if (!analyser) return;

    if (!dataArray) {
      dataArray = new Uint8Array(analyser.frequencyBinCount);
    }

    analyser.getByteFrequencyData(dataArray);
    drawWaveform(dataArray);

    if (!isPlayingBack && mediaRecorder.state === 'recording') {
      animationId = requestAnimationFrame(animateWaveform);
    }
  }

  function updateVolumeLabel() {
    const volume = volumeSlider.value;
    volumeLabel.textContent = volume + '%';
  }

  volumeSlider.oninput = () => {
    updateVolumeLabel();
    if (audioElement) {
      audioElement.volume = volumeSlider.value / 100;
    }
  };

  try {
    // Solicitar acceso al micrófono
    micStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false
      } 
    });

    // Crear contexto de audio para análisis
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    const source = audioContext.createMediaStreamSource(micStream);
    source.connect(analyser);

    mediaRecorder = new MediaRecorder(micStream);

    btnRecord.onclick = async () => {
      if (mediaRecorder.state === 'inactive') {
        audioChunks = [];
        audioBlob = null;
        audioUrl = null;
        mediaRecorder.start(100);
        btnRecord.classList.add('recording');
        btnRecord.textContent = '⏹ Detener';
        btnPlay.disabled = true;
        btnAccept.disabled = true;
        btnCancel.disabled = false;
        startTimer();
        updateStatus('Grabando...', 'timer');
        animateWaveform();
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
      audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      audioUrl = URL.createObjectURL(audioBlob);
      btnPlay.disabled = false;
      btnAccept.disabled = false;
      updateStatus('Grabación completada');

      // Mostrar forma de onda grabada
      const reader = new FileReader();
      reader.onload = (e) => {
        const arrayBuffer = e.target.result;
        audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
          const channelData = audioBuffer.getChannelData(0);
          const samples = Math.floor(channelData.length / 100);
          const downsampled = [];

          for (let i = 0; i < 100; i++) {
            let sum = 0;
            for (let j = 0; j < samples; j++) {
              sum += Math.abs(channelData[i * samples + j]);
            }
            downsampled.push((sum / samples) * 128);
          }

          drawWaveform(downsampled);
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
          updateStatus('Grabación completada');
          return;
        }

        audioElement = new Audio(audioUrl);
        audioElement.volume = volumeSlider.value / 100;
        isPlayingBack = true;
        btnPlay.textContent = '⏸ Pausar';
        audioElement.play();
        updateStatus('Reproduciendo...');

        audioElement.onended = () => {
          isPlayingBack = false;
          btnPlay.textContent = '▶ Reproducir';
          updateStatus('Grabación completada');
          drawWaveform();
        };
      }
    };

    btnCancel.onclick = () => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        btnRecord.classList.remove('recording');
        btnRecord.textContent = '🎤 Grabar';
        stopTimer();
        if (animationId) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
      }
      if (audioElement && isPlayingBack) {
        audioElement.pause();
        isPlayingBack = false;
      }
      audioChunks = [];
      audioBlob = null;
      audioUrl = null;
      audioElement = null;
      btnPlay.disabled = true;
      btnAccept.disabled = true;
      btnPlay.textContent = '▶ Reproducir';
      drawWaveform();
      updateStatus('Cancelado');
    };

    // Promise que se resuelve cuando el usuario acepta
    window.audioPromise = new Promise(resolve => {
      btnAccept.onclick = async () => {
        if (audioBlob) {
          const reader = new FileReader();
          reader.onload = (e) => {
            const base64 = e.target.result.split(',')[1];
            resolve(base64);
            btnRecord.disabled = true;
            btnPlay.disabled = true;
            btnAccept.disabled = true;
            btnCancel.disabled = true;
            updateStatus('Audio aceptado ✓');
          };
          reader.readAsDataURL(audioBlob);
        }
      };
    });

  } catch (error) {
    updateStatus('Error: No se pudo acceder al micrófono', 'error');
    btnRecord.disabled = true;
    console.error('Error:', error);
  }
})();
</script>
"""


class AudioInput(object):
    """
    Panel interactivo para grabar audio en Google Colab.

    Uso:
        panel = AudioInput()
        audio_array = panel.record(duration=5)
        # o
        filename = panel.record_to_file('mi_audio.wav', duration=5)
    """

    def record(self, duration=None, sample_rate=16000):
        """
        Graba audio con interfaz interactiva.

        Args:
            duration: Duración máxima en segundos (opcional, sin límite si es None)
            sample_rate: Frecuencia de muestreo en Hz (default: 16000)

        Returns:
            numpy.ndarray: Array de audio (mono) con valores en rango [-1, 1]
        """
        display(HTML(audio_html))

        # Obtener el audio codificado en base64
        base64_audio = eval_js("audioPromise")

        # Decodificar
        audio_binary = b64decode(base64_audio)

        # Guardar temporalmente y cargar con scipy
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_binary)
        temp_file.close()

        try:
            sr, audio_data = wavfile.read(temp_file.name)

            # Convertir a mono si es estéreo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Normalizar a rango [-1, 1]
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

            return audio_data
        finally:
            import os
            os.unlink(temp_file.name)

    def record_to_file(self, filename='recording.wav', duration=None, sample_rate=16000):
        """
        Graba audio y lo guarda en un archivo.

        Args:
            filename: Nombre del archivo (default: 'recording.wav')
            duration: Duración máxima en segundos (opcional)
            sample_rate: Frecuencia de muestreo en Hz (default: 16000)

        Returns:
            str: Ruta completa del archivo guardado
        """
        display(HTML(audio_html))

        # Obtener el audio
        base64_audio = eval_js("audioPromise")
        audio_binary = b64decode(base64_audio)

        # Guardar en archivo
        output_dir = tempfile.mkdtemp()
        output_path = f"{output_dir}/{filename}"

        with open(output_path, 'wb') as f:
            f.write(audio_binary)

        return output_path

    def record_with_preview(self, duration=None, sample_rate=16000):
        """
        Graba audio y retorna tanto el array como una vista previa.

        Args:
            duration: Duración máxima en segundos (opcional)
            sample_rate: Frecuencia de muestreo en Hz (default: 16000)

        Returns:
            tuple: (audio_array, audio_widget)
                - audio_array: numpy.ndarray con los datos de audio
                - audio_widget: Widget de IPython.display.Audio para reproducir
        """
        audio_array = self.record(duration=duration, sample_rate=sample_rate)
        audio_widget = Audio(audio_array, rate=sample_rate)

        return audio_array, audio_widget