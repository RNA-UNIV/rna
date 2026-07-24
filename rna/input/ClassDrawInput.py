from IPython.display import HTML, display
from google.colab.output import eval_js
from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np

canvas_html = """
<style>
  .draw-panel {{
    border: 2px dashed #ccc;
    padding: 12px;
    border-radius: 6px;
    background-color: #f9f9f9;
    font-family: Arial, sans-serif;
    display: inline-block;
    min-width: auto;
    max-width: 100%;
  }}

  .draw-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 4px 8px 4px;
    flex-wrap: wrap;
  }}

  .draw-status {{
    font-size: 0.85rem;
    color: #666;
    font-weight: bold;
    white-space: nowrap;
    min-width: 120px;
  }}

  .draw-status.drawing {{
    color: #1fa3ec;
  }}

  .draw-status.ready {{
    color: #28a745;
  }}

  .draw-status.cancelled {{
    color: #d63420;
  }}

  .draw-properties {{
    display: flex;
    gap: 12px;
    font-size: 0.75rem;
    color: #888;
    align-items: center;
    flex-wrap: wrap;
    margin-left: auto;
  }}

  .draw-properties span {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  .property-value {{
    color: #444;
    font-weight: 600;
  }}

  .draw-container {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }}

  .canvas-wrapper {{
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
    padding: 4px;
    position: relative;
  }}

  .canvas-wrapper canvas {{
    display: block;
    touch-action: none;
    cursor: crosshair;
    width: {3}px;
    height: {4}px;
  }}

  .preview-wrapper {{
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
    padding: 4px;
  }}

  .preview-wrapper canvas {{
    display: block;
    image-rendering: pixelated;
    width: {3}px;
    height: {4}px;
  }}

  .controls-row {{
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    padding-top: 8px;
    flex-wrap: wrap;
  }}

  .controls-left {{
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: wrap;
  }}

  .controls-right {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-left: auto;
  }}

  .draw-btn {{
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
  }}

  .draw-btn:hover:enabled {{
    opacity: 0.8;
    transform: scale(0.98);
  }}

  .draw-btn:disabled {{
    cursor: not-allowed;
    opacity: 0.4;
  }}

  .btn-clear {{
    background-color: #6c757d;
    color: white;
  }}

  .btn-clear:hover:enabled {{
    background-color: #5a6268;
  }}

  .btn-accept {{
    background-color: #28a745;
    color: white;
  }}

  .btn-cancel {{
    background-color: #dc3545;
    color: white;
  }}

  .btn-cancel:hover:enabled {{
    background-color: #c82333;
  }}

  .btn-export {{
    background-color: #6f42c1;
    color: white;
  }}

  .btn-export:hover:enabled {{
    background-color: #5a32a3;
  }}

  .slider-control {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .slider-control label {{
    font-size: 0.7rem;
    color: #666;
    white-space: nowrap;
  }}

  .slider-control input[type="range"] {{
    width: 60px;
    height: 4px;
    cursor: pointer;
  }}

  .slider-control .slider-value {{
    font-size: 0.7rem;
    color: #444;
    font-weight: 600;
    min-width: 20px;
    text-align: center;
  }}

  .toggle-control {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  .toggle-control label {{
    font-size: 0.7rem;
    color: #666;
    white-space: nowrap;
    cursor: pointer;
  }}

  .toggle-switch {{
    position: relative;
    width: 32px;
    height: 18px;
    background-color: #ccc;
    border-radius: 10px;
    cursor: pointer;
    transition: background-color 0.3s;
    flex-shrink: 0;
  }}

  .toggle-switch.active {{
    background-color: #28a745;
  }}

  .toggle-switch .toggle-slider {{
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    background-color: white;
    border-radius: 50%;
    transition: transform 0.3s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }}

  .toggle-switch.active .toggle-slider {{
    transform: translateX(14px);
  }}

  @media (max-width: 768px) {{
    .draw-panel {{
      padding: 8px;
    }}

    .draw-header {{
      flex-direction: column;
      align-items: stretch;
    }}

    .draw-properties {{
      margin-left: 0;
      justify-content: center;
    }}

    .draw-container {{
      flex-direction: column;
      align-items: center;
    }}

    .controls-row {{
      flex-direction: column;
      align-items: stretch;
    }}

    .controls-right {{
      margin-left: 0;
      justify-content: center;
    }}

    .draw-btn {{
      font-size: 0.7rem;
      padding: 3px 8px;
      height: 24px;
    }}

    .slider-control input[type="range"] {{
      width: 40px;
    }}
  }}
</style>

<div class="draw-panel">
  <!-- FILA 1: Estado + Propiedades -->
  <div class="draw-header">
    <div class="draw-status drawing" id="status">✏️ Dibujando</div>
    <div class="draw-properties">
      <span>📐 <span class="property-value" id="canvas-size">{0}×{1}</span></span>
      <span>✏️ <span class="property-value" id="line-width">{5}</span>px</span>
      <span>🔲 <span class="property-value" id="pixel-count">{6}</span>px</span>
      <span>🔄 <span class="property-value" id="smooth-status">On</span></span>
    </div>
  </div>

  <!-- FILA 2: Canvas + Preview -->
  <div class="draw-container">
    <div class="canvas-wrapper">
      <canvas id="drawCanvas"></canvas>
    </div>
    <div class="preview-wrapper">
      <canvas id="previewCanvas"></canvas>
    </div>
  </div>

  <!-- FILA 3: Controles -->
  <div class="controls-row">
    <div class="controls-left">
      <button class="draw-btn btn-clear" id="btn-clear">🗑️ Limpiar</button>
      <button class="draw-btn btn-accept" id="btn-accept">✅ Aceptar</button>
      <button class="draw-btn btn-cancel" id="btn-cancel">✖️ Cancelar</button>
    </div>
    <div class="controls-right">
      <div class="slider-control">
        <label>✏️ Grosor</label>
        <input type="range" id="line-width-slider" min="0.5" max="5" step="0.5" value="{5}">
        <span class="slider-value" id="line-width-value">{5}</span>
      </div>
      <div class="toggle-control">
        <label>Suave</label>
        <div class="toggle-switch active" id="smooth-toggle">
          <div class="toggle-slider"></div>
        </div>
      </div>
      <button class="draw-btn btn-export" id="btn-export" disabled>💾</button>
    </div>
  </div>
</div>

<script>
(function() {{
  // Configuración
  const MODEL_W = {0};
  const MODEL_H = {1};
  const SCALE = {2};
  const DEFAULT_LINE_WIDTH = {5};

  // Elementos DOM
  const statusEl = document.getElementById('status');
  const drawCanvas = document.getElementById('drawCanvas');
  const previewCanvas = document.getElementById('previewCanvas');
  const btnClear = document.getElementById('btn-clear');
  const btnAccept = document.getElementById('btn-accept');
  const btnCancel = document.getElementById('btn-cancel');
  const btnExport = document.getElementById('btn-export');
  const lineWidthSlider = document.getElementById('line-width-slider');
  const lineWidthValue = document.getElementById('line-width-value');
  const smoothToggle = document.getElementById('smooth-toggle');
  const smoothStatus = document.getElementById('smooth-status');
  const canvasSizeDisplay = document.getElementById('canvas-size');
  const lineWidthDisplay = document.getElementById('line-width');
  const pixelCountDisplay = document.getElementById('pixel-count');

  // Contextos
  const ctx = drawCanvas.getContext('2d');
  const previewCtx = previewCanvas.getContext('2d');

  // Estado
  let isDrawing = false;
  let isComplete = false;
  let isCancelled = false;
  let lineWidth = DEFAULT_LINE_WIDTH;
  let smooth = true;
  let mouseX = 0;
  let mouseY = 0;

  // Configurar tamaños
  function setupCanvases() {{
    // Canvas de dibujo: tamaño interno en pixels
    const visualSize = MODEL_W * SCALE;
    drawCanvas.width = visualSize;
    drawCanvas.height = MODEL_H * SCALE;

    // Canvas de preview: mismo tamaño visual
    previewCanvas.width = visualSize;
    previewCanvas.height = MODEL_H * SCALE;
    previewCtx.imageSmoothingEnabled = false;

    // Actualizar display de propiedades
    canvasSizeDisplay.textContent = `${{MODEL_W}}×${{MODEL_H}}`;
    pixelCountDisplay.textContent = MODEL_W * MODEL_H;
  }}

  // Inicializar canvas con fondo negro
  function clearCanvas() {{
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    updatePreview();
    if (!isComplete && !isCancelled) {{
      statusEl.textContent = '✏️ Dibujando';
      statusEl.className = 'draw-status drawing';
    }}
    btnExport.disabled = true;
  }}

  // Actualizar preview - como en la versión original
  function updatePreview() {{
    // Canvas temporal para resize
    const tmp = document.createElement('canvas');
    tmp.width = MODEL_W;
    tmp.height = MODEL_H;
    const tctx = tmp.getContext('2d');

    // Aplicar suavizado según el toggle
    tctx.imageSmoothingEnabled = smooth;

    // Redimensionar al tamaño real del modelo
    tctx.drawImage(drawCanvas, 0, 0, MODEL_W, MODEL_H);

    // Limpiar preview
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

    // Dibujar en preview (escalado pixelado)
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(
      tmp, 
      0, 0, MODEL_W, MODEL_H,
      0, 0, previewCanvas.width, previewCanvas.height
    );
  }}

  // Configurar eventos de dibujo
  function setupDrawing() {{
    // Mouse
    drawCanvas.addEventListener('mousedown', (e) => {{
      if (isComplete || isCancelled) return;
      isDrawing = true;
      const rect = drawCanvas.getBoundingClientRect();
      const scaleX = drawCanvas.width / rect.width;
      const scaleY = drawCanvas.height / rect.height;
      mouseX = (e.clientX - rect.left) * scaleX;
      mouseY = (e.clientY - rect.top) * scaleY;
      ctx.beginPath();
      ctx.moveTo(mouseX, mouseY);
    }});

    drawCanvas.addEventListener('mousemove', (e) => {{
      if (!isDrawing || isComplete || isCancelled) return;
      const rect = drawCanvas.getBoundingClientRect();
      const scaleX = drawCanvas.width / rect.width;
      const scaleY = drawCanvas.height / rect.height;
      mouseX = (e.clientX - rect.left) * scaleX;
      mouseY = (e.clientY - rect.top) * scaleY;
      ctx.lineTo(mouseX, mouseY);
      ctx.stroke();
      updatePreview();
    }});

    drawCanvas.addEventListener('mouseup', () => {{
      isDrawing = false;
    }});

    drawCanvas.addEventListener('mouseleave', () => {{
      isDrawing = false;
    }});

    // Touch
    drawCanvas.addEventListener('touchstart', (e) => {{
      e.preventDefault();
      if (isComplete || isCancelled) return;
      isDrawing = true;
      const t = e.touches[0];
      const rect = drawCanvas.getBoundingClientRect();
      const scaleX = drawCanvas.width / rect.width;
      const scaleY = drawCanvas.height / rect.height;
      mouseX = (t.clientX - rect.left) * scaleX;
      mouseY = (t.clientY - rect.top) * scaleY;
      ctx.beginPath();
      ctx.moveTo(mouseX, mouseY);
    }}, {{ passive: false }});

    drawCanvas.addEventListener('touchmove', (e) => {{
      e.preventDefault();
      if (!isDrawing || isComplete || isCancelled) return;
      const t = e.touches[0];
      const rect = drawCanvas.getBoundingClientRect();
      const scaleX = drawCanvas.width / rect.width;
      const scaleY = drawCanvas.height / rect.height;
      mouseX = (t.clientX - rect.left) * scaleX;
      mouseY = (t.clientY - rect.top) * scaleY;
      ctx.lineTo(mouseX, mouseY);
      ctx.stroke();
      updatePreview();
    }}, {{ passive: false }});

    drawCanvas.addEventListener('touchend', () => {{
      isDrawing = false;
    }});
  }}

  // Actualizar grosor del trazo
  function updateLineWidth() {{
    lineWidth = parseFloat(lineWidthSlider.value);
    lineWidthValue.textContent = lineWidth;
    lineWidthDisplay.textContent = lineWidth;
    ctx.lineWidth = lineWidth * SCALE;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
  }}

  // Actualizar suavizado
  function updateSmooth() {{
    smooth = smoothToggle.classList.contains('active');
    smoothStatus.textContent = smooth ? 'On' : 'Off';
    updatePreview();
  }}

  // Obtener imagen final
  function getFinalImage() {{
    const tmp = document.createElement('canvas');
    tmp.width = MODEL_W;
    tmp.height = MODEL_H;
    const tctx = tmp.getContext('2d');
    tctx.imageSmoothingEnabled = smooth;
    tctx.drawImage(drawCanvas, 0, 0, MODEL_W, MODEL_H);
    return tmp.toDataURL('image/png');
  }}

  // Botón Limpiar
  btnClear.onclick = () => {{
    if (isComplete || isCancelled) return;
    clearCanvas();
  }};

  // Botón Aceptar
  btnAccept.onclick = () => {{
    if (isComplete) return;
    isComplete = true;
    statusEl.textContent = '✅ Listo';
    statusEl.className = 'draw-status ready';
    btnAccept.disabled = true;
    btnCancel.disabled = true;
    btnClear.disabled = true;
    btnExport.disabled = false;
    lineWidthSlider.disabled = true;
    smoothToggle.style.cursor = 'default';
    smoothToggle.style.pointerEvents = 'none';

    // Resolver la promesa con la imagen
    window.__drawData = getFinalImage();
    window.__drawResolve(true);
  }};

  // Botón Cancelar
  btnCancel.onclick = () => {{
    if (isCancelled) return;
    isCancelled = true;
    statusEl.textContent = '✖️ Cancelado';
    statusEl.className = 'draw-status cancelled';
    btnAccept.disabled = true;
    btnCancel.disabled = true;
    btnClear.disabled = true;
    btnExport.disabled = true;
    lineWidthSlider.disabled = true;
    smoothToggle.style.cursor = 'default';
    smoothToggle.style.pointerEvents = 'none';
    clearCanvas();
    window.__drawResolve(null);
  }};

  // Botón Exportar PNG
  btnExport.onclick = () => {{
    if (!isComplete) return;
    const dataUrl = getFinalImage();
    const link = document.createElement('a');
    link.href = dataUrl;
    const timestamp = new Date().toISOString().slice(0,19).replace(/:/g, '-');
    link.download = `drawing_${{timestamp}}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    statusEl.textContent = '💾 PNG descargado';
    statusEl.className = 'draw-status ready';
  }};

  // Toggle suavizado
  smoothToggle.onclick = (e) => {{
    if (isComplete || isCancelled) return;
    e.stopPropagation();
    smoothToggle.classList.toggle('active');
    updateSmooth();
  }};

  // Slider de grosor
  lineWidthSlider.oninput = () => {{
    if (isComplete || isCancelled) return;
    updateLineWidth();
  }};

  // Inicialización
  setupCanvases();
  clearCanvas();
  updateLineWidth();
  setupDrawing();

  // Crear promesa para el resultado
  window.__drawResolve = null;
  window.__drawData = null;
  window.__drawPromise = new Promise((resolve) => {{
    window.__drawResolve = resolve;
  }});

}})();
</script>
"""


class DrawInput(object):
    """
    Panel interactivo para dibujar en Google Colab.
    Versión mejorada con interfaz consistente y escalado funcional.
    """

    def draw(
            self,
            size=(28, 28),
            line_width=2.0,
            scale=15,
            smooth=True
    ):
        """
        Dibuja una imagen con interfaz interactiva.

        Args:
            size (tuple): Tamaño de la imagen de salida (ancho, alto)
            line_width (float): Grosor del trazo inicial (0.5-5.0)
            scale (int): Factor que controla el tamaño visual del canvas.
                        El canvas de dibujo tendrá tamaño (width*scale) x (height*scale) pixels.
                        Ej: size=(28,28), scale=1 → 28x28 pixels
                            size=(28,28), scale=10 → 280x280 pixels
            smooth (bool): Suavizado activado por defecto

        Returns:
            PIL.Image: Imagen dibujada en formato RGBA, o None si se cancela
        """
        w, h = size

        # Validar parámetros
        line_width = max(0.5, min(5.0, line_width))
        scale = max(1, int(scale))

        html = canvas_html.format(
            w,  # {0} MODEL_W
            h,  # {1} MODEL_H
            scale,  # {2} SCALE
            w * scale,  # {3} Visual width
            h * scale,  # {4} Visual height
            line_width,  # {5} DEFAULT_LINE_WIDTH
            w * h  # {6} total pixels
        )

        display(HTML(html))

        # Esperar a que el usuario termine
        result = eval_js("window.__drawPromise")

        # Obtener los datos
        data_url = eval_js("window.__drawData")

        if data_url is None:
            return None

        # Decodificar imagen
        binary = b64decode(data_url.split(',')[1])
        buffer = BytesIO(binary)
        image = Image.open(buffer).convert("RGBA")

        return image

    def draw_to_array(
            self,
            size=(28, 28),
            line_width=2.0,
            scale=15,
            smooth=True,
            normalize=True
    ):
        """
        Dibuja y retorna la imagen como array numpy.

        Args:
            size (tuple): Tamaño de la imagen de salida (ancho, alto)
            line_width (float): Grosor del trazo inicial
            scale (int): Factor de escala visual del canvas
            smooth (bool): Suavizado activado por defecto
            normalize (bool): Normalizar valores a [0,1]

        Returns:
            numpy.ndarray: Imagen como array, o None si se cancela
        """
        image = self.draw(size=size, line_width=line_width,
                          scale=scale, smooth=smooth)

        if image is None:
            return None

        # Convertir a array y normalizar
        array = np.array(image.convert('L'))  # Escala de grises

        if normalize:
            array = array.astype(np.float32) / 255.0

        return array

    def draw_to_file(
            self,
            filename='drawing.png',
            size=(28, 28),
            line_width=2.0,
            scale=15,
            smooth=True
    ):
        """
        Dibuja y guarda la imagen en un archivo.

        Args:
            filename (str): Nombre del archivo de salida
            size (tuple): Tamaño de la imagen de salida (ancho, alto)
            line_width (float): Grosor del trazo inicial
            scale (int): Factor de escala visual del canvas
            smooth (bool): Suavizado activado por defecto

        Returns:
            str: Ruta del archivo guardado, o None si se cancela
        """
        image = self.draw(size=size, line_width=line_width,
                          scale=scale, smooth=smooth)

        if image is None:
            return None

        # Guardar en archivo temporal
        import tempfile
        import os

        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)

        return output_path