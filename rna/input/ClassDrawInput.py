from IPython.display import HTML, display
from google.colab.output import eval_js

from base64 import b64decode
from io import BytesIO

from PIL import Image
import numpy as np


canvas_html = """
<style>

.panel-container{
    display:flex;
    gap:20px;
    align-items:flex-start;
}

.canvas-block{
    display:flex;
    flex-direction:column;
    align-items:center;
}

canvas{
    border:2px dashed black;
    touch-action:none;
}

button{
    display:inline-block;
    margin-top:10px;
    padding:10px;
    background-color:#1fa3ec;
    border:0;
    border-radius:0.317rem;
    color:white;
    height:40px;
    font-size:1rem;
}

button:hover:enabled{
    opacity:0.75;
}

.label{
    margin-bottom:8px;
    font-family:Arial;
    font-size:14px;
    font-weight:bold;
}

</style>

<div class="panel-container">

    <div class="canvas-block">

        <div class="label">
            Dibujo
        </div>

        <canvas
            id="drawCanvas"
            width="%d"
            height="%d">
        </canvas>

        <button id="finishBtn">
            Finalizar Dibujo
        </button>

    </div>

    %s

</div>

<script>

const MODEL_W = %d;
const MODEL_H = %d;

const SHOW_PREVIEW = %s;
const SMOOTH = %s;

const drawCanvas = document.getElementById('drawCanvas');
const ctx = drawCanvas.getContext('2d');

ctx.lineWidth = %f;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

const button = document.getElementById('finishBtn');

let previewCanvas = null;
let previewCtx = null;

if(SHOW_PREVIEW){

    previewCanvas = document.getElementById('previewCanvas');
    previewCtx = previewCanvas.getContext('2d');
}


// fondo negro
ctx.fillStyle = 'black';

ctx.fillRect(
    0,
    0,
    drawCanvas.width,
    drawCanvas.height
);


let mouse = {x:0, y:0};
let drawing = false;


function getMousePos(canvas, evt){

    const rect = canvas.getBoundingClientRect();

    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}


function updatePreview(){

    if(!SHOW_PREVIEW) return;

    // canvas temporal = tamaño REAL del modelo
    const tmp = document.createElement('canvas');

    tmp.width = MODEL_W;
    tmp.height = MODEL_H;

    const tctx = tmp.getContext('2d');

    // smoothing del resize
    tctx.imageSmoothingEnabled = SMOOTH;

    // resize REAL
    tctx.drawImage(
        drawCanvas,
        0,
        0,
        MODEL_W,
        MODEL_H
    );

    // limpiar preview
    previewCtx.clearRect(
        0,
        0,
        previewCanvas.width,
        previewCanvas.height
    );

    // upscale pixelado
    previewCtx.imageSmoothingEnabled = false;

    previewCtx.drawImage(
        tmp,
        0,
        0,
        MODEL_W,
        MODEL_H,
        0,
        0,
        previewCanvas.width,
        previewCanvas.height
    );
}


// MOUSE

drawCanvas.addEventListener('mousedown', (e)=>{

    drawing = true;

    mouse = getMousePos(drawCanvas, e);

    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
});

drawCanvas.addEventListener('mousemove', (e)=>{

    if(!drawing) return;

    mouse = getMousePos(drawCanvas, e);

    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();

    updatePreview();
});

drawCanvas.addEventListener('mouseup', ()=>{
    drawing = false;
});

drawCanvas.addEventListener('mouseleave', ()=>{
    drawing = false;
});


// TOUCH

drawCanvas.addEventListener('touchstart', (e)=>{

    e.preventDefault();

    drawing = true;

    const t = e.touches[0];

    mouse = getMousePos(drawCanvas, t);

    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
});

drawCanvas.addEventListener('touchmove', (e)=>{

    e.preventDefault();

    if(!drawing) return;

    const t = e.touches[0];

    mouse = getMousePos(drawCanvas, t);

    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();

    updatePreview();
});

drawCanvas.addEventListener('touchend', ()=>{
    drawing = false;
});


// preview inicial
updatePreview();


// exportacion FINAL

var data = new Promise(resolve=>{

    button.onclick = ()=>{

        // canvas temporal = entrada REAL modelo
        const tmp = document.createElement('canvas');

        tmp.width = MODEL_W;
        tmp.height = MODEL_H;

        const tctx = tmp.getContext('2d');

        tctx.imageSmoothingEnabled = SMOOTH;

        tctx.drawImage(
            drawCanvas,
            0,
            0,
            MODEL_W,
            MODEL_H
        );

        resolve(tmp.toDataURL('image/png'));

        button.style.visibility = 'hidden';
    }

})

</script>
"""


class DrawInput(object):

    def draw(
        self,
        size=(28,28),
        line_width=1.2,
        scale=15,
        show_preview=True,
        smooth=True
    ):

        w, h = size

        preview_html = ""

        if show_preview:

            preview_html = f"""
            <div class="canvas-block">

                <div class="label">
                    Imagen Generada
                </div>

                <canvas
                    id="previewCanvas"
                    width="{w*scale}"
                    height="{h*scale}">
                </canvas>

            </div>
            """

        html = canvas_html % (
            w * scale,                       # canvas dibujo W
            h * scale,                       # canvas dibujo H
            preview_html,
            w,                               # MODEL_W
            h,                               # MODEL_H
            "true" if show_preview else "false",
            "true" if smooth else "false",
            line_width * scale
        )

        display(HTML(html))

        data = eval_js("data")

        binary = b64decode(data.split(',')[1])

        buffer = BytesIO(binary)

        image = Image.open(buffer).convert("RGBA")

        return image