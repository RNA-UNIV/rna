import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Detección de entorno ──────────────────────────────────────────
def _detect_env():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return 'script'
        if 'google.colab' in str(type(shell)):
            return 'colab'
        if type(shell).__name__ == 'ZMQInteractiveShell':
            return 'jupyter'
    except ImportError:
        pass
    return 'script'

def _setup_backend(env):
    if env == 'script':
        for backend in ['TkAgg', 'Qt5Agg', 'WXAgg', 'GTK3Agg']:
            try:
                matplotlib.use(backend)
                fig = plt.figure()
                plt.close(fig)
                return
            except Exception:
                continue
    else:
        matplotlib.use('Agg')

ENV = _detect_env()
_setup_backend(ENV)

# ── Paleta seaborn muted, 8 colores ──────────────────────────────
_COLORES = [
    '#4878cf',  # azul
    '#d65f5f',  # rojo
    '#59a14f',  # verde
    '#af7aa1',  # violeta
    '#e1a030',  # amarillo ocre
    '#4db3be',  # teal
    '#b07040',  # marrón
    '#7070a0',  # gris azulado
]

_COLOR_LINEA = '#444444'

# ── Registro de figuras ───────────────────────────────────────────
_figures = {}

def _init_figure(fig_id, entradas, salida, titulos):
    if ENV == 'script':
        plt.ion()

    fig, ax = plt.subplots(figsize=(7, 6))

    # ── Cambiar fondos a gris claro ────────────────────────
    #fig.patch.set_facecolor('#f0f0f0')      # fondo de la figura
    ax.set_facecolor('#f0f0f0')            # fondo del área del gráfico
    ax.grid(True, color='black', alpha=1)

    try:
        fig.canvas.manager.set_window_title(fig_id)
    except Exception:
        pass

    clases = np.unique(salida)
    scatters = []
    for i, c in enumerate(clases):
        color = _COLORES[i % len(_COLORES)]
        sc, = ax.plot([], [], 'o', color=color, zorder=3,
                      label=str(c), markersize=6, alpha=0.85)
        scatters.append(sc)
    scatters = tuple(scatters)

    line, = ax.plot([], [], '-', color=_COLOR_LINEA, lw=1.8, zorder=2, label='Frontera')

    if len(titulos) == 2:
        ax.set_xlabel(titulos[0])
        ax.set_ylabel(titulos[1])

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ipy_handle = None
    if ENV in ('jupyter', 'colab'):
        from IPython import display as D
        ipy_handle = D.display(fig, display_id=True)
    else:
        plt.show(block=False)

    _figures[fig_id] = {
        'fig':        fig,
        'ax':         ax,
        'scatters':   scatters,
        'line':       line,
        'clases':     clases,
        'ipy_handle': ipy_handle,
    }

def _render(fig_id):
    s = _figures[fig_id]
    if ENV == 'script':
        s['fig'].canvas.draw()
        s['fig'].canvas.flush_events()
        plt.pause(0.01)
    else:
        s['ipy_handle'].update(s['fig'])

# ── API pública ───────────────────────────────────────────────────
def dibuPtosRecta(entradas, salida, W, b, titulos=[], fig_id='default', reset=False, titulo=''):
    if entradas.shape[1] != 2:
        return

    if not fig_id:  # None, 0, '', False → genera id nuevo
        import uuid
        fig_id = str(uuid.uuid4())

    if fig_id in _figures:
        if not plt.fignum_exists(_figures[fig_id]['fig'].number) or reset:
            del _figures[fig_id]

    if fig_id not in _figures:
        _init_figure(fig_id, entradas, salida, titulos)

    s  = _figures[fig_id]
    ax = s['ax']

    ax.set_xlim(entradas[:, 0].min() - 0.05, entradas[:, 0].max() + 0.05)
    ax.set_ylim(entradas[:, 1].min() - 0.05, entradas[:, 1].max() + 0.05)

    clases = s['clases']
    if len(clases) == 2:
        mn, mx = min(clases), max(clases)
        s['scatters'][0].set_data(entradas[salida == mn, 0], entradas[salida == mn, 1])
        s['scatters'][1].set_data(entradas[salida == mx, 0], entradas[salida == mx, 1])
    else:
        s['scatters'][0].set_data(entradas[:, 0], entradas[:, 1])

    if abs(W[1]) > 1e-9:
        X = np.array([entradas[:, 0].min(), entradas[:, 0].max()])
        Y = (-W[0] / W[1]) * X - (b / W[1])
        s['line'].set_data(X, np.squeeze(np.asarray(Y)))

    if titulo:
        s['ax'].set_title(titulo)

    _render(fig_id)

    return fig_id

def setTitulo(titulo, fig_id='default'):
    if fig_id in _figures:
        _figures[fig_id]['ax'].set_title(titulo)
        _render(fig_id)

def waitDibu(fig_id='default'):
    """
    Script  : bloquea hasta que el usuario cierre la ventana.
    Notebook: no-op, el último frame ya quedó en el output via handle.update().
    """
    if ENV == 'script' and fig_id in _figures:
        plt.ioff()
        plt.show(block=True)
