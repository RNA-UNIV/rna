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
    # jupyter y colab: no tocar el backend, ya está configurado


ENV = _detect_env()
_setup_backend(ENV)

_COLORES = [
    # tab20 (índices pares)
    '#1f77b4',  # azul
    '#ff7f0e',  # naranja
    '#2ca02c',  # verde
    '#d62728',  # rojo
    '#9467bd',  # violeta
    '#8c564b',  # marrón
    '#e377c2',  # rosa
    '#7f7f7f',  # gris
    '#bcbd22',  # oliva
    '#17becf',  # cian

    # Complementarios
    '#8dd3c7',  # turquesa pastel (Set3)
    '#ffed6f',  # amarillo claro (Set3)
    '#80b1d3',  # azul claro (Set3)
    '#b3de69',  # verde lima (Set3)
    '#fb8072',  # coral (Set3)
    '#bebada',  # lavanda (Set3)
]

_COLOR_LINEA = _COLORES[-1]
_FACE_COLOR = '#f0f0f0'

# ── Registro de figuras ───────────────────────────────────────────
_figures = {}


def _init_figure(fig_id, entradas, salida, titulos):
    if ENV == 'script':
        plt.ion()

    with plt.ioff():  # suprime el render automático de inline
        fig, ax = plt.subplots()

    # ── Cambiar fondos a gris claro ────────────────────────
    # fig.patch.set_facecolor('#f0f0f0')      # fondo de la figura
    # ax.set_facecolor(_FACE_COLOR)           # fondo del área del gráfico
    # ax.grid(True, color='black', alpha=1)
    ax.grid(True, alpha=1)
    try:
        fig.canvas.manager.set_window_title(fig_id)
    except Exception:
        pass

    clases = np.unique(salida)
    scatters = []
    for i, c in enumerate(clases):
        color = _COLORES[i % len(_COLORES)]
        sc, = ax.plot([], [], 'o', color=color, zorder=3, label=str(c), markersize=6, alpha=0.85)
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
        'fig': fig,
        'ax': ax,
        'scatters': scatters,
        'line': line,
        'clases': clases,
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

    if not fig_id:
        # cerrar todas las figuras registradas huérfanas
        for fid, s in list(_figures.items()):
            plt.close(s['fig'])
            del _figures[fid]
        import uuid
        fig_id = str(uuid.uuid4())

    if fig_id in _figures:
        if not plt.fignum_exists(_figures[fig_id]['fig'].number) or reset:
            del _figures[fig_id]

    if fig_id not in _figures:
        _init_figure(fig_id, entradas, salida, titulos)

    s = _figures[fig_id]
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
    Cierra y libera la figura asociada a fig_id, dejando visible el último frame.
    Script  : bloquea hasta que el usuario cierre la ventana, luego limpia.
    Notebook: no bloquea; el último frame ya quedó en el output via handle.update().
              Libera la entrada de _figures para que la próxima corrida no
              encuentre estado viejo ni acumule figuras huérfanas.
    """
    if fig_id not in _figures:
        return

    s = _figures[fig_id]

    if ENV == 'script':
        plt.ioff()
        plt.show(block=True)  # bloquea hasta que el usuario cierre la ventana
        plt.close(s['fig'])
    else:
        # Jupyter/Colab: no tocamos el output ya impreso (es el resultado final
        # que se quiere ver), solo liberamos memoria y dejamos de "trackear"
        # la figura para que no sea reusada ni vuelva a renderizarse.
        plt.close(s['fig'])

    del _figures[fig_id]
