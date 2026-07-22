import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


class ImageCard:
    """
    Representación visual de una imagen.

    Proporciona tres secciones HTML:

        header
        content
        footer

    que luego serán utilizadas por CardViewer o GridCardViewer.

    Soporta imágenes en formato RGB, escala de grises (GRAY)
    o espacios de color HSV/HSL (se convierten a RGB solo
    para visualización, sin modificar el array original).
    """

    def __init__(
        self,
        image,
        format="RGB",
        colormap="gray",
        title="",
        subtitle="",
        show_histogram=True,
        figsize=(3.5, 2.0),
        hist_figsize=(3.5, 1.2),
    ):

        self.image = image
        self.format = format.upper()

        self.colormap = colormap

        self.title = title
        self.subtitle = subtitle

        self.show_histogram = show_histogram

        self.figsize = figsize
        self.hist_figsize = hist_figsize

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    @property
    def header(self):

        html = ""

        if self.title:
            html += (
                f"<div style='font-weight:bold;"
                f"font-size:14px;'>"
                f"{self.title}"
                f"</div>"
            )

        if self.subtitle:
            html += (
                f"<div style='font-size:12px;"
                f"color:#666;'>"
                f"{self.subtitle}"
                f"</div>"
            )

        return html

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    @property
    def content(self):

        image_b64 = self._image_base64()

        return (
            f"<img "
            f"src='data:image/png;base64,{image_b64}' "
            f"style='width:100%;height:auto;'>"
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    @property
    def footer(self):

        html = (
            f"<div style='font-size:11px;"
            f"color:#999;margin-top:4px;'>"
            f"{self.format} · {self._shape_str()}"
            f"</div>"
        )

        if self.show_histogram:
            hist_b64 = self._histogram_base64()
            html += (
                f"<img "
                f"src='data:image/png;base64,{hist_b64}' "
                f"style='width:100%;height:auto;margin-top:4px;'>"
            )

        return html

    # ------------------------------------------------------------------

    def _shape_str(self):

        shape = self.image.shape

        if len(shape) == 2:
            return f"{shape[0]}x{shape[1]}"

        return f"{shape[0]}x{shape[1]}x{shape[2]}"

    # ------------------------------------------------------------------

    def _display_rgb(self):
        """
        Devuelve la imagen lista para mostrarse con imshow,
        convirtiendo a RGB si hace falta (HSL/HSV), sin
        modificar self.image.
        """

        img = self.image

        if self.format == "HSV":
            return hsv_to_rgb(img)

        if self.format == "HSL":
            return self._hsl_to_rgb(img)

        return img

    # ------------------------------------------------------------------

    def _hsl_to_rgb(self, img):

        h, l, s = img[..., 0], img[..., 1], img[..., 2]

        # HSL -> HSV -> RGB (vectorizado, evita colorsys por pixel)
        v = l + s * np.minimum(l, 1 - l)
        s_hsv = np.where(v == 0, 0, 2 * (1 - l / np.where(v == 0, 1, v)))

        hsv = np.stack([h, s_hsv, v], axis=-1)

        return hsv_to_rgb(hsv)

    # ------------------------------------------------------------------

    def _image_base64(self):

        fig, ax = plt.subplots(figsize=self.figsize)

        if self.format == "GRAY":
            ax.imshow(self.image, cmap=self.colormap)
        else:
            ax.imshow(self._display_rgb())

        ax.axis("off")

        plt.tight_layout()

        buffer = io.BytesIO()

        plt.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
        )

        plt.close(fig)

        return base64.b64encode(buffer.getvalue()).decode()

    # ------------------------------------------------------------------

    def _histogram_base64(self):

        fig, ax = plt.subplots(figsize=self.hist_figsize)

        self._plot_histogram(ax)

        plt.tight_layout()

        buffer = io.BytesIO()

        plt.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
        )

        plt.close(fig)

        return base64.b64encode(buffer.getvalue()).decode()

    # ------------------------------------------------------------------

    def _plot_histogram(self, ax):
        """
        Dibuja el histograma de la imagen sobre un ax ya
        creado. Separado de _histogram_base64 para poder
        reutilizarlo en variantes que combinan imagen +
        histograma en un mismo layout (ver ImageHistCard).
        """

        if self.format == "GRAY":
            ax.hist(
                self.image.ravel(),
                bins=256,
                color="gray",
                alpha=0.8,
            )

        elif self.format == "RGB":
            colors = ("red", "green", "blue")
            for i, color in enumerate(colors):
                ax.hist(
                    self.image[..., i].ravel(),
                    bins=256,
                    color=color,
                    alpha=0.5,
                    histtype="step",
                )

        else:
            # HSL / HSV u otros: canales sin correspondencia
            # visual directa con RGB, así que uso colores
            # neutros y etiquetas por canal en vez de rojo/
            # verde/azul.
            labels = ("canal 1", "canal 2", "canal 3")
            colors = ("#888888", "#e08214", "#3182bd")
            for i, (label, color) in enumerate(zip(labels, colors)):
                ax.hist(
                    self.image[..., i].ravel(),
                    bins=256,
                    color=color,
                    alpha=0.5,
                    histtype="step",
                    label=label,
                )
            ax.legend(fontsize=6, loc="upper right")

        ax.set_xticks([])
        ax.set_yticks([])