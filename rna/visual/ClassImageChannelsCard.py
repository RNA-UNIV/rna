import base64
import io

import matplotlib.pyplot as plt

from rna.visual.ClassImageCard import ImageCard


class ImageChannelsCard(ImageCard):
    """
    Variante de ImageCard que además muestra los 3 canales
    por separado, en un layout de 2x2 (imagen completa +
    canal 1, canal 2, canal 3).

    Si la imagen es GRAY (sin 3 canales), cae al
    comportamiento simple de ImageCard.
    """

    _CHANNEL_INFO = {
        "RGB": (("R", "Reds"), ("G", "Greens"), ("B", "Blues")),
        "HSV": (("H", "gray"), ("S", "gray"), ("V", "gray")),
        "HSL": (("H", "gray"), ("S", "gray"), ("L", "gray")),
    }

    @property
    def content(self):

        if self.format == "GRAY" or self.image.ndim != 3:
            # Sin 3 canales, no hay grid 2x2 posible.
            return super().content

        image_b64 = self._channels_grid_base64()

        return (
            f"<img "
            f"src='data:image/png;base64,{image_b64}' "
            f"style='width:100%;height:auto;'>"
        )

    # ------------------------------------------------------------------

    def _channels_grid_base64(self):

        fig, axes = plt.subplots(
            2, 2,
            figsize=(self.figsize[0] * 1.6, self.figsize[1] * 1.6),
        )

        ax_main, ax_c1, ax_c2, ax_c3 = axes.ravel()

        # --- Panel principal: imagen completa ---
        ax_main.imshow(self._display_rgb())
        ax_main.set_title("Imagen", fontsize=9)
        ax_main.axis("off")

        # --- Paneles de canales individuales ---
        channel_info = self._CHANNEL_INFO.get(
            self.format,
            (("C1", "gray"), ("C2", "gray"), ("C3", "gray")),
        )

        for idx, (ax, (label, cmap)) in enumerate(
            zip((ax_c1, ax_c2, ax_c3), channel_info)
        ):
            ax.imshow(self.image[..., idx], cmap=cmap)
            ax.set_title(label, fontsize=9)
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