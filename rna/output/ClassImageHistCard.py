import base64
import io

import matplotlib.pyplot as plt

from rna.output.ClassImageCard import ImageCard


class ImageHistCard(ImageCard):
    """
    Variante de ImageCard que muestra la imagen junto a su
    histograma en un layout 1x2 dentro del content.

    Fuerza show_histogram=False para que el footer no
    duplique el histograma que ya se muestra en el content.
    """

    def __init__(self, *args, **kwargs):

        kwargs["show_histogram"] = False

        super().__init__(*args, **kwargs)

    @property
    def content(self):

        image_b64 = self._image_hist_base64()

        return (
            f"<img "
            f"src='data:image/png;base64,{image_b64}' "
            f"style='width:100%;height:auto;'>"
        )

    # ------------------------------------------------------------------

    def _image_hist_base64(self):

        fig, (ax_img, ax_hist) = plt.subplots(
            1, 2,
            figsize=(self.figsize[0] * 1.8, self.figsize[1]),
        )

        # --- Panel imagen ---
        if self.format == "GRAY":
            ax_img.imshow(self.image, cmap=self.colormap)
        else:
            ax_img.imshow(self._display_rgb())
        ax_img.axis("off")

        # --- Panel histograma (reusa la lógica de ImageCard) ---
        self._plot_histogram(ax_hist)

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