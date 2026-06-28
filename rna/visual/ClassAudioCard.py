import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class AudioCard:
    """
    Representación visual de un audio.

    Proporciona tres secciones HTML:

        header
        content
        footer

    que luego serán utilizadas por CardViewer o GridCardViewer.
    """

    def __init__(
        self,
        audio,
        sample_rate,
        title="",
        subtitle="",
        figsize=(3.5, 2.0),
    ):

        self.audio = audio
        self.sample_rate = sample_rate

        self.title = title
        self.subtitle = subtitle

        self.figsize = figsize

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

        waveform = self._waveform_base64()

        return (
            f"<img "
            f"src='data:image/png;base64,{waveform}' "
            f"style='width:100%;height:auto;'>"
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    @property
    def footer(self):

        audio = self._audio_base64()

        return f"""
        <audio controls style="width:100%; margin-top:6px;">
            <source
                src="data:audio/wav;base64,{audio}"
                type="audio/wav">
        </audio>
        """

    # ------------------------------------------------------------------

    def _mono_audio(self):

        signal = self.audio

        if signal.ndim > 1:
            signal = signal[:, 0]

        return signal

    # ------------------------------------------------------------------

    def _waveform_base64(self):

        signal = self._mono_audio()

        fig, ax = plt.subplots(figsize=self.figsize)

        t = np.arange(len(signal)) / self.sample_rate

        ax.plot(t, signal)

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")

        ax.grid(True)

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

    def _audio_base64(self):

        signal = self._mono_audio()

        buffer = io.BytesIO()

        sf.write(
            buffer,
            signal,
            self.sample_rate,
            format="wav",
        )

        return base64.b64encode(buffer.getvalue()).decode()