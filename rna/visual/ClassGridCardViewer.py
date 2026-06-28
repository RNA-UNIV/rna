from IPython.display import HTML, display

from rna.visual.ClassCardViewer import CardViewer


class GridCardViewer:

    def __init__(
        self,
        cards,
        cols=4,
    ):

        self.cards = cards
        self.cols = cols

    def html(self):

        html = """
        <table
            style="
                width:100%;
                border-collapse:collapse;
                table-layout:fixed;
            ">
        """

        for i in range(0, len(self.cards), self.cols):

            html += "<tr>"

            row = self.cards[i:i+self.cols]

            for card in row:

                viewer = CardViewer(card)

                html += f"""
                <td
                    style="
                        padding:6px;
                        vertical-align:top;
                    ">
                    {viewer.html()}
                </td>
                """

            while len(row) < self.cols:

                html += "<td></td>"
                row.append(None)

            html += "</tr>"

        html += "</table>"

        return html

    def show(self):

        display(HTML(self.html()))