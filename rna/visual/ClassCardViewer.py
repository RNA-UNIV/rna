from IPython.display import HTML, display


class CardViewer:

    CARD_STYLE = """
        border:1px solid #dddddd;
        border-radius:8px;
        padding:8px;
        background:#fafafa;
        text-align:center;
    """

    def __init__(self, card):

        self.card = card

    def html(self):

        return f"""
        <div style="{self.CARD_STYLE}">

            {self.card.header}

            {self.card.content}

            {self.card.footer}

        </div>
        """

    def show(self):

        display(HTML(self.html()))