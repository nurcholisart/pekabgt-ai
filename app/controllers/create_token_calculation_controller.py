import tiktoken
from app.params import CreateTokenCalculationAPIParam
from app.responses import CreateTokenCalculationAPIResponse
from bs4 import BeautifulSoup

DEFAULT_MODEL = "gpt-4"
COST_PER_1000_TOKEN = 0.0004


class CreateTokenCalculationController:
    def __init__(self, params: CreateTokenCalculationAPIParam) -> None:
        self.params = params

    def __call__(self) -> CreateTokenCalculationAPIResponse:
        enc = tiktoken.encoding_for_model(self.params.model)

        words_count = sum(len(content.split()) for content in self.plain_contents)
        tokens_count = sum(len(enc.encode(content)) for content in self.plain_contents)
        estimated_cost = "{:.7f}".format((tokens_count * COST_PER_1000_TOKEN / 1000))

        return CreateTokenCalculationAPIResponse(
            words_count=words_count,
            tokens_count=tokens_count,
            estimated_cost=estimated_cost,
        )

    @property
    def plain_contents(self) -> list[str]:
        self._plain_contents: list[str] = []

        if len(self._plain_contents) > 0:
            return self._plain_contents

        for content in self.params.contents:
            soup = BeautifulSoup(content, "html.parser")
            plain_content = soup.get_text()
            self._plain_contents.append(plain_content)

        return self._plain_contents
