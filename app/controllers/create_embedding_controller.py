from app.params import CreateEmbeddingAPIParam

class CreateEmbeddingController:
    def __init__(self, params: CreateEmbeddingAPIParam) -> None:
        self.params = params