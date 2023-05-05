from pathlib import Path

DSE_MODEL_DIR = Path(__file__).parent / "dse_models"


class DSEEngine:
    def __init__(self, model, dse_config):
        self.model = model
        self.dse_config = dse_config
