from .transformer import TransformerForecaster
from .s5 import S5Forecaster
from .mamba_model import MambaForecaster
from .xlstm_model import XLSTMForecaster

MODEL_REGISTRY = {
    "transformer": TransformerForecaster,
    "s5": S5Forecaster,
    "mamba": MambaForecaster,
    "xlstm": XLSTMForecaster,
}
