import torch

import torch.nn as nn 
import torch.nn.functional as F
from torch import FloatTensor, LongTensor 
from torch.nn import Module, ModuleList, Linear, Sequential, RNN, LSTM, GRU, LayerNorm

from typing import List, Tuple, Optional, Callable, Any, Sequence, TypeVar

tensor_map = Callable[[FloatTensor], FloatTensor]
tensor_maps = Sequence[tensor_map]

