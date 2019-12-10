import torch

import torch.nn as nn 
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList, Linear, Sequential, RNN, LSTM, GRU, LayerNorm, Conv1d, Conv2d, Conv3d, MaxPool1d, MaxPool2d, MaxPool3d, BatchNorm2d

from typing import List, Tuple, Optional, Callable, Any, Sequence, TypeVar

tensor_map = Callable[[FloatTensor], FloatTensor]
tensor_maps = Sequence[tensor_map]

