from utils.imports import *
from utils.utils import PositionWiseFeedForward, MultiHeadAttention, PositionalEncoding

# extracting features from videos with 3D convolutions
class CNN3D(Module):
	def __init__(self, *args, **kwargs) -> None:
		super(CNN3D, self).__init__()
		NotImplemented
	def forward(self, x: FloatTensor) -> FloatTensor:
		NotImplemented

# extract features from text with 1D convolutions ??
class CNN1D(Module):
	def __init__(self, *args, **kwargs) -> None:
		super(CNN1D, self).__init__()
		NotImplemented
	def forward(self, x: FloatTensor) -> FloatTensor:
		NotImplemented


# audio ?


# extract features from text with self-attention ??
class TransformerEncoderLayer(Module):
	def __init__(self, vector_size: int, num_heads: int=1, d_ff: float=2048, dropout_rate: float=0.1) -> None:
		super(TransformerEncoderLayer, self).__init__()
		self.position_wise = PositionWiseFeedForward(d_model=vector_size, d_ff=d_ff, activation_fn=F.gelu)
		self.mha = MultiHeadAttention(num_heads=num_heads, d_k=d_model, d_model=d_model, d_v=d_model)
		self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tuple[FloatTensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        inputs, mask = x[0], x[1]
        attended = self.mha(inputs, inputs, inputs, mask)
        attended = attended + inputs
        attended_norm = self.layer_norm_1(attended)
        attended_norm = F.dropout(attended_norm, self.dropout_rate)

        transformed = self.position_wise(attended_norm)
        transformed = attended_norm + transformed
        transformed_norm = self.layer_norm_2(transformed)
        transformed_norm = F.dropout(transformed_norm, self.dropout_rate)

        return transformed_norm, mask

class TransformerEncoder(Module):
    def __init__(self, module_maker: Module, num_layers: int, *args, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.positional_encoding = PositionalEncoding()
        self.network = Sequential(*[module_maker(*args, **kwargs) for _ in range(num_layers)])

    def forward(self, x: FloatTensor, mask: LongTensor) -> Tensor:
    	x = self.positional_encoding(x)
        return self.network((x, mask))[0]

# wrapper for all three encoders
class ModalityEncoder(Module):
	def __init__(self, features: Module, d_in: int, d_out: int) -> None:
		super(ModalityEncoder, self).__init__()
		self.features = features 
		self.linear = Linear(d_in, d_out)

	def forward(self, x:FloatTensor) -> FloatTensor:
		features = self.features(x)
		out = self.linear(features)
		out = F.softmax(out, dim=-1)

		return out 



