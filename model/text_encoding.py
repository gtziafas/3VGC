from utils.utils import *


# extract features from text with 1D convolutions ??
class CNN1D(Module):
    def __init__(self, activation_fn: tensor_map = F.gelu, num_channels: int = 100,
                 kernel_sizes: Tuple[int, int, int] = (3, 4, 3)) -> None:
        super(CNN1D, self).__init__()
        self.activation_fn = activation_fn
        self.conv1 = ModuleList([self.conv_block(in_channels=1, out_channels=num_channels // 2,
                                                 conv_kernel=kernel_sizes[0], conv_stride=2),
                                 self.conv_block(in_channels=1, out_channels=num_channels // 2,
                                                 conv_kernel=kernel_sizes[1], conv_stride=2)])
        self.conv2 = self.conv_block(in_channels=num_channels, out_channels=num_channels,
                                     conv_kernel=kernel_sizes[2], conv_stride=kernel_sizes[2])

    @staticmethod
    def conv_block(in_channels: int, out_channels: int, conv_kernel: int, conv_stride: int,
                   pool_kernel: int = 2) -> Module:
        return Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
            MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel))

    def forward(self, x: FloatTensor) -> FloatTensor:
        batch_size = x.shape[0]

        x = x.view(batch_size, -1).unsqueeze(1)  # -> [B x 1 x S*d_embedd]
        x = [self.activation_fn(c(x)) for c in self.conv1]  # -> 2 x [B x 50 x (S*d_embedd)/4]
        x = torch.cat(x, dim=1)  # -> [B x 100 x (S*d_embedd)/4]
        x = self.activation_fn(self.conv2(x))  # -> [B x 100 x (S*d_embedd)/24]

        x = x.view(batch_size, -1)  # flatten for linear projection
        return x


# extract features from text with self-attention ??
class TransformerEncoderLayer(Module):
    def __init__(self, vector_size: int, num_heads: int = 6, d_ff: float = 2048, dropout_rate: float = 0.1) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.position_wise = PositionWiseFeedForward(d_model=vector_size, d_ff=d_ff, activation_fn=F.gelu)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_k=vector_size, d_model=vector_size, d_v=vector_size)
        self.layer_norm_1 = LayerNorm(vector_size)
        self.layer_norm_2 = LayerNorm(vector_size)
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
    def __init__(self, module_maker: Module, num_layers: int = 1, *args, **kwargs) -> None:
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding()
        self.network = Sequential(*[module_maker(*args, **kwargs) for _ in range(num_layers)])

    def forward(self, x: FloatTensor, mask: LongTensor) -> FloatTensor:
        x = self.positional_encoding(x)
        return self.network((x, mask))[0]


def default_text_model(dt_in: int = 24900, dt_out: int = 8000, num_classes: int = 8):
    encoder = ModalityEncoder(feature_extractor=CNN1D, d_in = dt_in, d_out = dt_out,
                              activation_fn = F.gelu, num_channels = 100, 
                              kernel_sizes=(3, 4, 3))

    classifier = ModalityClassifier(num_classes=num_classes, layer_dims = (dt_out, int(t_out/3), int(dt_out /8)))
    return Sequential(encoder, classifier)


def attention_text_model(dt_in: int = 24900, dt_out: int = 8000, num_classes: int = 8):
    transformer_enc = TransformerEncoder(nummodule_maker=TransformerEncoderLayer, num_layers=1,
                                         vector_size=300)

    cnn1d_enc = CNN1D(activation_fn=F.gelu, num_channels=100, kernel_sizes=(3,4,3))

    encoder = ModalityEncoder(feature_extractor=Sequential(transformer_enc, cnn1d_enc),
                              d_in = dt_in, d_out = dt_out)

    classifier = ModalityClassifier(num_classes=num_classes, layer_dims = (dt_out, int(t_out/3), int(dt_out /8)))
    return Sequential(encoder, classifier)


def text_example():
    d_t = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # text encoding
    max_seq_len, d_embedd = 20, 300
    d_feats = 24900  # because of CNN1D modules ~= max_seq_len * d_embedd / 24
    text_encoder = ModalityEncoder(feature_extractor=CNN1D,
                                   d_in=24900, d_out=d_t,
                                   activation_fn=F.gelu,
                                   num_channels=100,
                                   kernel_sizes=(3, 4, 3)).to(device)

    # placeholding
    classifier = Sequential(text_encoder,
                            Linear(d_t, 8)).to(device)

    # 10 random captions from consecutive 5-sec video samples
    batch_size = 10
    captions = torch.rand(batch_size, max_seq_len, d_embedd, device='cuda')
    prediction = classifier(captions)

    print(prediction)
