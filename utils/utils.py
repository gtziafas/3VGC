from utils.imports import *


# wrapper for all three encoders
class ModalityEncoder(Module):
    def __init__(self, feature_extractor: Module, d_in: int, d_out: int, *args, **kwargs) -> None:
        super(ModalityEncoder, self).__init__()
        self.features = feature_extractor(*args, **kwargs)
        self.linear = Linear(in_features=d_in, out_features=d_out)

    def forward(self, x: FloatTensor) -> FloatTensor:
        features = self.features(x)
        out = self.linear(features)
        out = F.softmax(out, dim=-1)

        return out


# wrapper for all three unimodal classifiers
class ModalityClassifier(Module):
    def __init__(self, layer_dims: Tuple[int], activation_fn: tensor_map, dropout_rate: float = 0.2, num_classes: int = 8) -> None:
        self.depth = len(layer_dims)
        self.activation_fn = activation_fn 
        self.dropout_rate = dropout_rate 

        self.ws = ModuleList(list(map(lambda d: Linear(self.layer_dims[d], self.layer_dims[d+1]), range(self.depth-1))))
        self.ws.append(Linear(self.layer_dims[-1], num_classes))

    def forward(self, x: FloatTensor) -> FloatTensor: 
        for d in self.depth:
            x = F.dropout(self.activation_fn(self.ws[d](x)), self.dropout_rate)

        return self.ws[-1](x)


class PositionWiseFeedForward(Module):
    def __init__(self, d_model: int, activation_fn: tensor_map, d_ff: int, dropout_rate: float = 0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

    def forward(self, x):
        w1 = self.w_1(x)
        w1 = F.dropout(self.activation_fn(w1), self.dropout_rate)

        return self.w_2(w1)


# attn(Q,K,V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
def scaled_dot_product(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                       mask: Optional[LongTensor] = None) -> FloatTensor:
    dk = keys.shape[-1]
    dividend = torch.tensor(dk, device=queries.device, dtype=torch.float)

    weights = torch.bmm(queries, keys.transpose(2, 1)) / torch.sqrt(dividend)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)  # [B, M, N] -- each m thing attends a probability distribution over N things
    return torch.bmm(weights, values)


# mha(Q,K,V) = h x attn(Q @ W_1, K @ W_2, V @ W_3) @ W_0
def multihead_attn_fn(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                      qts: tensor_maps, kts: tensor_maps, vts: tensor_maps, wo: tensor_map,
                      mask: Optional[LongTensor] = None, dropout_rate: float = 0.1) -> FloatTensor:
    qs = [qt(queries) for qt in qts]
    ks = [kt(keys) for kt in kts]
    vs = [vt(values) for vt in vts]
    outputs = [scaled_dot_product(qs[i], ks[i], vs[i], mask) for i in range(len(qs))]
    outputs = F.dropout(torch.cat(outputs, dim=-1), dropout_rate)
    return wo(outputs)


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.q_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.k_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.v_transformations = ModuleList([Linear(in_features=d_model, out_features=d_v, bias=False)
                                             for _ in range(num_heads)])
        self.Wo = Linear(in_features=num_heads * d_v, out_features=d_model, bias=False)

    def forward(self, queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                mask: Optional[LongTensor] = None) -> FloatTensor:
        return multihead_attn_fn(queries, keys, values, self.q_transformations, self.k_transformations,
                                 self.v_transformations, self.Wo, mask)


# x -> x + pe(x)
class PositionalEncoding(Module):
    def __init__(self, freq: int = 10000, dropout_rate: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.freq = freq
        self.dropout_rate = dropout_rate

    def forward(self, x: FloatTensor) -> FloatTensor:
        b, n, d_model = x.shape[0:3]
        pe = torch.zeros(n, d_model, device=x.device)
        position = torch.arange(0, n, device=x.device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device, dtype=torch.float) *
                             - (torch.log(torch.tensor(self.freq, dtype=torch.float, device=x.device)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.repeat(b, 1, 1)

        return F.dropout(pe, self.dropout_rate) + x


# the gelu activation fn 
class GeLU(Module):
    def __init__(self) -> None:
        super(GeLU, self).__init__() 

    def forward(self, x: FloatTensor) -> FloatTensor:
        return F.gelu(x)