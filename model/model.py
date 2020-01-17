from utils.imports import *
from model.contextualization import ContextualizationLayer
from model.fusion import EarlyFusion


class TrimodalFusionClassifier(Module):
    def __init__(self, encoders: ModuleList, model_sizes: Tuple[int, int, int], fusion_fn: tensor_map,
                 feature_sizes: Tuple[int, int, int], num_classes: int = 8, dropout_rate: float = 0.1) -> None:
        super(TrimodalFusionClassifier, self).__init__()
        self.d_equ, self.d_hidden, self.d_fused = model_sizes[0:3]
        self.video_encoder, self.audio_encoder, self.text_encoder = encoders[0:3]
        self.dim_equ = ModuleList([Linear(feature_sizes[_], self.d_equ) for _ in range(3)])
        self.context = ModuleList(
            [ContextualizationLayer(inp_dim=self.d_equ, hidden_dim=self.d_hidden) for _ in range(3)])
        self.classifier = Linear(self.d_fused, num_classes)
        self.fusion = fusion_fn
        self.dropout_rate = dropout_rate

    def forward(self, x: Tuple[FloatTensor, FloatTensor, FloatTensor]) -> FloatTensor:
        video, audio, text = x[0:3]

        # get feature vectors from each modality (pre-trained / end-to-end) -> [3 x B x S x {d_v , d_a , d_t}]
        encodings = [self.video_encoder(video), self.audio_encoder(audio), self.text_encoder(text)]

        # modality dim equalization -> [3 x B x S x d_equ]
        encodings_equ = [self.dim_equ[i](e) for i, e in enumerate(encodings)]
        encodings_equ = [F.dropout(F.gelu(e), p=self.dropout_rate) for e in encodings_equ]

        # contextualize feature vectors -> [3 x B x 2*d_hidden]
        context = [self.context[i](e) for i, e in enumerate(encodings_equ)]
        context = [F.dropout(c, p=self.dropout_rate) for c in context]

        # fusion method -> [B x d_fused]
        fused = self.fusion(context)
        fused = F.dropout(fused, p=self.dropout_rate)

        # map to labels -> [B x num_classes]
        out = self.classifier(fused)

        return out


def test():
    d_v = 10
    d_a = 10
    d_t = 10
    d_equ = 10
    d_hidden = 5
    d_fused = 3 * 2 * d_hidden

    # placeholding linear layers for encoding
    video_encoder = Linear(3 * 15 * 10 * 10, d_v)
    audio_encoder = Linear(60, d_a)
    text_encoder = Linear(10, d_t)

    enc_listed = ModuleList([video_encoder, audio_encoder, text_encoder])
    fusion_fn = EarlyFusion()

    model = TrimodalFusionClassifier(encoders=enc_listed, model_sizes=(d_equ, d_hidden, d_fused),
                                     feature_sizes=(d_v, d_a, d_t), fusion_fn=fusion_fn).to('cuda')

    v = torch.rand(2, 5, 3, 15, 10, 10, device='cuda').view(2, 5, -1)
    a = torch.rand(2, 5, 60, device='cuda')
    t = torch.rand(2, 5, 10, device='cuda')

    out = model((v, a, t))

    print(out.shape, out)
