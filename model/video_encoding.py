from utils.imports import *
from utils.utils import ModalityEncoder, GeLU

# extracting features from videos with 3D convolutions
class CNN3D(Module):
    def __init__(self, activation_fn: Module, num_channels: int, kernel_sizes: List[Tuple[int, int]], dropout_rate: float = 0.0) -> None:
        super(CNN3D, self).__init__()
        self.activation_fn = activation_fn
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.depth = len(kernel_sizes)
        self.dropout_rate = dropout_rate

        self.feature_list = [self.conv_block(in_channels=3, out_channels=num_channels, conv_kernel=kernel_sizes[0][0], 
                        pool_kernel=kernel_sizes[0][1])]
        self.feature_list[1:] = [self.conv_block(in_channels = self.num_channels*(d+1), out_channels = self.num_channels*(d+2),
                            conv_kernel=kernel_sizes[d+1][0], pool_kernel=kernel_sizes[d+1][1]) for d in range(self.depth-1)]
        self.features = Sequential(*self.feature_list)

    def conv_block(self, in_channels: int, out_channels: int, conv_kernel: int, pool_kernel: int,
                   conv_stride: int = 2) -> Module:
        return Sequential(
            Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
            self.activation_fn,
            MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel),
            Dropout(p=self.dropout_rate))

    def forward(self, x: FloatTensor) -> FloatTensor:
        batch_size = x.shape[0]

        x = self.features(x) # multiple feature maps
        x = x.view(batch_size, -1) # flatten for linear projection
        return x

    def temp(self, x: FloatTensor) -> FloatTensor:
        for k in range(len(self.feature_list)):
            x = self.feature_list[k]
            print(x.shape)
        return x.view(x.shape[0], -1)


def video_example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, height, width = 20, 64, 64  # no frames, frame height, frame width 
    batch_size = 4 
    d_v = 9600

    # noise video input
    video = torch.rand(batch_size, 3, frames, height, width, device=device)

    # define the 3D CNN + linear encoder
    video_encoder = ModalityEncoder(feature_extractor=CNN3D,
                                    d_in = d_v , d_out = 1024,
                                    activation_fn = GeLU(),
                                    num_channels = 32, 
                                    kernel_sizes = [(4,3)])
    classifier = Sequential(video_encoder, Linear(1024, 8)).to(device)

    # placeholding results - no softmax
    prediction = classifier(video)
    print(prediction)

