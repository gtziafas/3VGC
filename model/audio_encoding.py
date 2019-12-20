from utils.imports import *
from utils.utils import ModalityEncoder

import torchaudio
import torchaudio.transforms as tf

<<<<<<< HEAD
def wav_to_mfcc(wav: Sequence[float], sample_rate: float, device: str = 'cpu') -> FloatTensor:
	wav = torch.tensor(wav, dtype=torch.float, device=device)
	return tf.MFCC(sample_rate)(wav)

def wav_to_mel_spec(wav: Sequence[float], sample_rate:float, device: str = 'cpu') -> FloatTensor:
	wav = torch.tensor(wav, dtype=torch.float, device=device)
	return tf.MelSpectrogram(sample_rate)(wav) 
=======

def wav_to_mfcc(wav: Sequence[float], sample_rate: float) -> FloatTensor:
    wav = torch.tensor(wav, dtype=torch.float)
    return tf.MFCC(sample_rate)(wav)


def wav_to_mel_spec(wav: Sequence[float], sample_rate: float) -> FloatTensor:
    wav = torch.tensor(wav, dtype=torch.float)
    return tf.MelSpectrogram(sample_rate)(wav)

>>>>>>> 00160478d5e1c5c939918acb8b7f279f697cfc8e

class DempsterShaferFusion(Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DempsterShaferFusion, self).__init__()
        NotImplemented

    def forward(self, x: FloatTensor) -> FloatTensor:
        NotImplemented


# extract audio features from Log-Mel Spectrogram + Mel-Freq Coeffs + Chroma + Spectral Contrast + Tonnentz ??
class TwoStreamCNN(Module):
    def __init__(self, log_mel_stream: Module, mel_freq_stream: Module, fusion_fn: tensor_map) -> None:
        super(TwoStreamCNN, self).__init__()
        self.log_mel_stream = log_mel_stream
        self.mel_freq_stream = mel_freq_stream
        self.fusion = fusion_fn

    def forward(self, x: Tuple[FloatTensor]) -> FloatTensor:
        # input x is a Tuple containing all 5 features
        mfcc, log_mel, chroma, spectral_contrast, tonnentz = x[0:5]

        # sanity
        assert mfcc.shape[3] == chroma.shape[3]
        assert chroma.shape[3] == log_mel.shape[3]
        assert log_mel.shape[3] == spectral_contrast.shape[3]
        assert tonnentz.shape[3] == spectral_contrast.shape[3]

        # stack last 3 features with two spectrograms
        x1 = torch.cat([mfcc, chroma, spectral_contrast, tonnentz], dim=2)
        x2 = torch.cat([log_mel, chroma, spectral_contrast, tonnentz], dim=2)
        x1 = self.mel_freq_stream(x1)
        x2 = self.log_mel_stream(x2)
        x = self.fusion([x1, x2])

        return x


class CNN2D(Module):
    def __init__(self, activation_fn: tensor_map = F.gelu, in_channels: int = 2, dropout_rate: float = 0.5) -> None:
        super(CNN2D, self).__init__()
        self.activation_fn = activation_fn
        self.conv1 = self.conv_block(in_channels=in_channels, out_channels=32, conv_kernel=3, conv_stride=2)
        self.conv2 = self.conv_block(in_channels=32, out_channels=32, conv_kernel=3, conv_stride=2, pool_kernel=2)
        self.conv3 = self.conv_block(in_channels=32, out_channels=64, conv_kernel=3, conv_stride=2)
        self.conv4 = self.conv_block(in_channels=64, out_channels=64, conv_kernel=3, conv_stride=2)
        self.linear = Linear(in_features=25344, out_features=2048)
        self.dropout_rate = dropout_rate

    def conv_block(self, in_channels: int, out_channels: int, conv_kernel: int, conv_stride: int,
                   pool_kernel: int = 0) -> Module:
        block = [
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
            BatchNorm2d(num_features=out_channels)]
        if pool_kernel:
            block.append(MaxPool2d(kernel_size=pool_kernel))
        return Sequential(*block)

    def forward(self, x: FloatTensor) -> FloatTensor:
        batch_size = x.shape[0]

        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        x = F.dropout(x, p=self.dropout_rate)
        x = self.activation_fn(self.conv3(x))
        x = self.activation_fn(self.conv4(x))
        x = F.dropout(x, p=self.dropout_rate)
        x = torch.sigmoid(self.linear(x.view(batch_size, -1)))

        return x


def audio_example():
    d_a = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path_to_wav_files = ["../audio_junk/vlog5.wav", "../audio_junk/sports5.wav"]
    wavs = [torchaudio.load(p, normalization=True)[0] for p in path_to_wav_files]
    sample_rates = [torchaudio.load(p, normalization=True)[1] for p in path_to_wav_files]

    mel_freqs = torch.stack(
        [tf.MelSpectrogram(sample_rate)(wav).to(device) for wav, sample_rate in zip(wavs, sample_rates)])
    mfccs = torch.stack([tf.MFCC(sample_rate)(wav).to(device) for wav, sample_rate in zip(wavs, sample_rates)])

    # somehow
    chroma = torch.empty_like(mel_freqs)
    spectral_contrast = torch.empty_like(mel_freqs)
    tonnentz = torch.empty_like(mel_freqs)

    from model.fusion import EarlyFusion

    audio_enc = ModalityEncoder(feature_extractor=TwoStreamCNN,
                                d_in=4096, d_out=1024,
                                log_mel_stream=CNN2D(),
                                mel_freq_stream=CNN2D(),
                                fusion_fn=EarlyFusion).to(device)

    # placeholding
    classifier = Sequential(audio_enc, Linear(512, 8))

    predictions = classifier((mfccs, mel_freqs, chroma, spectral_contrast, tonnentz))
    print(predictions)
