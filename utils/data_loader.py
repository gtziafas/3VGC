from utils.imports import *

from random import sample
from abc import abstractmethod, ABC

# GloVe word embeddings
import spacy

nlp = spacy.load('en_core_web_md')

# path to data in Peregrine 
text_data_path = '/data/s3913171/3VGC/subs_full.txt'
video_data_dir = '/data/s3913171/3VGC/videos_full/'
audio_data_dir = '/data/s3913171/3VGC/audio_full/'


def parse(line: str) -> TextSample:
    return line.split(' ')


def shuffle_chunk(chunk: TextSamples) -> TextSamples:
    return sample(chunk, len(chunk))


# define an abstract DataLoader class
class DataLoader(ABC):
    @abstractmethod
    def __next__(self) -> TextSample:
        pass

    @abstractmethod
    def get_batch(self) -> TextSamples:
        pass

    @abstractmethod
    def get_processed_batch(self) -> Any:
        pass


def VideoDataset():
    pass


def AudioDataset():
    pass


# a custom lazy DataLoader class for loading text samples as sequences of strings
class TextDataLoader(DataLoader):
    def __init__(self, filepath: str, chunk_size: int, batch_size: int,
                 post_proc: Callable[[TextSamples], Any]) -> None:
        self.filepath = filepath
        self.line_iterator = open(self.filepath, 'r')
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk = self.get_contiguous_chunk
        self.post_proc = post_proc

    # chunk a part of the input training data and shuffle it locally, return an iterator over them
    def get_contiguous_chunk(self) -> Iterator[TextSample]:
        return iter(shuffle_chunk([parse(self.get_next_line()) for _ in range(self.chunk_size)]))

    # get next training sample, or restart from top if reached the end
    def get_next_line(self) -> str:
        try:
            return self.line_iterator.__next__()
        except StopIteration:
            self.line_iterator = open(self.filepath, 'r')
            return self.line_iterator.__next__()

    def __next__(self) -> TextSample:
        try:
            return self.chunk.__next__()
        except StopIteration:
            self.chunk = self.get_contiguous_chunk()
            return self.chunk.__next__()

    def get_batch(self) -> TextSamples:
        return [self.__next__() for _ in range(self.batch_size)]

    def get_processed_batch(self) -> Any:
        return self.post_proc(self.get_batch())


# a TextDataLoader for our text dataset, wrapped to give SpaCy word vectors
def default_text_dataloader(path: str = text_data_path, chunk_size: int = 10000,
                            batch_size: int = 64, len_threshold: int = 25) -> TextDataLoader:
    # convert input strings to GLoVe vectors
    def glove_embeddings(sentences: TextSamples) -> FloatTensors:
        sentences = list(filter(lambda sentence: len(sentence) < len_threshold, sentences))
        docs = list(map(nlp, sentences))
        vectors = map(lambda doc: [word.vector for word in doc], docs)
        return list(map(lambda sentence_vectors: torch.tensor(sentence_vectors, dtype=torch.float), vectors))

    return TextDataLoader(path, chunk_size, batch_size, glove_embeddings)
