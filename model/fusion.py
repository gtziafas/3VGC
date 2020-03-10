from utils.imports import *


class EarlyFusion(Module):
    def __init__(self) -> None:
        super(EarlyFusion, self).__init__()

    def forward(self, tensor_list: List[FloatTensor]) -> FloatTensor:
        return torch.cat(tensor_list, dim=-1)


class MaxFusion(Module):
    def __init__(self) -> None:
        super(MaxFusion, self).__init__()

    def forward(self, tensor_list: List[FloatTensor]) -> FloatTensor:
        return torch.max(torch.stack(tensor_list), dim=0)[0]


class AverageFusion(Module):
    def __init__(self) -> None:
        super(AverageFusion, self).__init__()

    def forward(self, tensor_list: List[FloatTensor]) -> FloatTensor:
        return torch.mean(torch.stack(tensor_list), dim=0)


class TensorFusion(Module):
    def __init__(self) -> None:
        super(TensorFusion, self).__init__()

    def forward(self, tensor_list: List[FloatTensor]) -> FloatTensor:
        batch_size, vector_size = tensor_list[0].shape[0], tensor_list[0].shape[1]
        vectors = [tensor.flatten() for tensor in tensor_list]

        outter_2d = torch.ger(vectors[0], vectors[1])
        outter_3d = torch.ger(outter_2d.flatten(), vectors[2]).view(batch_size, vector_size, vector_size, vector_size)

        return outter_3d

# TODO: HierarchicalFusion
