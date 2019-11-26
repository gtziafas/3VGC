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


# todo: TensorFusion, HierarchicalFusion