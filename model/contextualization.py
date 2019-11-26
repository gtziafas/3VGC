from utils.imports import * 

# take a sequence of samples IR^{S x inp_dim} and give a code vector in IR^2*hidden_dim
class ContextualizationLayer(Module):
	def __init__(self, inp_dim: int, hidden_dim: int) -> None:
		super(ContextualizationLayer, self).__init__()
		self.inp_dim = inp_dim
		self.hidden_dim = hidden_dim

		#self.context = RNN(input_size=inp_dim, hidden_size=hidden_dim, nonlinearity='tanh') ## ELMAN ##
		#self.context = LSTM(input_size=inp_dim, hidden_size=hidden_dim, bidirectional=True) ## bi-LSTM ##
		self.context = GRU(input_size=inp_dim, hidden_size=hidden_dim, bidirectional=True)   ## bi-GRU ##

	def forward(self, x: FloatTensor) -> FloatTensor:
		h, _ = self.context(x)
		# return only last hidden state
		return h[:,-1,:] 