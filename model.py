from module import *

class MODEL(nn.Module):
	def __init__(self, C, V1, V2, V3, stride1, stride2):
		super().__init__()

		self.embedding1 = nn.Embedding(V1, C)
		self.embedding2 = nn.Embedding(V2, C)
		self.embedding3 = nn.Embedding(V3, C)
		self.compress1 = InvertibleCompress(C, stride=stride1)
		self.compress2 = InvertibleCompress(C, stride=stride2)
		self.linear_compress1 = nn.Linear(C * stride1, C)
		self.linear_compress2 = nn.Linear(C * stride2, C)
		self.linear_reverse1 = nn.Linear(C, C * stride1)
		self.linear_reverse2 = nn.Linear(C, C * stride2)
		self.film = FiLM(C, 3)

		self.block1 = nn.ModuleList([RWKVBlockRosaFree(C, layer_id=layer, n_layer=6) for layer in range(6)])
		self.block2 = nn.ModuleList([RWKVBlockRosaFree(C, layer_id=layer, n_layer=6) for layer in range(6)])
		self.block3 = nn.ModuleList([RWKVBlockRosaFree(C, layer_id=layer, n_layer=6) for layer in range(6)])
		self.concat = nn.ModuleList([RWKVBlockRosaFree(C, layer_id=layer, n_layer=4) for layer in range(4)])

		self.lno = nn.LayerNorm(C)
		self.o1 = nn.Linear(C, V1)
		self.o2 = nn.Linear(C, V2)
		self.o3 = nn.Linear(C, V3)

	def forward(self, x1, x2, x3):
		x1 = self.embedding1(x1)
		x2 = self.embedding2(x2)
		x3 = self.embedding3(x3)
		x1 = self.linear_compress1(self.compress1(x1))
		x2 = self.linear_compress2(self.compress2(x2))

		v_first = torch.zeros_like(x1)
		for block in self.block1:
			x1, v_first = block(x1, v_first)
		x1_output = self.film(x1, torch.full((x1.shape[0],), 0, dtype=torch.long, device=x1.device))

		v_first = torch.zeros_like(x2)
		for block in self.block2:
			x2, v_first = block(x2, v_first)
		x2_output = self.film(x2, torch.full((x2.shape[0],), 1, dtype=torch.long, device=x2.device))

		v_first = torch.zeros_like(x3)
		for block in self.block3:
			x3, v_first = block(x3, v_first)
		x3_output = self.film(x3, torch.full((x3.shape[0],), 2, dtype=torch.long, device=x3.device))

		concatenate = torch.concat([x1_output, x2_output, x3_output], dim=1)
		v_first = torch.zeros_like(concatenate)
		for block in self.concat:
			concatenate, v_first = block(concatenate, v_first)

		x1_output, x2_output, x3_output = torch.split(concatenate, [x1_output.shape[1], x2_output.shape[1], x3_output.shape[1]], dim=1)
		x1_reconstruct = self.compress1(self.linear_reverse1(x1_output), reverse=True)
		x1_output = self.o1(self.lno(x1_reconstruct))
		x2_reconstruct = self.compress2(self.linear_reverse2(x2_output), reverse=True)
		x2_output = self.o2(self.lno(x2_reconstruct))
		x3_output = self.o3(self.lno(x3_output))
		return x1_output, x2_output, x3_output, x1_reconstruct, x2_reconstruct