import os, torch, math, torch.distributed as dist, loralib as lora
from typing import cast, Tuple
from torch import nn
from torch.utils.cpp_extension import load
from rosa_soft import rosa_anchor_ops, RosaRuntime
MyModule = torch.jit.ScriptModule
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
# DDP setup
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
is_distributed = world_size > 1
if is_distributed:
	torch.cuda.set_device(local_rank)
	device = torch.device('cuda', local_rank)
	dist.init_process_group(backend='nccl', init_method='env://')
	rank = int(os.environ.get('RANK', '0'))
else:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	rank = 0

HEAD_SIZE = 64
CHUNK_LEN = 1
flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f'-D_CHUNK_LEN_={CHUNK_LEN}', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization']
load(name='rwkv_clamp', sources=['cuda/rwkv7_clampw.cu', 'cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=flags)


class RWKVTokenizer:
	table: list[list[list[bytes]]]
	good: list[set[int]]
	max_len: list[int]

	def __init__(self, file_name):
		self.idx2token = dict()
		sorted_token = list()  # must be already sorted
		lines = open(file_name, 'r', encoding='utf-8').readlines()
		for l in lines:
			idx = int(l[:l.index(' ')])
			x = eval(l[l.index(' '):l.rindex(' ')])
			x = x.encode('utf-8') if isinstance(x, str) else x
			assert isinstance(x, bytes)
			assert len(x) == int(l[l.rindex(' '):])
			sorted_token += [x]
			self.idx2token[idx] = x

		self.token2idx = dict()
		for k, v in self.idx2token.items():
			self.token2idx[v] = int(k)

		# precompute some tables for fast matching
		self.table = [[list() for _ in range(256)] for _ in range(256)]
		self.good = [set() for _ in range(256)]
		self.max_len = [0 for _ in range(256)]

		for i in reversed(range(len(sorted_token))):  # reverse order - match longer tokens first
			s = sorted_token[i]
			if len(s) >= 2:
				s0 = int(s[0])
				s1 = int(s[1])
				self.table[s0][s1] += [s]
				self.max_len[s0] = max(self.max_len[s0], len(s))
				self.good[s0].add(s1)

	def encode(self, src: str):
		src = src.encode('utf-8')
		src_len: int = len(src)
		tokens: list[int] = list()
		i: int = 0
		while i < src_len:
			s: bytes = src[i: i + 1]
			if i < src_len - 1:
				s1: int = int(src[i + 1])
				s0: int = int(src[i])
				if s1 in self.good[s0]:
					sss: bytes = src[i: i + self.max_len[s0]]
					try:
						s = next(filter(sss.startswith, self.table[s0][s1]))
					except Exception as exception:
						print(exception)
						pass
			tokens.append(self.token2idx[s])
			i += len(s)
		return tokens

	def decode(self, tokens):
		return bytes().join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8')

	def print_tokens(self, tokens):
		for i in tokens:
			s = self.idx2token[i]
			try:
				s = s.decode('utf-8')
			except Exception as exception:
				print(exception)
				pass
			print(f'{repr(s)}{i}', end=' ')
		# print(repr(s), i)
		print()


tokenizer = RWKVTokenizer('rwkv_vocab_v20230424')


class RWKVClampCUDAOP(torch.autograd.Function):
	@staticmethod
	def forward(ctx, r, w, k, v, a, b):
		batch, sequence, head, channel = r.shape
		assert sequence % CHUNK_LEN == 0  # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
		assert all(i.dtype == torch.bfloat16 for i in [r, w, k, v, a, b])
		assert all(i.is_contiguous() for i in [r, w, k, v, a, b])
		y = torch.empty_like(v)
		s = torch.empty(batch, head, sequence // CHUNK_LEN, channel, channel, dtype=torch.float32, device=w.device)
		sa = torch.empty(batch, sequence, head, channel, dtype=torch.float32, device=w.device)
		torch.ops.rwkv_clamp.forward(r, w, k, v, a, b, y, s, sa)
		ctx.save_for_backward(r, w, k, v, a, b, s, sa)
		return y

	@staticmethod
	def backward(ctx, *grad_outputs):
		dy = grad_outputs[0]
		assert all(i.dtype == torch.bfloat16 for i in [dy])
		assert all(i.is_contiguous() for i in [dy])
		r, w, k, v, a, b, s, sa = ctx.saved_tensors
		dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
		torch.ops.rwkv_clamp.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
		return dr, dw, dk, dv, da, db


def rwkv_clamp_cuda(r, w, k, v, a, b):
	batch, sequence, head_dim = r.shape
	r, w, k, v, a, b = [i.view(batch, sequence, head_dim // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]
	return RWKVClampCUDAOP.apply(r.to(torch.bfloat16), w.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), a.to(torch.bfloat16), b.to(torch.bfloat16)).view(batch, sequence, head_dim)


load(name='rwkv7_cmix_bf16_v5', sources=['cuda/rwkv7_cmix_bf16_v5.cpp', 'cuda/rwkv7_cmix_bf16_v5.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False, verbose=True)


class ChannelMixingLayer(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, x_k, key_weight, value_weight):
		x, x_k, key_weight, value_weight = [t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t for t in (x, x_k, key_weight, value_weight)]
		out, mixed, act = torch.ops.rwkv_channel_mix_bf16_v5.forward(x.contiguous(), x_k.contiguous(), key_weight.contiguous(), value_weight.contiguous())
		ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, act)
		return out

	@staticmethod
	def backward(ctx, *grad_outputs):
		x, x_k, key_weight, value_weight, mixed, act = ctx.saved_tensors
		grad_x, grad_x_k, grad_key_weight, grad_value_weight = torch.ops.rwkv_channel_mix_bf16_v5.backward(grad_outputs[0].contiguous(), x, x_k, key_weight, value_weight, mixed, act)
		return grad_x, grad_x_k, grad_key_weight, grad_value_weight


load(name='rwkv_time_mix_mix6_bf16_v5', sources=['cuda/rwkv7_tmix_mix6_bf16_v5.cpp', 'cuda/rwkv7_tmix_mix6_bf16_v5.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False, verbose=True)


def _setup_context(ctx, inputs, output):
	_ = output
	ctx.save_for_backward(*inputs)


def _backward(ctx, grad_outputs):
	return tuple(torch.ops.rwkv_time_mix_mix6_bf16_v5.backward(grad_outputs[0].contiguous(), grad_outputs[1].contiguous(), grad_outputs[2].contiguous(), grad_outputs[3].contiguous(), grad_outputs[4].contiguous(), grad_outputs[5].contiguous(), *ctx.saved_tensors))


try:
	torch.library.register_autograd('rwkv_time_mix_mix6_bf16_v5::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	# already registered (e.g. C++ impl loaded first) — skip
	pass


@torch.jit.script
def time_mix_mix6_bf16_v5(x: torch.Tensor, x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	outs = torch.ops.rwkv_time_mix_mix6_bf16_v5.forward(x.to(torch.bfloat16).contiguous(), x_r.to(torch.bfloat16).contiguous(), x_w.to(torch.bfloat16).contiguous(), x_k.to(torch.bfloat16).contiguous(), x_v.to(torch.bfloat16).contiguous(), x_a.to(torch.bfloat16).contiguous(),
	                                                    x_g.to(torch.bfloat16).contiguous())
	return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]


load(name='rwkv_time_mix_kk_pre_bf16_v5', sources=['cuda/rwkv7_tmix_kk_pre_bf16_v5.cpp', 'cuda/rwkv7_tmix_kk_pre_bf16_v5.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False, verbose=True)
load(name='invertible_downsample', sources=['cuda/invertible_downsample.cpp'], verbose=True, is_python_module=False)
load(name='invertible_channel_mix', sources=['cuda/invertible_channel_mix.cpp'], verbose=True, is_python_module=False)

# load(name='real_nvp', sources=['cuda/real_nvp_block.cpp'], verbose=True, is_python_module=False)
torch.ops.real_nvp = None


def _setup_context(ctx, inputs, output):
	_ = output
	x, stride = inputs
	ctx.stride = int(stride)


def _backward(ctx, *grad_outputs):
	gy = grad_outputs[0].contiguous()
	gx = torch.ops.invertible_downsample.backward(gy, ctx.stride)
	return gx, None


try:
	torch.library.register_autograd('invertible_downsample::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	pass


def _setup_context(ctx, inputs, output):
	_ = output
	x, w_matrix = inputs
	ctx.save_for_backward(x, w_matrix)


def _backward(ctx, *grads):
	g = grads[0].contiguous()
	x, w_matrix = ctx.saved_tensors
	batch, sequence, channel = x.shape
	x2 = x.view(batch * sequence, channel)
	g2 = g.view(batch * sequence, channel)
	gx = torch.matmul(g2, w_matrix.t()).view(batch, sequence, channel)
	gw = torch.matmul(x2.t(), g2)
	return gx, gw


try:
	torch.library.register_autograd('invertible_channel_mix::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	pass


def _setup_context(ctx, inputs, output):
	k, k_k, a, k_a, _head_size = inputs
	inv_d = output[3]
	ctx.save_for_backward(k, k_k, a, k_a, inv_d)


def _backward(ctx, grads):
	k, k_k, a, k_a, inv_d = ctx.saved_tensors
	grad_new_k = grads[0].contiguous()
	grad_neg_kk = grads[1].contiguous()
	grad_kka = grads[2].contiguous()
	return tuple(torch.ops.rwkv_time_mix_kk_pre_bf16_v5.backward(grad_new_k, grad_neg_kk, grad_kka, k, k_k, a, k_a, inv_d, HEAD_SIZE)) + (None,)


try:
	torch.library.register_autograd('rwkv_time_mix_kk_pre_bf16_v5::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	# already registered (e.g. C++ impl loaded first) — skip
	pass


def time_mix_kk_pre_bf16_v5(k: torch.Tensor, k_k: torch.Tensor, a: torch.Tensor, k_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	outs = torch.ops.rwkv_time_mix_kk_pre_bf16_v5.forward(k.to(torch.bfloat16).contiguous(), k_k.to(torch.bfloat16).contiguous(), a.to(torch.bfloat16).contiguous(), k_a.to(torch.bfloat16).contiguous(), HEAD_SIZE)
	return outs[0], outs[1], outs[2]


load(name='rwkv_time_mix_lnx_rkvres_xg_bf16_v1', sources=['cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp', 'cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False,
     verbose=True)


def _setup_context(ctx, inputs, output):
	x, r, k, v, r_k, weight, bias, g = inputs
	mean = output[1]
	std = output[2]
	ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, mean, std)


def _backward(ctx, grads):
	x, r, k, v, r_k, weight, bias, g, mean, std = ctx.saved_tensors
	grad_xg = grads[0].contiguous()
	return tuple(torch.ops.rwkv_time_mix_lnx_rkvres_xg_bf16_v1.backward(grad_xg, x, r, k, v, r_k, weight, bias, g, mean, std))


try:
	torch.library.register_autograd('rwkv_time_mix_lnx_rkvres_xg_bf16_v1::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	# already registered (e.g. C++ impl loaded first) — skip
	pass


@torch.jit.script
def time_mix_lnx_rkvres_xg_bf16_v1(x: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, r_k: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
	outs = torch.ops.rwkv_time_mix_lnx_rkvres_xg_bf16_v1.forward(x.to(torch.bfloat16).contiguous(), r.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous(), r_k.to(torch.bfloat16).contiguous(), weight.to(torch.bfloat16).contiguous(),
	                                                             bias.to(torch.bfloat16).contiguous(), g.to(torch.bfloat16).contiguous())
	return outs[0]


load(name='rwkv_time_mix_a_gate_bf16', sources=['cuda/rwkv7_tmix_a_gate_bf16.cpp', 'cuda/rwkv7_tmix_a_gate_bf16.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False, verbose=True)


def _setup_context(ctx, inputs, output):
	_ = output
	a0, a12 = inputs
	ctx.save_for_backward(a0, a12)


def _backward(ctx, grad_out):
	a0, a12 = ctx.saved_tensors
	return tuple(torch.ops.rwkv_time_mix_a_gate_bf16.backward(grad_out.contiguous(), a0, a12))


try:
	torch.library.register_autograd('rwkv_time_mix_a_gate_bf16::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	# already registered (e.g. C++ impl loaded first) — skip
	pass


@torch.jit.script
def time_mix_a_gate_bf16(a0: torch.Tensor, a12: torch.Tensor) -> torch.Tensor:
	return torch.ops.rwkv_time_mix_a_gate_bf16.forward(a0.to(torch.bfloat16).contiguous(), a12.to(torch.bfloat16).contiguous())


load(name='rwkv_time_mix_vres_gate_bf16_v3', sources=['cuda/rwkv7_tmix_vres_gate_bf16_v3.cpp', 'cuda/rwkv7_tmix_vres_gate_bf16_v3.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], is_python_module=False,
     verbose=True)


def _setup_context(ctx, inputs, output):
	_ = output
	v, v_first, v0, v12 = inputs
	ctx.save_for_backward(v, v_first, v0, v12)


def _backward(ctx, grad_out):
	v, v_first, v0, v12 = ctx.saved_tensors
	return tuple(torch.ops.rwkv_time_mix_vres_gate_bf16_v3.backward(grad_out.contiguous(), v, v_first, v0, v12))


try:
	torch.library.register_autograd('rwkv_time_mix_vres_gate_bf16_v3::forward', _backward, setup_context=_setup_context)
except RuntimeError:
	# already registered (e.g. C++ impl loaded first) — skip
	pass


@torch.jit.script
def time_mix_vres_gate_bf16_v3(v: torch.Tensor, v_first: torch.Tensor, v0: torch.Tensor, v12: torch.Tensor) -> torch.Tensor:
	return torch.ops.rwkv_time_mix_vres_gate_bf16_v3.forward(v.to(torch.bfloat16).contiguous(), v_first.to(torch.bfloat16).contiguous(), v0.to(torch.bfloat16).contiguous(), v12.to(torch.bfloat16).contiguous())


L2WRAP_CE_CUDA_V2 = load(name='rwkv_l2wrap_ce_bf16_v2', sources=['cuda/rwkv7_l2wrap_ce_bf16_v2.cpp', 'cuda/rwkv7_l2wrap_ce_bf16_v2.cu'], extra_cflags=['-O3'], extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization'], verbose=True)


class L2WrapCrossEntropyCUDA(torch.autograd.Function):
	@staticmethod
	def forward(ctx, logits, targets, ignore_index: int = -100):
		logits = logits.contiguous()
		targets = targets.contiguous()
		loss, lse, max_vals, argmax = L2WRAP_CE_CUDA_V2.forward(logits, targets, int(ignore_index))
		ctx.save_for_backward(logits, targets.view(-1), lse, max_vals, argmax)
		ctx.ignore_index = int(ignore_index)
		return loss

	@staticmethod
	def backward(ctx, *grad_output):
		logits, targets, lse, max_vals, argmax = ctx.saved_tensors
		grad_logits = L2WRAP_CE_CUDA_V2.backward(grad_output[0].contiguous().float(), logits, targets, lse, max_vals, argmax, ctx.ignore_index)
		# forward inputs: logits, targets, ignore_index -> so return 3 gradients (None for non-tensor args)
		return grad_logits, None, None


def l2wrap_cross_entropy(logits, targets, ignore_index: int = -100):
	return L2WrapCrossEntropyCUDA.apply(logits, targets, ignore_index)


########################################################################################################
class InvertibleDownsample1D(nn.Module):
	def __init__(self, stride):
		super().__init__()
		self.stride = stride

	def forward(self, x, reverse=False):
		# unify types/layout for native kernel
		batch, sequence, channel = x.shape
		if not reverse:
			assert sequence % self.stride == 0, f'sequence length {sequence} is not divisible by stride {self.stride}'
			if torch.ops.invertible_downsample is not None and x.is_cuda:
				return torch.ops.invertible_downsample.forward(x.contiguous(), self.stride)
			else:
				return x.reshape(batch, sequence // self.stride, channel * self.stride).contiguous()
		else:
			assert channel % self.stride == 0, f'channel {channel} not divisible by stride {self.stride}'
			if torch.ops.invertible_downsample is not None and x.is_cuda:
				return torch.ops.invertible_downsample.backward(x.contiguous(), self.stride)
			else:
				return x.reshape(batch, sequence * self.stride, channel // self.stride).contiguous()


class InvertibleChannelMix(nn.Module):
	def __init__(self, C):
		super().__init__()
		# keep same orthogonal init as Python reference
		q_matrix, _ = torch.linalg.qr(torch.randn(C, C, device=device))
		self.register_buffer('q_matrix', q_matrix)
		self.register_buffer('q_matrix_transpose', q_matrix.T)

	def forward(self, x, reverse=False):
		# x: [B, T, C]
		# use native kernel if available (expects x, W)
		w_matrix = self.q_matrix_transpose if reverse else self.q_matrix
		if torch.ops.invertible_channel_mix is not None and x.is_cuda:
			return torch.ops.invertible_channel_mix.forward(x.contiguous(), w_matrix.contiguous())
		else:
			# fallback: matmul (works on CPU/GPU)
			return x @ w_matrix


class RealNVPBlock(nn.Module):
	def __init__(self, C):
		super().__init__()
		self.split = C // 2
		self.rest = C - self.split
		self.s = nn.Sequential(nn.Linear(self.split, self.rest), nn.ReLU(), nn.Linear(self.rest, self.rest))
		self.t = nn.Sequential(nn.Linear(self.split, self.rest), nn.ReLU(), nn.Linear(self.rest, self.rest))

	def forward(self, x, reverse=False):
		# x: [B, T, C]
		x1 = x[:, :, :self.split]
		# compute scale/shift in full precision if needed
		s = torch.tanh(self.s(x1))
		t = self.t(x1)
		if torch.ops.real_nvp is not None and x.is_cuda:
			# C++ forward expects (x, s, t, reverse_flag) and returns (y, log_determinant)
			out = torch.ops.real_nvp.forward(x.contiguous(), s.contiguous(), t.contiguous(), bool(reverse))
			# some bindings return tuple/list, adapt:
			if isinstance(out, (tuple, list)) and len(out) >= 1:
				y = out[0]
			else:
				y = out
			return y
		else:
			x2 = x[:, :, self.split:]
			if not reverse:
				y2 = x2 * torch.exp(s) + t
				return torch.cat([x1, y2], dim=-1)
			else:
				x2_rec = (x2 - t) * torch.exp(-s)
				return torch.cat([x1, x2_rec], dim=-1)


class InvertibleCompress(nn.Module):
	def __init__(self, C, stride=4, n_flows=2):
		super().__init__()
		self.down = InvertibleDownsample1D(stride)
		self.flows = nn.ModuleList(list())
		for _ in range(n_flows):
			self.flows.append(nn.ModuleList([InvertibleChannelMix(C * stride), RealNVPBlock(C * stride)]))

	def forward(self, x, reverse=False):
		if not reverse:
			# compress: 下采样 + 多层可逆流
			x = self.down(x)
			for mix, flow in self.flows:
				x = mix(x, reverse=False)
				x = flow(x, reverse=False)
			return x
		else:
			# decompress: 逆序还原
			for mix, flow in reversed(self.flows):
				x = flow(x, reverse=True)
				x = mix(x, reverse=True)
			x = self.down(x, reverse=True)
			return x


class FiLM(nn.Module):
	def __init__(self, C: int, n_modalities: int, embed_dim: int = 0, hidden_dim: int = 0):
		"""
		C         : 主干通道数
		n_modalities : 模态类别数
		embed_dim : 模态 embedding 维度，默认 max(32, C//4)
		hidden_dim: MLP 隐层维度，默认 max(embed_dim, C//2)
		"""
		super().__init__()
		if not embed_dim:
			embed_dim = max(32, C // 4)
		if not hidden_dim:
			hidden_dim = max(embed_dim, C // 2)

		self.embed = nn.Embedding(n_modalities, embed_dim)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 2 * C)  # produce gamma and beta
		)

		# 初始化使 FiLM 初始近似恒等映射
		nn.init.zeros_(self.mlp[-1].bias)
		nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-3)

	def forward(self, x: torch.Tensor, modal_id: torch.LongTensor):
		"""
		x        : [B, T, C]
		modal_id : [B]  每个样本的模态 id
		返回    : out [B, T, C]
		"""
		# embed modal id -> [B, embed_dim]
		cond = self.embed(modal_id)  # [B, D]
		# map to gamma/beta -> [B, 2C]
		gamma_beta = self.mlp(cond)  # [B, 2C]
		gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, C]
		# broadcast to time dim -> [B, 1, C]
		gamma = gamma.unsqueeze(1)
		beta = beta.unsqueeze(1)
		# apply FiLM: out = x * (1 + gamma) + beta
		return x * (1 + gamma) + beta


class RosaLayer(nn.Module):
	def __init__(self, C, window=32, fine_tune=False):
		super().__init__()
		self.C = C
		self.head = max(1, C // 8)
		self.dim = 8
		self.ln_a = nn.LayerNorm(C)
		self.ln_b = nn.LayerNorm(C)
		self.Wq = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
		self.Wk = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
		self.Wv = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
		self.runtime = None
		self.telemetry = {'top prob': 0, 'entropy norm': 0, 'candidate mass': 0, 'count': 0}
		self.window = window

	def forward(self, xa, xb):
		# x: [B, T, C]
		batch, sequence, channel = xa.shape
		xa = self.ln_a(xa)
		xb = self.ln_b(xb)
		q = self.Wq(xa).view(batch, sequence, self.head, self.dim)
		k = self.Wk(xb).view(batch, sequence, self.head, self.dim)
		v = self.Wv(xb).view(batch, sequence, self.head, self.dim)
		# if self.training:
		# 	# 调用 CUDA kernel：期望 [B, T, H, D]
		# 	y = cast(torch.Tensor, rosa_anchor_ops(q, k, v, window_size=self.window))  # y: [B, T, H, D]
		# 	return y.view(batch, sequence, channel)
		# if not self.runtime:
		# 	self.runtime = RosaRuntime(self.head, self.head)
		# out, _ = self.runtime.update(q, k, v, return_packed=False)
		# return out.view(batch, sequence, channel)
		y, telemetry = rosa_anchor_ops(q, k, v, window_size=self.window, return_telemetry=True)  # y: [B, T, H, D]
		if not self.training:
			self.telemetry['top prob'] += float(telemetry.top_prob.detach().cpu())
			self.telemetry['entropy norm'] += float(telemetry.entropy_norm.detach().cpu())
			self.telemetry['candidate mass'] += float(telemetry.candidate_mass.detach().cpu())
			self.telemetry['count'] += 1
		return y.view(batch, sequence, channel)


class RWKVTimeMixX070(MyModule):
	def __init__(self, head_size, n_embd, layer_id, n_layer, fine_tune=False):
		super().__init__()
		self.layer_id = layer_id
		self.head_size = head_size
		self.n_head = n_embd // self.head_size
		assert n_embd % self.n_head == 0
		assert n_layer > 1
		H = self.n_head
		N = self.head_size
		C = n_embd
		with torch.no_grad():
			ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
			ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
			ddd = torch.ones(1, 1, C)
			for i in range(C):
				ddd[0, 0, i] = i / C
			self.x_r = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
			self.x_w = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
			self.x_k = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
			self.x_v = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
			self.x_a = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
			self.x_g = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

			def ortho_init(x: torch.Tensor, scale: float):
				with torch.no_grad():
					shape = x.shape
					if len(shape) == 2:
						gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
						nn.init.orthogonal_(x, gain=gain * scale)
					elif len(shape) == 3:
						gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
						for i in range(shape[0]):
							nn.init.orthogonal_(x[i], gain=gain * scale)
					else:
						assert False
					return x

			www = torch.zeros(C)
			zigzag = torch.zeros(C)
			linear = torch.zeros(C)
			for n in range(C):
				linear[n] = n / (C - 1) - 0.5
				zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
				zigzag[n] = zigzag[n] * abs(zigzag[n])
				www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

			D_DECAY_LORA = 16
			self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
			self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
			self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)
			D_AAA_LORA = 16
			self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
			self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
			self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)
			D_MV_LORA = 16
			self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
			self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
			self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)
			D_GATE_LORA = 16
			self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
			self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

			self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
			self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
			self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

			self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
			self.receptance = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
			self.key = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
			self.value = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
			self.output = lora.Linear(C, C, r=16, bias=False) if fine_tune else nn.Linear(C, C, bias=False)
			self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

			self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
			self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
			self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
			self.output.weight.data.zero_()

	def forward(self, x, v_first):
		# batch, sequence, channel = x.size()
		# head = self.n_head

		# xx = self.time_shift(x) - x
		# xr = x + xx * self.x_r
		# xw = x + xx * self.x_w
		# xk = x + xx * self.x_k
		# xv = x + xx * self.x_v
		# xa = x + xx * self.x_a
		# xg = x + xx * self.x_g
		xr, xw, xk, xv, xa, xg = time_mix_mix6_bf16_v5(x, self.x_r.view(-1), self.x_w.view(-1), self.x_k.view(-1), self.x_v.view(-1), self.x_a.view(-1), self.x_g.view(-1))

		r = self.receptance(xr)
		w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2  # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in rwkv clamp kernel
		k = self.key(xk)
		v = self.value(xv)
		if self.layer_id == 0:
			v_first = v  # store the v of the first layer
		else:
			# v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)  # add value residual
			v = time_mix_vres_gate_bf16_v3(v, v_first, self.v0, (xv @ self.v1) @ self.v2)  # faster
		# a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # variable 'a' is 'in-context learning rate'
		a = time_mix_a_gate_bf16(self.a0, (xa @ self.a1) @ self.a2)
		g = torch.sigmoid(xg @ self.g1) @ self.g2

		# kk = k * self.k_k
		# kk = F.normalize(kk.view(batch, sequence, head, -1), dim=-1, p=2.0).view(batch, sequence, channel)
		# k = k * (1 + (a - 1) * self.k_a)
		# x = rwkv_clamp_cuda(r, w, k, v, -kk, kk * a)
		k, neg_kk, kka = time_mix_kk_pre_bf16_v5(k, self.k_k.view(-1), a, self.k_a.view(-1))
		x = rwkv_clamp_cuda(r, w, k, v, neg_kk, kka)

		# x = self.ln_x(x.view(batch * sequence, channel)).view(batch, sequence, channel)
		# x = x + ((r.view(batch, sequence, head, -1) * k.view(batch, sequence, head, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(batch, sequence, head, -1)).view(batch, sequence, channel)
		# x = self.output(x * g)
		x = time_mix_lnx_rkvres_xg_bf16_v1(x, r, k, v, self.r_k, self.ln_x.weight, self.ln_x.bias, g)
		x = self.output(x)
		return x, v_first


class MultiHeadAttentionPooling(nn.Module):
	def __init__(self, dim, num_heads=4):
		super().__init__()
		self.num_heads = num_heads
		self.attention = nn.Linear(dim, num_heads)  # 每个 head 一个打分

	def forward(self, x):
		# x: [B, T, C]
		batch, sequence, channel = x.shape
		# [B, T, H]
		scores = self.attention(x)
		# 对时间维做 softmax
		weights = torch.softmax(scores, dim=1)  # [B, T, H]
		# 加权求和：对每个 head 单独 pool
		# x: [B, T, C] → [B, T, 1, C]
		x_expanded = x.unsqueeze(2)  # [B, T, 1, C]
		# weights: [B, T, H] → [B, T, H, 1]
		w_expanded = weights.unsqueeze(-1)  # [B, T, H, 1]
		# [B, T, H, C]
		weighted = x_expanded * w_expanded
		# 对时间维求和 → 每个 head 一个 pooled 向量
		# [B, H, C]
		pooled_heads = weighted.sum(dim=1)
		# 拼接所有 head
		# [B, H*C]
		pooled = pooled_heads.reshape(batch, self.num_heads * channel)
		return pooled


class FFN(nn.Module):  # fast CUDA version
	def __init__(self, n_embd, layer_id, n_layer, fine_tune=False):
		super().__init__()
		self.layer_id = layer_id
		with torch.no_grad():
			ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
			ddd = torch.ones(1, 1, n_embd)
			for i in range(n_embd):
				ddd[0, 0, i] = i / n_embd
			self.x_k = nn.Parameter(torch.ones_like(ddd) - torch.pow(ddd, ratio_1_to_almost0 ** 4))
		self.key = lora.Linear(n_embd, n_embd * 4, r=16, bias=False) if fine_tune else nn.Linear(n_embd, n_embd * 4, bias=False)
		self.value = lora.Linear(n_embd * 4, n_embd, r=16, bias=False) if fine_tune else nn.Linear(n_embd * 4, n_embd, bias=False)
		self.key.weight.data.uniform_(-0.5 / (n_embd ** 0.5), 0.5 / (n_embd ** 0.5))
		self.value.weight.data.zero_()

	def forward(self, x):
		return ChannelMixingLayer.apply(x, self.x_k.view(-1), self.key.weight, self.value.weight)


class RWKVBlock(nn.Module):
	def __init__(self, C, layer_id, n_layer, head_size=64, fine_tune=False):
		super().__init__()
		self.ln_a = nn.LayerNorm(C)
		self.ln_b = nn.LayerNorm(C)
		self.rosa = RosaLayer(C, fine_tune=fine_tune)
		self.time_mixing = RWKVTimeMixX070(head_size, C, layer_id, n_layer, fine_tune=fine_tune)
		self.ffn = FFN(C, layer_id, n_layer, fine_tune=fine_tune)

	def forward(self, x, v_first):
		# v_first 由上一层传入（初始为 torch.zeros_like(x)）
		xr = self.rosa(x, x)
		xx, v_first = self.time_mixing(self.ln_a(x), v_first)
		x = x + xx + xr
		return x + self.ffn(self.ln_b(x)), v_first


class RWKVBlockRosaFree(nn.Module):
	def __init__(self, C, layer_id, n_layer, head_size=64, fine_tune=False):
		super().__init__()
		self.ln_a = nn.LayerNorm(C)
		self.ln_b = nn.LayerNorm(C)
		self.time_mixing = RWKVTimeMixX070(head_size, C, layer_id, n_layer, fine_tune=fine_tune)
		self.ffn = FFN(C, layer_id, n_layer, fine_tune=fine_tune)

	def forward(self, x, v_first):
		# v_first 由上一层传入（初始为 torch.zeros_like(x)）
		xx, v_first = self.time_mixing(self.ln_a(x), v_first)
		x = x + xx
		return x + self.ffn(self.ln_b(x)), v_first