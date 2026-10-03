############################################################################
# Copyright 2025-2026 pyke.io              https://github.com/pykeio/THORN #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

from dataclasses import dataclass, field, fields, MISSING
from functools import partial, lru_cache
from itertools import combinations
from os import environ
from typing import cast, overload, Callable, Literal, Optional, TypedDict, Union, NotRequired, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.optimizer import _get_value

__version__ = '3.0.0'
__all__ = ['THORN', 'THORNOrthoGroup', 'THORNNonOrthoGroup', 'THORNGroup']

_has_triton = False
try:
	import triton # ty:ignore[unresolved-import, unused-ignore-comment]
	_has_triton = True
except ModuleNotFoundError:
	pass
except RuntimeError:
	pass

_use_triton = _has_triton and environ.get('THORN_DISABLE_TRITON') not in ['1', 'true', 'True']
if _use_triton:
	import triton                # ty:ignore[unresolved-import, unused-ignore-comment]
	import triton.language as tl # ty:ignore[unresolved-import, unused-ignore-comment]

	_autotune_conf = [
		triton.Config(
			{'BLOCK_SIZE_M': blk_m, 'BLOCK_SIZE_K': blk_k, 'GROUP_SIZE_M': grp_sz},
			num_stages=n_stages, num_warps=n_warps,
		)
		for blk_m    in [32, 64, 128]
		for blk_k    in [32, 64]
		for grp_sz   in [8]
		for n_stages in [3, 4, 5]
		for n_warps  in [4, 8]
	]

	@triton.autotune(configs=_autotune_conf, key=['M', 'K'])
	@triton.jit
	def _sym_addmm_kernel(
		A, B, C, O,
		alpha, beta,
		M, K,
		stride_ab, stride_am, stride_ak,
		stride_bb, stride_bk, stride_bn,
		stride_cb, stride_cm, stride_cn,
		stride_ob, stride_om, stride_on,
		BLOCK_SIZE_M: tl.constexpr,
		BLOCK_SIZE_K: tl.constexpr,
		GROUP_SIZE_M: tl.constexpr,
	):
		# adapted from https://github.com/nil0x9/flash-muon
		pid = tl.program_id(axis=0)
		batch_id = tl.program_id(axis=1)
		num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
		num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
		num_pid_in_group = GROUP_SIZE_M * num_pid_n
		group_id = pid // num_pid_in_group
		first_pid_m = group_id * GROUP_SIZE_M
		group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
		pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
		pid_n = (pid % num_pid_in_group) // group_size_m

		A += batch_id * stride_ab
		B += batch_id * stride_bb
		C += batch_id * stride_cb
		O += batch_id * stride_ob

		if pid_m > pid_n:
			return

		offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
		offs_n = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
		offs_k = tl.arange(0, BLOCK_SIZE_K)

		a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
		b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

		acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

		for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
			mask_k = offs_k < K - k * BLOCK_SIZE_K
			a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
			acc = tl.dot(a, b, acc)
			a_ptrs += BLOCK_SIZE_K * stride_ak
			b_ptrs += BLOCK_SIZE_K * stride_bk

		offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
		offs_on = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
		block_mask = (offs_om[:, None] < M) & (offs_on[None, :] < M)

		c_ptrs = C + stride_cm * offs_om[:, None] + stride_cn * offs_on[None, :]
		c = tl.load(c_ptrs, mask=block_mask, other=0.0).to(tl.float32)

		o_ptrs = O + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
		o = (alpha * acc + beta * c).to(A.dtype.element_ty)
		tl.store(o_ptrs, o, mask=block_mask)

		if pid_m < pid_n:
			ot_ptrs = O + stride_om * offs_on[:, None] + stride_on * offs_om[None, :]
			ot_mask = (offs_on[:, None] < M) & (offs_om[None, :] < M)
			tl.store(ot_ptrs, tl.permute(o, (1, 0)), mask=ot_mask)
elif not _has_triton:
	import warnings
	warnings.warn('Triton is not installed, so THORN will fall back to non-symmetric matmul. Installing Triton can speed up optimizer.step() by up to 50%; the effect would be most notable on larger models.')

@torch.no_grad()
def _sym_addmm(
	A: torch.Tensor,
	B: torch.Tensor,
	C: torch.Tensor,
	alpha: float = 1.0,
	beta: float = 1.0,
	out: torch.Tensor | None = None,
) -> torch.Tensor:
	"""
	out = beta * C + alpha * (A @ B), assuming A @ B and C are symmetrical

	A = (...N, M, K)
	B = (...N, K, M)
	C = (...N, M, M) or (M, M)
	out = (...N, M, M)
	"""
	assert A.device == B.device == C.device
	assert A.dtype == B.dtype == C.dtype
	assert A.ndim == B.ndim and A.ndim >= 2
	assert C.ndim >= 2

	*N, M, K = A.shape
	assert B.shape == (*N, K, M), f"B must be ({*N, K, M}), got {tuple(B.shape)}"
	assert C.shape[-2:] == (M, M), f"C must be (..., {M}, {M}), got {tuple(C.shape)}"

	assert C.shape == A.shape or C.ndim == 2

	NN = 1
	for d in N:
		NN *= d

	if out is not None:
		assert out.shape == (*N, M, M), f"out must be ({*N, M, M}), got {tuple(out.shape)}"
		assert out.dtype == A.dtype
		assert out.device == A.device
		assert out.is_contiguous()

	A = A.contiguous().reshape(NN, M, K)
	B = B.contiguous().reshape(NN, K, M)

	if C.ndim == 2:
		C = C.contiguous()
		stride_cb = 0
	else:
		C = C.contiguous().reshape(NN, M, M)
		stride_cb = C.stride(0)

	if out is not None:
		O = out.reshape(NN, M, M)
	elif _use_triton:
		O = torch.empty((NN, M, M), dtype=A.dtype, device=A.device)

	if not _use_triton or not (A.is_cuda and B.is_cuda and C.is_cuda):
		# note torch.addmm doesn't support out == C like our kernel does, so we can't pass out here
		O = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)
	else:
		grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) ** 2, NN)
		with torch.cuda.device(A.device.index):
			_sym_addmm_kernel[grid](
				A, B, C, O,
				alpha, beta,
				M, K,
				A.stride(0), A.stride(1), A.stride(2),
				B.stride(0), B.stride(1), B.stride(2),
				stride_cb, C.stride(-2), C.stride(-1),
				O.stride(0), O.stride(1), O.stride(2),
			)

	return O.reshape(*N, M, M)

try:
	if environ.get('THORN_COMPILE') not in ['1', 'true', 'True']:
		raise Exception()

	@torch.compile(dynamic=False, fullgraph=True)
	@torch.no_grad()
	def test(x: torch.Tensor):
		return x + 1.

	x = test(torch.tensor([1.0, 2.0]).cuda())
	assert torch.allclose(x.cpu(), torch.tensor([2.0, 3.0]))

	del x
	del test
	_optional_compile = torch.compile
except:
	def _optional_compile(model: None = None, *, fullgraph: bool = False, dynamic: bool = False):
		def fn(c):
			return c
		return fn

@lru_cache(maxsize=None)
def _optimal_composition(l: float, num_iters: int, safety_factor_eps: float = 0.0, cushion: float = 0.0):
	"""
	Calculate coefficients for `_polar_decomp`, from https://arxiv.org/pdf/2505.16932
	"""
	assert 0 <= l <= 1

	def optimal_quintic(l: float, u: float):
		assert 0 <= l <= u

		eps_d = 1e-15
		if l / u >= 1. - eps_d:
			return (15. / 8.) / u, \
				(-10. / 8.) / (u ** 3.), \
				(3. / 8.) / (u ** 5.)

		q = (3. * l + u) / 4.
		r = (l + 3. * u) / 4.
		E, old_E = float('inf'), None
		a, b, c = 1., 1., 1.
		while not old_E or abs(old_E - E) > eps_d:
			old_E = E
			LHS = torch.tensor([
				[l, l ** 3., l ** 5.,  1.],
				[q, q ** 3., q ** 5., -1.],
				[r, r ** 3., r ** 5.,  1.],
				[u, u ** 3., u ** 5., -1.],
			], dtype=torch.float64) # *needs* to be computed in f64
			a, b, c, E = torch.linalg.solve(LHS, torch.ones(4, dtype=torch.float64))
			q, r = torch.sqrt((-3. * b + torch.tensor([-1., 1.], dtype=torch.float64) * (9. * b ** 2. - 20. * a * c) ** 0.5) / (10. * c))
		return float(a), float(b), float(c)

	u = 1.
	safety_factor = 1. + safety_factor_eps
	coefficients = []
	for iter in range(num_iters):
		a, b, c = optimal_quintic(max(l, cushion * u), u)
		if cushion * u > l:
			pl = a * l + b * l ** 3. + c * l ** 5.
			pu = a * u + b * u ** 3. + c * u ** 5.
			rescaler = 2. / (pl + pu)
			a *= rescaler
			b *= rescaler
			c *= rescaler

		if iter < num_iters - 1:
			a /= safety_factor
			b /= safety_factor ** 3.
			c /= safety_factor ** 5.

		coefficients.append((a, b, c))

		l = a * l + b * l ** 3. + c * l ** 5.
		u = 2. - l

	return coefficients

@lru_cache(maxsize=None)
def _gram_ns_restarts(*coeffs: tuple[float, float, float]):
	E = torch.logspace(0, -10, 10000, dtype=torch.float64) # *needs* to be computed in f64
	K = -4e-4

	def stability_cond(t: torch.Tensor):
		min, max = t.aminmax()
		return (max / min).item()

	for num_restarts in range(1, len(coeffs)):
		q_max = float('inf')
		restarts = []
		possible_positions = list(range(1, len(coeffs)))
		for i, combo in enumerate(combinations(possible_positions, num_restarts)):
			qv = []
			e = E.clone()
			for j, (a, b, c) in enumerate(coeffs):
				if j == 0 or j in combo:
					if j != 0:
						e *= q
					r = e ** 2 + K
					q = torch.ones(len(e), dtype=torch.float64)
				z = a + r * (b + r * c)
				q *= z
				r *= z ** 2
				qv.append(q)
			q = max([stability_cond(q) for q in qv])
			if q < q_max:
				q_max = q
				restarts = list(combo)
		if q_max < 1e8:
			return restarts
	raise ValueError('No stable restarts with these parameters')

if TYPE_CHECKING:
	class _Backend:
		class Stream(torch.Stream):
			def __init__(self): ...
		class Event(torch.Event):
			def __init__(self): ...

		class stream:
			def __init__(self, stream: torch.Stream | None): ...
			def __enter__(self): ...
			def __exit__(self, type, value, traceback): ...

		@staticmethod
		def current_stream() -> torch.Stream: ...

_TargetNorm = Union[Literal['init', 'min', 'max'], Callable[[torch.Tensor], Union[torch.Tensor, float]]]

@dataclass
class _THORNParameterGroup:
	orthogonalize: bool
	params: list[Parameter]
	lr: float
	none_grad: bool = field(default=True)
	weight_decay: float = field(default=0.1)
	betas: tuple[float, float] = field(default_factory=lambda: (0.95, 0.95))
	scaling_mode: Optional[Literal['md', 'moonlight', 'jordan']] = field(default=None)
	target_rms: float = field(default=0.2)
	target_norm: Optional[_TargetNorm] = field(default=None)
	iters: int = field(default=5)
	rectify: bool = field(default=True)
	lower_bound: float = field(default=1e-3)
	safety_factor: float = field(default=0.03)
	cushion: float = field(default=0.02)
	momentum_align: bool = field(default=False)
	coeffs: list[tuple[float, float, float]] = field(init=False)
	restarts: list[int] = field(init=False)
	decouple_md: bool = field(default=False)
	gain_lr: float = field(default=1e-3)
	split_gain: bool = field(default=True)
	force_per_neuron_norm: bool = field(default=False)

	@classmethod
	def from_kwargs(cls, **kwargs: dict) -> '_THORNParameterGroup':
		return cls(**{k: kwargs[k] for k in kwargs if k in cls.__dataclass_fields__}) # type: ignore

	def __post_init__(self):
		if self.lr < 0.0:
			raise ValueError(f'Invalid learning rate `{self.lr}`; should be >= 0')
		if not (0.0 <= self.betas[0] < 1.0):
			raise ValueError(f'Invalid beta1 `{self.betas[0]}`; should be in [0, 1)')
		if not (0.0 <= self.betas[1] < 1.0):
			raise ValueError(f'Invalid beta2 `{self.betas[1]}`; should be in [0, 1)')

		self.coeffs = _optimal_composition(
			l=self.lower_bound,
			num_iters=self.iters,
			safety_factor_eps=self.safety_factor,
			cushion=self.cushion
		)
		self.restarts = _gram_ns_restarts(*self.coeffs)

	@property
	def scaling_mode_(self) -> Literal['md', 'moonlight', 'jordan']:
		if self.scaling_mode is None:
			return 'md' if self.decouple_md else 'moonlight'
		return self.scaling_mode

	def target_norm_(self, p: torch.Tensor) -> torch.Tensor | float:
		target_norm = self.target_norm
		if target_norm is None:
			target_norm = 'max' if self.orthogonalize else 'init'
		if callable(target_norm):
			target_norm = target_norm(p)
		else:
			match target_norm:
				case 'init': target_norm = p.norm(dim=(-2, -1) if p.ndim > 1 else -1, keepdim=True)
				case 'min': target_norm = min(p.size(-2), p.size(-1)) ** 0.5
				case 'max': target_norm = max(p.size(-2), p.size(-1)) ** 0.5
		if isinstance(target_norm, (float, int)):
			target_norm = torch.full((), target_norm, dtype=p.dtype, device=p.device)
		else:
			target_norm = target_norm.to(dtype=p.dtype, device=p.device)
		return target_norm

@_optional_compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def _polar_decomp(X: torch.Tensor, group: _THORNParameterGroup):
	dtype = X.dtype
	if should_transpose := X.size(-2) > X.size(-1):
		X = X.mT

	X = X.to(torch.float32) # need to do norm in float32
	X /= X.norm(dim=(-2, -1), keepdim=True) + 1e-6
	# since the magnitudes of these matrices are ~1, prefer float16 over bfloat16 for its greater precision in this range
	X = X.to(torch.float16)

	I = torch.eye(X.size(-2), device=X.device, dtype=X.dtype)
	R = _sym_addmm(X, X.mT, I, beta=0)

	coeffs = group.coeffs[:group.iters]
	for i, (a, b, c) in enumerate(coeffs):
		# Gram Newton-Schulz: https://tridao.me/blog/2026/gram-newton-schulz/
		if i in group.restarts and i != 0:
			X = Q @ X
			R = _sym_addmm(X, X.mT, I, beta=0)

		Z = _sym_addmm(R, R, R, alpha=c, beta=b)
		Q = _sym_addmm(Q, Z, Q, beta=a) if i != 0 and i not in group.restarts else Z + a * I
		if i < len(coeffs) - 1 and i + 1 not in group.restarts:
			RZ = _sym_addmm(R, Z, R, beta=a)
			R = _sym_addmm(Z, RZ, RZ, beta=a)

	X = X.mT @ Q if should_transpose else Q @ X
	return X.to(dtype=dtype)

def _needs_neuron_norm(p: torch.Tensor, group: _THORNParameterGroup) -> bool:
	# Only apply per-neuron normalization to tall matrices, per https://arxiv.org/pdf/2606.27715
	# I question how that would fare on non-transformers hence the `force_per_neuron_norm` flag
	return group.betas[1] > 0 and (p.size(-2) > p.size(-1) or group.force_per_neuron_norm)

@torch.no_grad()
def _per_neuron_norm(u: torch.Tensor, target_norm: torch.Tensor | None, m2: torch.Tensor, group: _THORNParameterGroup):
	# Per-neuron normalization, from https://arxiv.org/abs/2510.05491
	if target_norm is None:
		target_norm = u.norm(dim=(-2, -1), keepdim=True)
	eps = torch.finfo(m2.dtype).eps
	v_mean = u.square().mean(dim=-1, keepdim=True)
	m2.lerp_(v_mean.to(m2.dtype), 1. - group.betas[1])
	u.mul_(m2.clamp_min(eps).rsqrt_())
	v_norm_new = u.norm(dim=(-2, -1), keepdim=True)
	u.mul_(target_norm.div(v_norm_new.clamp_min(eps)))
	return u

@torch.no_grad()
def _weight_decay(
	p: torch.Tensor,
	update: torch.Tensor,
	weight_decay: float
):
	# "Cautious" weight decay; only apply weight decay to elements in the same direction as the update
	# https://arxiv.org/abs/2510.12402
	if weight_decay > 0.0:
		mask = ((update * p) >= 0).to(dtype=p.dtype)
		update.addcmul_(p, mask.mul_(weight_decay))
	return update

def _lr_scale_ortho(p: torch.Tensor, group: _THORNParameterGroup):
	match group.scaling_mode_:
		case 'md':
			# https://haeggee.github.io/posts/magnitude-direction-decoupling#decoupling-magnitude-and-direction-the-details
			return (max(*p.shape[-2:]) / min(*p.shape[-2:])) ** 0.5
		case 'moonlight':
			# per formula 4 of https://arxiv.org/pdf/2502.16982
			assert group.target_rms > 0
			return group.target_rms * (max(1, *p.shape[-2:]) ** 0.5)
		case 'jordan':
			# original scaling behavior of Jordan et al
			return max(1, p.size(-2) / p.size(-1)) ** 0.5

@torch.no_grad()
def _compute_rect(group: _THORNParameterGroup, step: float | int):
	"""
	Compute variance rectification term, from RAdam: https://arxiv.org/abs/1908.03265
	"""
	beta2 = group.betas[1]

	if beta2 > 0.0:
		rho_inf = 2 / (1 - beta2) - 1
		rho = rho_inf - 2 * step * (beta2 ** step) / (1 - beta2 ** step)
		return (
			((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)) ** 0.5
			if rho > 4.0
			else 0.0
		) ** float(group.rectify)
	else:
		return 1.0

def _w1rand(s: int) -> tuple[int, int]:
	C = 0xD07EBC63274654C7
	s = (s + C) & 0xFFFFFFFFFFFFFFFF
	t = s * (s ^ C)
	return s, ((t >> 64) ^ t) & 0xFFFFFFFFFFFFFFFF

def _momentum_aligned_mask(
	g: torch.Tensor,
	state: dict,
	group: _THORNParameterGroup,
	*,
	tau: float = 2.0,
	p: float = 0.9,
	process_group: Optional[dist.ProcessGroup] = None,
	moment: Optional[torch.Tensor] = None
) -> float:
	if not group.momentum_align:
		return 1.0
	# momentum-aligned gradient masking: http://arxiv.org/abs/2602.15322
	g_flat, m_flat = g.flatten(), (state['moment'] if moment is None else moment).flatten()
	if process_group is not None:
		stats = torch.stack([
			(g_flat * m_flat).sum(),
			g_flat.square().sum(),
			m_flat.square().sum()
		]).float()
		dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)
		dot, g_sq, m_sq = stats
		cos_sim = dot / (g_sq.sqrt_() * m_sq.sqrt_()).clamp_min_(1e-12)
		s_t = torch.sigmoid(cos_sim / tau)
	else:
		s_t = torch.sigmoid(nn.functional.cosine_similarity(g_flat, m_flat, dim=0) / tau)
	state['s'] = p * state['s'] + (1 - p) * s_t.item()
	state['random_state'], mask = _w1rand(state['random_state'])
	return state['s'] * (1.0 if mask % 2 == 0 else 0.0)

def _adam_step(
	g: torch.Tensor,
	momentum: torch.Tensor,
	variance: torch.Tensor,
	step: int | torch.Tensor,
	beta1: float = 0.9,
	beta2: float = 0.95,
	degenerate = False
):
	momentum.lerp_(g, weight=1 - beta1)
	variance.mul_(beta2).addcmul_(g, g, value=1 - beta2)
	if not degenerate:
		denom = variance.div(1 - beta2 ** step).sqrt_()
		# atan2 instead of div per https://arxiv.org/pdf/2407.05872
		u = momentum.div(1 - beta1 ** step).atan2_(denom)
	else:
		# clone because we might later call _weight_decay which modifies in place
		u = momentum.clone()
	return u

def _recover_direction(
	p: torch.Tensor,
	state: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
	g = p.grad
	assert g is not None

	if 'row_gain' in state:
		gain = F.softplus(state['row_gain']) * F.softplus(state['col_gain'])
	else:
		gain = F.softplus(state['gain'])

	p.div_(gain)
	p_g = p * g
	g.mul_(gain)
	return g, p_g

def _update_magnitude(
	p_g: torch.Tensor,
	gain: torch.Tensor,
	gain2: torch.Tensor | None,
	dim: int | None,
	moment: torch.Tensor,
	variance: torch.Tensor,
	state: dict[str, torch.Tensor],
	group: _THORNParameterGroup
):
	d_gain = F.sigmoid(gain)
	if gain2 is not None:
		p_g = p_g * F.softplus(gain2)
	grad = p_g.sum(dim=dim).mul_(d_gain.squeeze(dim) if dim is not None else d_gain)
	gain.sub_(_adam_step(
		grad.unsqueeze(dim) if dim is not None else grad,
		moment,
		variance,
		state['step']
	), alpha=group.gain_lr)

def _update_magnitudes(p_g: torch.Tensor, state: dict[str, torch.Tensor], group: _THORNParameterGroup):
	if 'row_gain' in state:
		_update_magnitude(
			p_g,
			state['row_gain'],
			state['col_gain'],
			-1,
			state['row_gain_moment'],
			state['row_gain_variance'],
			state,
			group
		)
		_update_magnitude(
			p_g,
			state['col_gain'],
			state['row_gain'],
			-2,
			state['col_gain_moment'],
			state['col_gain_variance'],
			state,
			group
		)
	else:
		_update_magnitude(
			p_g,
			state['gain'],
			None,
			None,
			state['gain_moment'],
			state['gain_variance'],
			state,
			group
		)

def _reassemble_md(
	p: torch.Tensor,
	state: dict[str, torch.Tensor]
):
	if 'row_gain' in state:
		p.mul_(F.softplus(state['row_gain']))
		p.mul_(F.softplus(state['col_gain']))
	else:
		p.mul_(F.softplus(state['gain']))

@dataclass
class _DistributedTHORNState:
	backend: '_Backend'
	worker_rank: int
	process_group: dist.ProcessGroup
	group: _THORNParameterGroup
	compute_stream: torch.Stream
	comm_stream: torch.Stream

	gathered_grad: Optional[torch.Tensor] = None
	gather_event: Optional[torch.Event] = None
	scattered_u: Optional[torch.Tensor] = None
	scatter_event: Optional[torch.Event] = None
	computed_u: Optional[torch.Tensor] = None
	compute_event: Optional[torch.Event] = None
	# either whole gain or col gain
	grad_gain: Optional[torch.Tensor] = None
	grad_gain_event: Optional[torch.Event] = None
	reduced_norm: Optional[torch.Tensor] = None
	reduce_event: Optional[torch.Event] = None
	sub_event: Optional[torch.Event] = None

	@staticmethod
	def _calc_flops(G: torch.Tensor, steps: int) -> int:
		M, N = G.size(-2), G.size(-1)
		if M > N:
			M, N = N, M

		return steps * ((M ** 3) * 2 + (M ** 2 * N) * 4 + M * N * 2 + M ** 2 * 3)

	@staticmethod
	def _get_shard_mesh(p: DTensor, rank: int) -> tuple[torch.Tensor, dist.ProcessGroup]:
		assert isinstance(p, DTensor)

		if p.placements == (Shard(dim=0),):
			return p.device_mesh.mesh, p.device_mesh.get_group(mesh_dim=0)
		elif p.placements == (Replicate(), Shard(dim=0)):
			for shard_mesh in p.device_mesh.mesh:
				if rank in shard_mesh:
					return shard_mesh, p.device_mesh.get_group(mesh_dim=1)
			raise ValueError('shouldn\'t happen')
		else:
			raise ValueError(f'Unsupported placements {p.placements}')

	@torch.no_grad()
	def gather(
		self,
		p: DTensor,
		rank: int
	):
		with self.backend.stream(self.comm_stream):
			assert p.grad is not None
			g = cast(DTensor, p.grad.to(dtype=torch.float32))
			gather_list = [
				torch.empty_like(g.to_local(), dtype=torch.float32)
				for _ in range(dist.get_world_size(group=self.process_group))
			] if rank == self.worker_rank else None
			dist.gather(
				g.to_local(),
				dst=self.worker_rank,
				gather_list=gather_list,
				group=self.process_group
			)
			if rank == self.worker_rank:
				assert gather_list is not None
				if self.gathered_grad is not None:
					raise RuntimeError('Gather event already exists, which should not happen.')
				self.gathered_grad = torch.cat(gather_list, dim=0)
				self.gather_event = self.backend.Event()
				self.gather_event.record()
			else:
				self.gathered_grad = None
				self.gather_event = None

			gather_list = None
			if self.group.none_grad:
				# We can safely free p.grad without calling record_stream:
				#   p.grad.to_local().record_stream(comm_stream)
				# Explanation:
				# 1. p.grad is created on the default stream, but the default stream
				#    is synchronized with the comm stream later.
				# 2. There is no further activity on the default stream before the optimizer finishes.
				# Therefore, it is safe to free p.grad directly on the comm stream.
				p.grad = None

	@torch.no_grad()
	def compute_u(
		self,
		p: DTensor,
		state: dict[str, torch.Tensor],
		rank: int
	):
		with self.backend.stream(self.compute_stream):
			if rank == self.worker_rank:
				if self.gather_event is None:
					raise RuntimeError('Gather event must be set before compute.')

				self.compute_stream.wait_event(self.gather_event)
				assert self.gathered_grad is not None

				u = _polar_decomp(self.gathered_grad, self.group)
				if _needs_neuron_norm(p, self.group):
					u = _per_neuron_norm(u, state.get('target_norm'), state['moment2'], self.group)

				self.computed_u = u

			self.scattered_u = torch.empty_like(p.to_local(), dtype=torch.float32)
			self.compute_event = self.backend.Event()
			self.compute_event.record()

	@torch.no_grad()
	def scatter(
		self,
		p: DTensor,
		rank: int
	):
		with self.backend.stream(self.comm_stream):
			if self.compute_event is None:
				raise RuntimeError('Compute event must be set before scatter.')
			self.comm_stream.wait_event(self.compute_event)

			if rank == self.worker_rank:
				num_ranks = dist.get_world_size(group=self.process_group)

				# Clear the gathered gradient to free memory
				self.gathered_grad = None

				u = self.computed_u
				assert u is not None
				scatter_list = list(torch.split(u, p.size(0) // num_ranks, dim=0))
				scatter_list = [s.contiguous() for s in scatter_list]
			else:
				scatter_list = None

			torch.distributed.scatter(
				self.scattered_u, # type: ignore
				scatter_list=scatter_list,
				src=self.worker_rank,
				group=self.process_group
			)

			self.scatter_event = self.backend.Event()
			self.scatter_event.record()
			del scatter_list

	@torch.no_grad()
	def update_param(
		self,
		p: DTensor,
		state: dict[str, torch.Tensor],
		rank: int,
		magma_scale: float = 1.0
	):
		with self.backend.stream(self.compute_stream):
			if self.scatter_event is None:
				raise RuntimeError('Scatter event must be set before update')

			self.compute_stream.wait_event(self.scatter_event)
			assert self.scattered_u is not None
			u = DTensor.from_local(self.scattered_u, placements=p.placements, device_mesh=p.device_mesh)
			self.scattered_u = None
			if rank == self.worker_rank:
				self.computed_u = None

			u = _weight_decay(p, u, self.group.weight_decay)
			p.data.sub_(u, alpha=self.group.lr * _lr_scale_ortho(u, self.group) * magma_scale)
			u = None

			if not self.group.decouple_md:
				return

			self.reduced_norm = p.to_local().float().square_().sum()
			self.sub_event = self.backend.Event()
			self.sub_event.record()

		with self.backend.stream(self.comm_stream):
			self.comm_stream.wait_event(self.sub_event)
			assert self.reduced_norm is not None
			dist.all_reduce(self.reduced_norm, op=dist.ReduceOp.SUM, group=self.process_group)
			self.reduce_event = self.backend.Event()
			self.reduce_event.record()

		with self.backend.stream(self.compute_stream):
			self.compute_stream.wait_event(self.reduce_event)
			assert self.reduced_norm is not None
			global_norm = self.reduced_norm.sqrt_()
			self.reduced_norm = None

			if not _needs_neuron_norm(p, self.group):
				p.data.mul_(state['target_norm'] / (global_norm + 1e-8))

			if self.grad_gain_event is not None:
				self.compute_stream.wait_event(self.grad_gain_event)
				assert self.grad_gain is not None
				state['col_gain'].sub_(
					_adam_step(self.grad_gain.unsqueeze(-2), state['col_gain_moment'], state['col_gain_variance'], state['step']),
					alpha=self.group.gain_lr
				)
				self.grad_gain = None
				self.grad_gain_event = None

			_reassemble_md(p, state)

class THORNOrthoGroup(TypedDict, total=True):
	orthogonalize: Literal[True]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	none_grad: NotRequired[bool]
	scaling_mode: NotRequired[Literal['md', 'moonlight', 'jordan']]
	target_rms: NotRequired[float]
	decouple_md: NotRequired[bool]
	gain_lr: NotRequired[float]
	split_gain: NotRequired[bool]
	momentum_align: NotRequired[bool]
	force_per_neuron_norm: NotRequired[bool]
	iters: NotRequired[int]
	lower_bound: NotRequired[float]
	cushion: NotRequired[float]
	safety_factor: NotRequired[float]
	coeffs: NotRequired[list[tuple[float, float, float]]]

class THORNNonOrthoGroup(TypedDict, total=True):
	orthogonalize: Literal[False]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	none_grad: NotRequired[bool]
	rectify: NotRequired[bool]
	momentum_align: NotRequired[bool]
	decouple_md: NotRequired[bool]
	gain_lr: NotRequired[float]
	split_gain: NotRequired[bool]

THORNGroup = Union[THORNOrthoGroup, THORNNonOrthoGroup]

class THORN(Optimizer):
	param_groups: list[dict]
	is_distributed: bool
	rank: int | None = None
	backend: '_Backend'
	comm_stream: torch.Stream | None = None
	compute_stream: torch.Stream | None = None

	@overload
	def __init__(
		self,
		params: nn.Module,
		*,
		lr: float,
		none_grad: bool = True,
		weight_decay: float = 0.1,
		betas: tuple[float, float] = (0.95, 0.95),
		scaling_mode: Optional[Literal['md', 'moonlight', 'jordan']] = None,
		target_rms: float = 0.2,
		target_norm: Optional[_TargetNorm] = None,
		iters: int = 5,
		rectify: bool = True,
		lower_bound: float = 1e-3,
		safety_factor: float = 0.03,
		cushion: float = 0.02,
		momentum_align: bool = False,
		gradient_release: bool = False,
		decouple_md: bool = False,
		gain_lr: float = 1e-3,
		split_gain: bool = True,
		backend = torch.cuda
	):
		...

	@overload
	def __init__(self, params: list[THORNGroup]):
		...

	def __init__(self, params, *, backend = torch.cuda, **conf):
		self._update_rate = 1
		self.backend = cast(_Backend, backend) if TYPE_CHECKING else backend

		param_groups = THORN._auto_assign(params, **conf) if isinstance(params, nn.Module) else cast(list[THORNGroup], params)
		for group in param_groups:
			assert 'orthogonalize' in group
			if group['orthogonalize']:
				assert all(map(lambda x: x.ndim >= 2, group['params'])), 'Only parameters with ndim >= 2 can be orthogonalized'

		self.is_distributed = dist.is_initialized()
		if self.is_distributed:
			self.rank = dist.get_rank()

		defaults = dict()
		for field in fields(_THORNParameterGroup):
			if field.default is not MISSING:
				defaults[field.name] = field.default
			elif field.default_factory is not MISSING:
				defaults[field.name] = field.default_factory()

		super().__init__(cast(list[dict], param_groups), defaults)

		for group in self.param_groups:
			for p in group['params']:
				self.state[p]['group'] = group

		if isinstance(params, nn.Module) and conf.get('gradient_release', False):
			self.setup_gradient_release(params)

	@staticmethod
	def _auto_assign(module: nn.Module, **keys) -> list[THORNGroup]:
		embedding_params = []
		def filter_embeddings(module: nn.Module):
			if isinstance(module, nn.Embedding):
				embedding_params.append(id(module.weight))
		module.apply(filter_embeddings)

		ortho_params = []
		regular_params = []
		for name, param in module.named_parameters():
			if param.ndim >= 2 \
				and 'lm_head' not in name \
				and id(param) not in embedding_params:
				ortho_params.append(param)
			else:
				regular_params.append(param)

		return [
			{
				'orthogonalize': True,
				'params': ortho_params,
				**keys
			},
			{
				'orthogonalize': False,
				'params': regular_params,
				**keys
			}
		]

	def _assign_params(self, params: list[DTensor], group: _THORNParameterGroup) -> tuple[dict[int, _DistributedTHORNState], list[DTensor]]:
		assert self.rank is not None
		assert self.comm_stream is not None and self.compute_stream is not None

		param_to_state: dict[int, _DistributedTHORNState] = {}
		param_to_flops: dict[int, int] = {}

		for p in params:
			g = p.grad
			if g is None:
				continue

			flops = _DistributedTHORNState._calc_flops(g, group.iters)
			param_to_flops[id(p)] = flops

		ordered_params = sorted(params, key=lambda p: param_to_flops[id(p)], reverse=True)

		round_robin = 0
		mesh = None
		shard_mesh = None
		process_group = None
		for p in ordered_params:
			if mesh is None:
				mesh = p.device_mesh
				shard_mesh, process_group = _DistributedTHORNState._get_shard_mesh(p, self.rank)
			elif mesh != p.device_mesh:
				raise ValueError('All parameters must be on the same mesh.')

			assert shard_mesh is not None and process_group is not None

			param_to_state[id(p)] = _DistributedTHORNState(
				backend=self.backend,
				worker_rank=int(shard_mesh[round_robin].item()),
				process_group=process_group,
				group=group,
				compute_stream=self.compute_stream,
				comm_stream=self.comm_stream
			)

			round_robin = (round_robin + 1) % len(shard_mesh)

		return param_to_state, ordered_params

	@torch.no_grad()
	def _update_momentum(self, p: torch.Tensor, g: torch.Tensor, group: _THORNParameterGroup):
		state = self.state[p]
		momentum, beta1 = state['moment'], group.betas[0]
		momentum.mul_(beta1).add_(g)
		return g.add(momentum, alpha=beta1)

	def _base_ortho_step(self, p: Parameter, group: _THORNParameterGroup):
		assert p.grad is not None
		state = self.state[p]
		g = p.grad

		if group.decouple_md:
			g, p_g = _recover_direction(p, state)

		magma_scale = _momentum_aligned_mask(g, state, group)

		u = self._update_momentum(p, g, group)
		if magma_scale != 0.0 and (state['step'] + 1) % self._update_rate == 0:
			u = _polar_decomp(u, group).to(dtype=p.dtype)
			if _needs_neuron_norm(p, group):
				u = _per_neuron_norm(u, state.get('target_norm'), state['moment2'], group)
			u = _weight_decay(p, u, group.weight_decay)

			p.sub_(u, alpha=group.lr * _lr_scale_ortho(u, group) * magma_scale)
			if group.decouple_md:
				if not _needs_neuron_norm(p, group): # row norm didnt rescale to target norm so do it ourselves
					p.mul_(state['target_norm'] / (p.norm(dim=(-2, -1), keepdim=True) + 1e-8))

				_update_magnitudes(p_g, state, group)

		if group.decouple_md:
			_reassemble_md(p, state)

		if group.none_grad:
			del g
			p.grad = None

	def _sharded_ortho_step(self, params: list[DTensor], group: _THORNParameterGroup):
		if self.comm_stream is None or self.compute_stream is None:
			self.comm_stream = self.backend.Stream()
			self.compute_stream = self.backend.Stream()
		assert self.comm_stream is not None and self.compute_stream is not None
		assert self.rank is not None

		_, process_group = _DistributedTHORNState._get_shard_mesh(params[0], self.rank)

		update_params = []
		magma_scales = {}
		pending_grad_col = {}
		pending_grad_col_event = {}

		for p in params:
			g = p.grad
			if g is None:
				continue

			state = self.state[p]

			if group.decouple_md:
				g, p_g = _recover_direction(p, state)

			magma_scale = _momentum_aligned_mask(g, state, group, process_group=process_group)
			magma_scales[id(p)] = magma_scale
			should_update = magma_scale != 0.0 and (state['step'] + 1) % self._update_rate == 0
			if not should_update:
				del g
				p.grad = None
				continue

			g = self._update_momentum(p, g, group)

			if group.decouple_md:
				if 'row_gain' in state:
					# compute row gain now since we already shard across rows
					grad_row = (p_g * F.softplus(state['col_gain'])).sum(dim=-1).mul_(F.sigmoid(state['row_gain']).squeeze(-1))
					state['row_gain'].sub_(
						_adam_step(grad_row.unsqueeze(-1), state['row_gain_moment'], state['row_gain_variance'], state['step']),
						alpha=group.gain_lr
					)

					# start col gain reduction so its ready by the update_param step
					grad_col = (p_g * F.softplus(state['row_gain'])).sum(dim=-2).mul_(F.sigmoid(state['col_gain']).squeeze(-2))
					self.comm_stream.wait_stream(self.backend.current_stream())
					with self.backend.stream(self.comm_stream):
						dist.all_reduce(grad_col, op=dist.ReduceOp.SUM, group=process_group)
						grad_col_event = self.backend.Event()
						grad_col_event.record()

					pending_grad_col[id(p)] = grad_col
					pending_grad_col_event[id(p)] = grad_col_event
				elif 'gain' in state:
					grad_gain = p_g.sum().mul_(F.sigmoid(state['gain']))
					self.comm_stream.wait_stream(self.backend.current_stream())
					with self.backend.stream(self.comm_stream):
						dist.all_reduce(grad_gain, op=dist.ReduceOp.SUM, group=process_group)
						grad_gain_event = self.backend.Event()
						grad_gain_event.record()

					pending_grad_col[id(p)] = grad_gain
					pending_grad_col_event[id(p)] = grad_gain_event

			p.grad = g
			update_params.append(p)

		if not update_params:
			return

		param_to_state, ordered_params = self._assign_params(update_params, group)
		for p in ordered_params:
			dist_state = param_to_state[id(p)]
			dist_state.grad_gain = pending_grad_col.get(id(p))
			dist_state.grad_gain_event = pending_grad_col_event.get(id(p))

		def enqueue_gathers(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.gather(p, rank=self.rank)

		def enqueue_computes(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.compute_u(p, self.state[p], rank=self.rank)

		def enqueue_scatters(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.scatter(p, rank=self.rank)

		def enqueue_updates(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.update_param(p, self.state[p], rank=self.rank, magma_scale=magma_scales[id(p)])
				self.state[p]['step'] += 1

		chunk_size = dist.get_world_size(param_to_state[id(ordered_params[0])].process_group)

		self.comm_stream.wait_stream(self.backend.current_stream())

		i = 0
		enqueue_gathers(0, chunk_size)
		for i in range(0, len(ordered_params) + chunk_size - 1, chunk_size):
			enqueue_computes(i, chunk_size)
			if i > 0:
				enqueue_updates(i - chunk_size, chunk_size)
			enqueue_gathers(i + chunk_size, chunk_size)
			enqueue_scatters(i, chunk_size)
		enqueue_updates(i, chunk_size)

		self.backend.current_stream().wait_stream(self.compute_stream)

	def _step_params(self, params: list[torch.nn.Parameter], group: _THORNParameterGroup):
		distributed_params = []
		regular_params = []
		for p in params:
			g = p.grad
			if p is None or g is None:
				continue

			state = self.state[p]
			if 'step' not in state:
				state['step'] = 1

				if group.orthogonalize:
					state['moment'] = torch.zeros_like(p)
					if _needs_neuron_norm(p, group):
						state['moment2'] = torch.zeros((*p.shape[:-1], 1), dtype=p.dtype, device=p.device)
				else:
					state['moment'] = torch.zeros_like(p)
					state['variance'] = torch.zeros_like(p)

				if group.decouple_md:
					LN_E_SUB_1 = 0.5413248546 # x = ln(e - 1); softplus(x) = 1
					if p.ndim > 1 and group.split_gain:
						state['row_gain'] = torch.full((*p.shape[:-1], 1), LN_E_SUB_1, dtype=p.dtype, device=p.device)
						state['col_gain'] = torch.full((*p.shape[:-2], 1, *p.shape[-1:]), LN_E_SUB_1, dtype=p.dtype, device=p.device)
						for k in ['row_gain', 'col_gain']:
							state[f'{k}_moment'] = torch.zeros_like(state[k])
							state[f'{k}_variance'] = torch.zeros_like(state[k])
					else:
						state['gain'] = torch.full((), LN_E_SUB_1, dtype=p.dtype, device=p.device)
						state['gain_moment'] = torch.zeros((), dtype=p.dtype, device=p.device)
						state['gain_variance'] = torch.zeros((), dtype=p.dtype, device=p.device)

					state['target_norm'] = group.target_norm_(p)

				if group.momentum_align:
					# use torch.randint so torch.manual_seed works as expected
					# only 63 bits because of https://github.com/pytorch/pytorch/issues/191458 (completely ridiculous)
					seed = torch.randint(0, (1 << 63) - 1, (), device=g.device, dtype=torch.uint64).item()

					if (
						isinstance(p.data, DTensor)
						and group.orthogonalize
						and not all(isinstance(placement, Replicate) for placement in cast(DTensor, p).placements)
					):
						assert self.rank is not None
						# need to ensure prng seed is synchronized so magma steps dont diverge
						_, sync_group = _DistributedTHORNState._get_shard_mesh(cast(DTensor, p), self.rank)
						container = [seed]
						dist.broadcast_object_list(container, src=dist.get_global_rank(sync_group, 0), group=sync_group)
						seed = container[0]

					state['random_state'] = seed
					state['s'] = 1.0

			if isinstance(p.data, DTensor):
				if all(isinstance(placement, Replicate) for placement in cast(DTensor, p).placements) or not group.orthogonalize:
					regular_params.append(p)
				else:
					distributed_params.append(cast(DTensor, p))
			else:
				regular_params.append(p)

		if len(distributed_params) > 0:
			self._sharded_ortho_step(distributed_params, group)

		for p in regular_params:
			state = self.state[p]
			if group.orthogonalize:
				self._base_ortho_step(p, group)
			else:
				g = p.grad
				assert g is not None

				step = state['step']
				beta1, beta2 = group.betas
				rect = _compute_rect(group, step)

				if g.is_sparse and group.decouple_md or g.sparse_dim() != 1:
					p.grad = g = g.to_dense()
				elif g.is_sparse:
					g = g.coalesce()
					g_rows = g.values()
					idxs = g.indices()[0]
					if idxs.numel() == 0:
						continue

					moment_rows = state['moment'].index_select(0, idxs)
					variance_rows = state['variance'].index_select(0, idxs)

					magma_scale = _momentum_aligned_mask(g_rows, state, group, moment=moment_rows)

					u = _adam_step(g_rows, moment_rows, variance_rows, step, beta1, beta2, degenerate=rect == 0.0)
					state['moment'].index_copy_(0, idxs, moment_rows)
					state['variance'].index_copy_(0, idxs, variance_rows)

					if magma_scale != 0.0 and (step + 1) % self._update_rate == 0:
						u = _weight_decay(p.index_select(0, idxs), u, group.weight_decay)
						p.index_add_(0, idxs, u, alpha=-group.lr * rect * magma_scale)

					continue

				magma_scale = _momentum_aligned_mask(g, state, group)

				if group.decouple_md:
					g, p_g = _recover_direction(p, state)

				u = _adam_step(g, state['moment'], state['variance'], step, beta1, beta2, degenerate=rect == 0.0)

				if magma_scale != 0.0 and (step + 1) % self._update_rate == 0:
					u = _weight_decay(p, u, group.weight_decay)
					p.sub_(u, alpha=group.lr * rect * magma_scale)

					if group.decouple_md:
						p.mul_(state['target_norm'] / (p.norm(dim=(-2, -1) if p.ndim > 1 else -1, keepdim=True) + 1e-8))

						_update_magnitudes(p_g, state, group)

				if group.decouple_md:
					_reassemble_md(p, state)

				if group.none_grad:
					del g
					p.grad = None

			state['step'] += 1

	@overload
	def step(self, closure: None = None, *, param: Optional[torch.nn.Parameter] = None) -> None: ...
	@overload
	def step(self, closure: Callable[[], float], *, param: Optional[torch.nn.Parameter] = None) -> float: ...
	@torch.no_grad()
	def step(self, closure: Optional[Callable[[], float]] = None, *, param: Optional[torch.nn.Parameter] = None) -> Optional[float]:
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		if param is None:
			for group in self.param_groups:
				params = group['params']
				self._step_params(params, _THORNParameterGroup.from_kwargs(**group))
		else:
			state = self.state[param]
			group = _THORNParameterGroup.from_kwargs(**state['group'])
			self._step_params([param], group)

		return loss

	def setup_gradient_release(
		self,
		model: nn.Module,
		update_rate: int = 1,
		ignore_existing_hooks: bool = False
	):
		def _gradient_release_hook(param: torch.Tensor, optimizer: THORN):
			optimizer.step(param=cast(Parameter, param))
			param.grad = None

		hooks = []
		for p in model.parameters():
			if p.requires_grad:
				if p._post_accumulate_grad_hooks is not None and len(p._post_accumulate_grad_hooks) > 0 and not ignore_existing_hooks:
					for hook in hooks:
						if hasattr(hook, 'remove'):
							hook.remove()
					raise ValueError('Model already has post_accumulate_grad_hooks. If this is expected, pass `ignore_existing_hooks=True`.')
				hooks.append(p.register_post_accumulate_grad_hook(partial(_gradient_release_hook, optimizer=self)))
		model._gradient_release_hooks = hooks # type: ignore

		self._update_rate = update_rate
