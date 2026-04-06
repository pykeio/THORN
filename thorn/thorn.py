############################################################################
# Copyright 2025-2026 pyke.io                                              #
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
from typing import cast, overload, Callable, Literal, Optional, TypedDict, Union, NotRequired

import torch
import torch.cuda as cuda
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.optimizer import _get_value

_has_triton = False
try:
	import triton # ty:ignore[unresolved-import]
	_has_triton = True
except ModuleNotFoundError:
	pass
except RuntimeError:
	pass

_use_triton = _has_triton and environ.get('THORN_DISABLE_TRITON') != '1'
if _use_triton:
	import triton                # ty:ignore[unresolved-import]
	import triton.language as tl # ty:ignore[unresolved-import]

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
		stride_am, stride_ak,
		stride_bk, stride_bn,
		stride_cm, stride_cn,
		stride_om, stride_on,
		BLOCK_SIZE_M: tl.constexpr,
		BLOCK_SIZE_K: tl.constexpr,
		GROUP_SIZE_M: tl.constexpr,
	):
		# adapted from https://github.com/nil0x9/flash-muon
		pid = tl.program_id(axis=0)
		num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
		num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
		num_pid_in_group = GROUP_SIZE_M * num_pid_n
		group_id = pid // num_pid_in_group
		first_pid_m = group_id * GROUP_SIZE_M
		group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
		pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
		pid_n = (pid % num_pid_in_group) // group_size_m

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

	A = (M, K), B = (K, M), C/out = (M, M)
	"""
	assert A.is_cuda and B.is_cuda and C.is_cuda
	assert A.device == B.device == C.device
	assert A.dtype == B.dtype == C.dtype
	assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2

	M, K = A.shape
	assert B.shape == (K, M), f"B must be ({K}, {M}), got {tuple(B.shape)}"
	assert C.shape == (M, M), f"C must be ({M}, {M}), got {tuple(C.shape)}"

	A = A.contiguous()
	B = B.contiguous()
	C = C.contiguous()

	if out is not None:
		assert out.shape == (M, M), f"out must be ({M}, {M}), got {tuple(out.shape)}"
		assert out.dtype == A.dtype
		assert out.device == A.device
		O = out
	elif _use_triton:
		O = torch.empty((M, M), dtype=A.dtype, device=A.device)

	if not _use_triton:
		# torch.addmm doesn't support out == C like our kernel does, so we can't pass out here
		O = torch.addmm(C, A, B, alpha=alpha, beta=beta)
	else:
		grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) ** 2,)
		with torch.cuda.device(A.device.index):
			_sym_addmm_kernel[grid](
				A, B, C, O,
				alpha, beta,
				M, K,
				A.stride(0), A.stride(1),
				B.stride(0), B.stride(1),
				C.stride(0), C.stride(1),
				O.stride(0), O.stride(1)
			)

	return O

try:
	if environ.get('THORN_COMPILE') != '1':
		raise Exception()

	@torch.compile(dynamic=False, fullgraph=True)
	@torch.no_grad()
	def test(x: torch.Tensor):
		return x + 1.

	x = test(torch.tensor([1.0, 2.0]).cuda())
	assert torch.allclose(x.cpu(), torch.tensor([2.0, 3.0]))

	del x
	del test
	_optional_compile = torch.compile # type: ignore
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

		q = (3. * l + 1.) / 4.
		r = (l + 3.) / 4.
		E, old_E = float('inf'), None
		a, b, c = 1., 1., 1.
		while not old_E or abs(old_E - E) > eps_d:
			old_E = E
			LHS = torch.tensor([
				[l, l ** 3., l ** 5.,  1.],
				[q, q ** 3., q ** 5., -1.],
				[r, r ** 3., r ** 5.,  1.],
				[u, u ** 3., u ** 5., -1.],
			], dtype=torch.double)
			a, b, c, E = torch.linalg.solve(LHS, torch.ones(4, dtype=torch.double))
			q, r = torch.sqrt((-3. * b + torch.tensor([-1., 1.], dtype=torch.double) * (9. * b ** 2. - 20. * a * c) ** 0.5) / (10. * c))
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
	E = torch.logspace(0, -10, 10000, dtype=torch.float64)
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

@dataclass
class _THORNParameterGroup:
	orthogonalize: bool
	params: list[Parameter]
	lr: float
	none_grad: bool = field(default=True)
	eps: float = field(default=1e-8)
	weight_decay: float = field(default=0.1)
	betas: tuple[float, float] = field(default_factory=lambda: (0.95, 0.95))
	iters: int = field(default=5)
	rectify: bool = field(default=True)
	target_rms: float = field(default=0.2)
	lower_bound: float = field(default=1e-3)
	safety_factor: float = field(default=0.05)
	cushion: float = field(default=0.02)
	momentum_align: bool = field(default=False)
	coeffs: list[tuple[float, float, float]] = field(init=False)
	restarts: list[int] = field(init=False)

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

def _resize(g: torch.Tensor):
	if g.ndim > 2: # for conv filters
		g = g.reshape(g.size(0), -1).contiguous()
	return g

@_optional_compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def _polar_decomp(X: torch.Tensor, group: _THORNParameterGroup):
	dtype = X.dtype
	if should_transpose := X.size(-2) > X.size(-1):
		X = X.mT

	X = X.to(torch.float32)
	X /= X.norm(dim=(-2, -1), keepdim=True) + 1e-6
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

@torch.no_grad()
def _per_neuron_norm(u: torch.Tensor, m2: torch.Tensor | None, group: _THORNParameterGroup):
	# Per-neuron normalization, from https://arxiv.org/abs/2510.05491
	if group.betas[1] > 0:
		assert m2 is not None, 'beta2 cannot be enabled mid-run'
		v_norm = u.norm(dim=(-2, -1), keepdim=True)
		v_mean = u.square().mean(dim=-1, keepdim=True)
		m2.lerp_(v_mean.to(m2.dtype), 1. - group.betas[1])
		u.mul_(m2.clamp_min(group.eps).rsqrt_())
		v_norm_new = u.norm(dim=(-2, -1), keepdim=True)
		u.mul_(v_norm.div_(v_norm_new.clamp_min_(group.eps)))
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

def _lr_scale_ortho(p: torch.Tensor, target_rms: float = 0.2):
	if target_rms != 0.0:
		# Scale LR to match RMS update of AdamW so AdamW's LR can be reused
		# per formula 4 of https://arxiv.org/pdf/2502.16982
		return target_rms * (max(p.shape[:2]) ** 0.5)
	else:
		# Match original behavior of Jordan et al
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

def _momentum_aligned_mask(g: torch.Tensor, state: dict, group: _THORNParameterGroup, *, tau: float = 2.0, p: float = 0.9) -> float:
	if not group.momentum_align:
		return 1.0
	# momentum-aligned gradient masking: http://arxiv.org/abs/2602.15322
	s_t = torch.sigmoid(nn.functional.cosine_similarity(g.flatten(), state['moment'].flatten(), dim=0) / tau)
	state['s'] = p * state['s'] + (1 - p) * s_t.item()
	state['random_state'], mask = _w1rand(state['random_state'])
	return state['s'] * (1.0 if mask % 2 == 0 else 0.0)

@dataclass
class _DistributedTHORNState:
	worker_rank: int
	process_group: dist.ProcessGroup
	gathered_grad: Optional[torch.Tensor] = None
	scattered_u: Optional[torch.Tensor] = None
	computed_u: Optional[torch.Tensor] = None
	gather_event: Optional[torch.Event] = None
	scatter_event: Optional[torch.Event] = None
	compute_event: Optional[torch.Event] = None

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
		group: _THORNParameterGroup,
		rank: int,
		comm_stream: cuda.Stream
	):
		with cuda.stream(comm_stream):
			assert p.grad is not None
			g = cast(DTensor, _resize(p.grad).to(dtype=torch.float32))
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
				if self.gathered_grad is not None:
					raise RuntimeError('Gather event already exists, which should not happen.')
				self.gathered_grad = torch.cat(gather_list, dim=0)
				self.gather_event = cast(torch.Event, cuda.Event())
				self.gather_event.record()
			else:
				self.gathered_grad = None
				self.gather_event = None

			gather_list = None
			if group.none_grad:
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
		m2: torch.Tensor | None,
		group: _THORNParameterGroup,
		rank: int,
		compute_stream: cuda.Stream
	):
		with cuda.stream(compute_stream):
			if rank == self.worker_rank:
				if self.gather_event is None:
					raise RuntimeError('Gather event must be set before compute.')

				compute_stream.wait_event(self.gather_event)
				assert self.gathered_grad is not None

				u = _polar_decomp(self.gathered_grad, group)
				u = _per_neuron_norm(u, m2, group)

				self.computed_u = u

			self.scattered_u = torch.empty_like(_resize(p.to_local()), dtype=torch.float32) # type: ignore
			self.compute_event = cast(torch.Event, cuda.Event())
			self.compute_event.record()
			u = None

	@torch.no_grad()
	def scatter(
		self,
		p: DTensor,
		rank: int,
		comm_stream: cuda.Stream
	):
		with cuda.stream(comm_stream):
			if self.compute_event is None:
				raise RuntimeError('Compute event must be set before scatter.')
			comm_stream.wait_event(self.compute_event)

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

			self.scatter_event = cast(torch.Event, torch.cuda.Event())
			self.scatter_event.record()
			scatter_list = None

	def update_param(
		self,
		p: DTensor,
		group: _THORNParameterGroup,
		step: int,
		rank: int,
		compute_stream: cuda.Stream,
		scale: float = 1.0
	):
		with torch.cuda.stream(compute_stream):
			if self.scatter_event is None:
				raise RuntimeError('Scatter event must be set before update')

			compute_stream.wait_event(self.scatter_event)
			assert self.scattered_u is not None
			u_dtensor = DTensor.from_local(self.scattered_u, placements=p.placements, device_mesh=p.device_mesh)
			self.scattered_u = u_dtensor
			if rank == self.worker_rank:
				self.computed_u = None

			u = self.scattered_u.view_as(p)
			u = _weight_decay(p, u, group.weight_decay)
			p.data.sub_(u, alpha=group.lr * _lr_scale_ortho(u, target_rms=group.target_rms) * scale)

			self.scattered_u = None
			u_dtensor = None

class THORNOrthogonalizedParameterGroup(TypedDict, total=True):
	orthogonalize: Literal[True]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	iters: NotRequired[int]
	eps: NotRequired[float]
	none_grad: NotRequired[bool]
	lower_bound: NotRequired[float]
	safety_factor: NotRequired[float]
	cushion: NotRequired[float]
	target_rms: NotRequired[float]
	rectify: NotRequired[bool]
	momentum_align: NotRequired[bool]
	coeffs: NotRequired[list[tuple[float, float, float]]]

class THORNNonOrthogonalizedParameterGroup(TypedDict, total=True):
	orthogonalize: Literal[False]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	eps: NotRequired[float]
	rectify: NotRequired[bool]
	momentum_align: NotRequired[bool]
	none_grad: NotRequired[bool]

THORNParameterGroup = Union[THORNOrthogonalizedParameterGroup, THORNNonOrthogonalizedParameterGroup]

class THORN(Optimizer):
	param_groups: list[dict]
	is_distributed: bool
	rank: int | None = None
	comm_stream = cuda.Stream()
	compute_stream = cuda.Stream()

	@overload
	def __init__(
		self,
		module: nn.Module,
		lr: float,
		*,
		none_grad: bool = True,
		eps: float = 1e-8,
		weight_decay: float = 0.1,
		betas: tuple[float, float] = (0.95, 0.95),
		iters: int = 5,
		rectify: bool = True,
		lower_bound: float = 1e-3,
		safety_factor: float = 0.05,
		cushion: float = 0.02,
		momentum_align: bool = False,
		gradient_release: bool = False
	):
		...

	@overload
	def __init__(self, param_groups: list[THORNParameterGroup]):
		...

	def __init__(self, val, **conf):
		self._update_rate = 1

		param_groups = THORN._auto_assign(val, **conf) if isinstance(val, nn.Module) else cast(list[THORNParameterGroup], val)
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

		if isinstance(val, nn.Module) and conf.get('gradient_release', False):
			self.setup_gradient_release(val)

	@staticmethod
	def _auto_assign(module: nn.Module, **keys) -> list[THORNParameterGroup]:
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

	def _assign_params(self, params: list[DTensor], group: _THORNParameterGroup):
		assert self.rank is not None

		param_to_state: dict[int, _DistributedTHORNState] = {}
		param_to_flops: dict[int, int] = {}

		for p in params:
			g = p.grad
			if g is None:
				continue

			g = _resize(g)

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
				worker_rank=int(shard_mesh[round_robin].item()),
				process_group=process_group
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
		g = _resize(p.grad)

		magma_scale = _momentum_aligned_mask(g, state, group)
		u = self._update_momentum(p, g, group)
		if magma_scale != 0.0 and (state['step'] + 1) % self._update_rate == 0:
			u = _polar_decomp(u, group).to(dtype=p.dtype)
			u = _per_neuron_norm(u, state['moment2'], group)
			u = _weight_decay(p, u.view_as(p), group.weight_decay)
			p.data.sub_(u, alpha=group.lr * _lr_scale_ortho(u, target_rms=group.target_rms) * magma_scale)

		if group.none_grad:
			del g
			p.grad = None

	def _sharded_ortho_step(self, params: list[DTensor], group: _THORNParameterGroup):
		update_params = []
		update_scales = {}
		for p in params:
			g = p.grad
			if g is None:
				continue

			g = _resize(g)
			state = self.state[p]
			magma_scale = _momentum_aligned_mask(g, state, group)
			g = self._update_momentum(p, g, group)
			if magma_scale != 0.0 and (self.state[p]['step'] + 1) % self._update_rate == 0:
				p.grad = g.view_as(p)
				update_params.append(p)
				update_scales[id(p)] = magma_scale
			else:
				del g
				p.grad = None

		param_to_state, ordered_params = self._assign_params(params, group)

		def enqueue_gathers(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.gather(p, group, self.rank, self.comm_stream)

		def enqueue_computes(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.compute_u(
					p,
					self.state[p].get('moment2'),
					group,
					rank=self.rank,
					compute_stream=self.compute_stream
				)

		def enqueue_scatters(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.scatter(p, self.rank, self.comm_stream)

		def enqueue_updates(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				state.update_param(
					p,
					group,
					step=self.state[p]['step'],
					rank=self.rank,
					compute_stream=self.compute_stream,
					scale=update_scales[id(p)]
				)
				self.state[p]['step'] += 1

		chunk_size = dist.get_world_size(param_to_state[id(params[0])].process_group)

		self.comm_stream.wait_stream(cuda.current_stream())

		i = 0
		enqueue_gathers(0, chunk_size)
		for i in range(0, len(params) + chunk_size - 1, chunk_size):
			enqueue_computes(i, chunk_size)
			if i > 0:
				enqueue_updates(i - chunk_size, chunk_size)
			enqueue_gathers(i + chunk_size, chunk_size)
			enqueue_scatters(i, chunk_size)
		enqueue_updates(i, chunk_size)

		cuda.current_stream().wait_stream(self.compute_stream)

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
				state['random_state'] = torch.randint(0, 1 << 52, (), dtype=torch.uint64).item()
				state['s'] = 1.0
				if group.orthogonalize:
					g = _resize(g)
					state['moment'] = torch.zeros_like(g)
					state['moment2'] = torch.zeros((g.shape[0], 1), dtype=g.dtype, device=g.device)
				else:
					state['moment'] = torch.zeros_like(g)
					state['variance'] = torch.zeros_like(g)

			if isinstance(p.data, DTensor):
				if all(isinstance(placement, Replicate) for placement in cast(DTensor, p).placements) or not group.orthogonalize:
					regular_params.append(p)
				else:
					distributed_params.append(p)
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

				step = state['step']
				beta1, beta2 = group.betas

				magma_scale = _momentum_aligned_mask(g, state, group)
				rect = _compute_rect(group, step)

				momentum = state['moment']
				variance = state['variance']

				momentum.lerp_(g, weight=1 - beta1)
				variance.mul_(beta2).addcmul_(g, g, value=1 - beta2)
				if rect > 0.0:
					denom = variance.div(1 - beta2 ** step).sqrt_()
					# atan2 instead of div per https://arxiv.org/pdf/2407.05872
					u = momentum.atan2(denom)
				else:
					# clone because _weight_decay modifies in place
					u = momentum.clone()

				should_update = magma_scale != 0.0 and (state['step'] + 1) % self._update_rate == 0
				if should_update:
					u = _weight_decay(p, u, group.weight_decay)

				p.sub_(u, alpha=group.lr * rect * magma_scale)

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
