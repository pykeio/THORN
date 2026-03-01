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

from dataclasses import dataclass
from functools import partial
from typing import cast, overload, Callable, Literal, Optional, TypedDict, Union, NotRequired

import torch
import torch.cuda as cuda
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.optimizer import _get_value

@dataclass
class _THORNState:
	worker_rank: int
	process_group: dist.ProcessGroup
	gathered_grad: Optional[torch.Tensor] = None
	scattered_u: Optional[torch.Tensor] = None
	computed_u: Optional[torch.Tensor] = None
	gather_event: Optional[torch.Event] = None
	scatter_event: Optional[torch.Event] = None
	compute_event: Optional[torch.Event] = None

_has_triton = False
try:
	from triton.compiler.compiler import triton_key
	_has_triton = triton_key is not None
except ModuleNotFoundError:
	pass
except RuntimeError:
	pass

if _has_triton:
	import triton
	import triton.language as tl

	_autotune_conf = [
		triton.Config({'BLOCK_SIZE_M': blk_m, 'BLOCK_SIZE_K': blk_k, 'GROUP_SIZE_M': grp_sz}, num_stages=n_stages, num_warps=n_warps)
		for blk_m in [32, 64, 128] 
		for blk_k in [32, 64]
		for grp_sz in [8]
		for n_stages in [3, 4, 5]
		for n_warps  in [4, 8]
	]
	@triton.autotune(configs=_autotune_conf, key=['M', 'K'])
	@triton.jit
	def _mmt_kernel(x, y, M, K, stride_xm, stride_xk, stride_ym, stride_yn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
		"""
		Kernel computing y = x @ x.T, exploiting the fact that this produces a symmetrical matrix. From https://github.com/nil0x9/flash-muon
		"""

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

		offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
		offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
		offs_k = tl.arange(0, BLOCK_SIZE_K)
		# we use a & b ptrs to denote different rows of x.
		a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
		b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk) 

		accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

		for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
			a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
			b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
			accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
			a_ptrs += BLOCK_SIZE_K * stride_xk
			b_ptrs += BLOCK_SIZE_K * stride_xk
		c = accumulator.to(x.dtype.element_ty)

		offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
		offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
		c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
		c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
		tl.store(c_ptrs, c, mask=c_mask)

		# transpose and copy
		if pid_m < pid_n:
			ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
			ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
			tl.store(ct_ptrs, tl.permute(c, (1,0)), mask=ct_mask)

	@torch.no_grad()
	def _mmt_assign(x: torch.Tensor, y: torch.Tensor):
		assert x.is_cuda and y.is_cuda
		assert x.device == y.device
		assert x.dtype == y.dtype
		assert x.ndim == 2 and y.ndim == 2
		assert x.size(0) == y.size(0) == y.size(1)

		x = x.contiguous()
		M, K = x.shape
		grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']),)
		with torch.cuda.device(x.device.index):
			_mmt_kernel[grid](x, y, M, K, x.stride(0), x.stride(1), y.stride(0), y.stride(1))
else:
	import warnings
	warnings.warn('Triton not found; using slow MMT path.')

	@torch.no_grad()
	def _mmt_assign(x: torch.Tensor, y: torch.Tensor):
		torch.mm(x, x.mT, out=y)

try:
	@torch.compile(dynamic=False, fullgraph=True)
	@torch.no_grad()
	def test(x: torch.Tensor):
		return x + 1.
	
	x = test(torch.tensor([1.0, 2.0]).cuda())
	assert torch.allclose(x.cpu(), torch.tensor([2.0, 3.0]))

	del x
	del test
	_optional_compile = torch.compile # type: ignore
except Exception as e:
	def _optional_compile(model: None = None, *, fullgraph: bool = False, dynamic: bool = False):
		def fn(c):
			return c
		return fn

def _optimal_composition(l: float, num_iters: int, safety_factor_eps: float = 0.0, cushion: float = 0.0):
    """
    Calculate coefficients for `_polar_decomp`, from https://arxiv.org/pdf/2505.16932
    """
    u = 1.
    assert 0 <= l <= u

    def optimal_quintic(l: float, u: float):
        assert 0 <= l <= u

        if 1. - 5e-6 <= l / u:
            return (15. / 8.) / u, \
                (-10. / 8.) / (u ** 3.), \
                (3. / 8.) / (u ** 5.)

        q = (3. * l + 1.) / 4.
        r = (l + 3.) / 4.
        E, old_E = float('inf'), None
        a, b, c = 1., 1., 1.
        while not old_E or abs(old_E - E) > 1e-15:
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

    safety_factor = 1. + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l ** 3. + c * l ** 5.
            pu = a * u + b * u ** 3. + c * u ** 5.
            rescaler = 2. / (pl + pu)
            a = a * rescaler
            b = b * rescaler
            c = c * rescaler

        if iter < num_iters - 1:
            a = a / safety_factor
            b = b / safety_factor ** 3.
            c = c / safety_factor ** 5.

        coefficients.append((a, b, c))
        
        l = a * l + b * l ** 3. + c * l ** 5.
        u = 2. - l

    return coefficients

@_optional_compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def _polar_decomp(
	G: torch.Tensor,
	coeffs: list[tuple[float, float, float]],
	steps: int | None = None,
	eps: float = 1e-7,
	safety_factor_eps: float = 0.0
):
	X = G
	if G.size(-2) > G.size(-1):
		X = X.mT

	# Ensure spectral norm is at most 1
	X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1. + safety_factor_eps) + eps)

	X = X.contiguous()
	A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
	AA = torch.empty_like(A)
	B = torch.empty_like(A)
	C = torch.empty_like(X)

	is_large_matrix = G.size(-2) > 1024 or G.size(-1) > 1024
	_mm = torch.bmm if X.ndim > 2 else torch.mm
	_addmm = torch.baddbmm if X.ndim > 2 else torch.addmm

	for i, (a, b, c) in enumerate(coeffs[:steps]):
		_mmt_assign(X, A) # A = X @ X.mT
		_mmt_assign(A, AA) # AA = A @ A.mT - note that AA = AA^T because A is symmetrical.
		A.mul_(b) # A = b * A
		torch.add(A, AA, alpha=c, out=B) # B = A + c * AA
		
		if is_large_matrix:
			_mm(B, X, out=C) # C = B @ X
			C.add_(X, alpha=a) # C = C + a * X
		else:
			_addmm(X, B, X, beta=a, out=C) # C = a * X + B @ X

		X, C = C, X

	if G.size(-2) > G.size(-1):
		X = X.mT

	return X

@torch.no_grad()
def _apply_per_neuron_norm(
	u: torch.Tensor,
	v: torch.Tensor,
	beta2: float = 0.95,
	eps: float = 1e-7
):
	# Per-neuron normalization, from https://arxiv.org/abs/2510.05491
	v_norm = u.norm(dim=(-2, -1), keepdim=True)
	v_mean = torch.mean(u * u, dim=-1, keepdim=True)
	v.lerp_(v_mean.to(v.dtype), 1. - beta2)
	u.mul_((v + eps).rsqrt())
	v_norm_new = u.norm(dim=(-2, -1), keepdim=True)
	u.mul_(v_norm / (v_norm_new.add_(eps)))
	return u

def _resize(g: torch.Tensor):
	if g.ndim > 2: # for conv filters
		g = g.reshape(g.size(0), -1).contiguous()
	return g

@torch.no_grad()
def _gather(
	p: Parameter,
	state: _THORNState,
	rank: int,
	comm_stream: cuda.Stream,
	none_grad: bool = False
):
	with cuda.stream(comm_stream):
		assert p.grad is not None
		g = cast(DTensor, _resize(p.grad))

		if rank == state.worker_rank:
			num_ranks = dist.get_world_size(group=state.process_group)
			gather_list = [torch.empty_like(g.to_local(), dtype=torch.float32) for _ in range(num_ranks)]
		else:
			gather_list = None

		g = cast(DTensor, g.to(dtype=torch.float32))
		dist.gather(
			g.to_local(),
			dst=state.worker_rank,
			gather_list=gather_list,
			group=state.process_group
		)
		if rank == state.worker_rank:
			if state.gathered_grad is not None:
				raise RuntimeError('Gather event already exists, which should not happen.')
			state.gathered_grad = torch.cat(gather_list, dim=0)
			state.gather_event = cast(torch.Event, cuda.Event())
			state.gather_event.record()
		else:
			state.gathered_grad = None
			state.gather_event = None

		gather_list = None
		if none_grad:
			# We can safely free p.grad without calling record_stream:
			#   p.grad.to_local().record_stream(comm_stream)
			# Explanation:
			# 1. p.grad is created on the default stream, but the default stream
			#    is synchronized with the comm stream later.
			# 2. There is no further activity on the default stream before the optimizer finishes.
			# Therefore, it is safe to free p.grad directly on the comm stream.
			p.grad = None

@torch.no_grad()
def _compute_u(
	p: Parameter,
	v: torch.Tensor,
	state: _THORNState,
	rank: int,
	compute_stream: cuda.Stream,
	coeffs: list[tuple[float, float, float]],
	steps: int = 5,
	beta2: float = 0.95,
	eps: float = 1e-7,
	safety_factor_eps: float = 0.0
):
	with cuda.stream(compute_stream):
		if rank == state.worker_rank:
			if state.gather_event is None:
				raise RuntimeError('Gather event must be set before compute.')
			
			compute_stream.wait_event(state.gather_event)
			assert state.gathered_grad is not None

			u = _polar_decomp(state.gathered_grad, coeffs, steps, eps=eps, safety_factor_eps=safety_factor_eps)
			if beta2 > 0.0:
				u = _apply_per_neuron_norm(u, v, beta2=beta2, eps=eps)

			state.computed_u = u
		
		state.scattered_u = torch.empty_like(_resize(p.to_local()), dtype=torch.float32) # type: ignore
		state.compute_event = cast(torch.Event, cuda.Event())
		state.compute_event.record()
		u = None

@torch.no_grad()
def _apply_param_update(
	p: torch.Tensor,
	update: torch.Tensor,
	lr: float,
	weight_decay: float,
	scale: float = 1.
):
	if weight_decay > 0.0:
		# "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
		mask = (update * p) >= 0
		update.addcmul_(p, (weight_decay * mask).to(dtype=p.dtype))
	p.data.add_(update, alpha=-lr / scale)

@torch.no_grad()
def _scatter(
	p: DTensor,
	state: _THORNState,
	rank: int,
	comm_stream: cuda.Stream
):
	with cuda.stream(comm_stream):
		if state.compute_event is None:
			raise RuntimeError('Compute event must be set before scatter.')
		comm_stream.wait_event(state.compute_event)

		if rank == state.worker_rank:
			num_ranks = dist.get_world_size(group=state.process_group)

			# Clear the gathered gradient to free memory
			state.gathered_grad = None

			u = state.computed_u
			assert u is not None
			scatter_list = list(torch.split(u, p.size(0) // num_ranks, dim=0))
			scatter_list = [s.contiguous() for s in scatter_list]
		else:
			scatter_list = None

		torch.distributed.scatter(
			state.scattered_u, # type: ignore
			scatter_list=scatter_list,
			src=state.worker_rank,
			group=state.process_group
		)

		state.scatter_event = cast(torch.Event, torch.cuda.Event())
		state.scatter_event.record()
		scatter_list = None

def _update_param(
	p: DTensor,
	state: _THORNState,
	lr: float,
	update_scale: float,
	weight_decay: float,
	rank: int,
	compute_stream: cuda.Stream
):
	with torch.cuda.stream(compute_stream):
		if state.scatter_event is None:
			raise RuntimeError('Scatter event must be set before update')
		
		compute_stream.wait_event(state.scatter_event)
		assert state.scattered_u is not None
		u_dtensor = DTensor.from_local(state.scattered_u, placements=p.placements, device_mesh=p.device_mesh)
		state.scattered_u = u_dtensor
		if rank == state.worker_rank:
			state.computed_u = None

		u = cast(torch.Tensor, state.scattered_u).view_as(p)
		_apply_param_update(p, u, lr, weight_decay, scale=update_scale)

		state.scattered_u = None
		u_dtensor = None

@torch.no_grad()
def _update_adamw(
	params: list[torch.Tensor],
	grads: list[torch.Tensor],
	exp_avgs: list[torch.Tensor],
	exp_avg_sqs: list[torch.Tensor],
	state_steps: list[torch.Tensor],
	beta1: float,
	beta2: float,
	lr: float,
	weight_decay: float,
	eps: float,
	cautious: bool
):
	if not params:
		return

	grouped_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
		[
			params, grads, exp_avgs, exp_avg_sqs,
			state_steps
		] # type: ignore[list-item]
	)
	for (device, _), (
		(
			device_params_,
			device_grads_,
			device_exp_avgs_,
			device_exp_avg_sqs_,
			device_state_steps_
		),
		_
	) in grouped_tensors.items():
		device_params = cast(list[torch.Tensor], device_params_)
		device_grads = cast(list[torch.Tensor], device_grads_)
		device_exp_avgs = cast(list[torch.Tensor], device_exp_avgs_)
		device_exp_avg_sqs = cast(list[torch.Tensor], device_exp_avg_sqs_)
		device_state_steps = cast(list[torch.Tensor], device_state_steps_)

		torch._foreach_add_(device_state_steps, 1.)
		if weight_decay != 0:
			torch._foreach_mul_(device_params, 1. - lr * weight_decay)
		
		torch._foreach_lerp_(device_exp_avgs, device_grads, 1. - beta1)
		torch._foreach_mul_(device_exp_avg_sqs, beta2)
		torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1. - beta2)

		bias_correction1 = [1. - beta1 ** _get_value(step) for step in device_state_steps]
		bias_correction2 = [1. - beta2 ** _get_value(step) for step in device_state_steps]
		bias_correction2_sqrt = [b ** 0.5 for b in bias_correction2]

		denom = torch._foreach_sqrt(device_exp_avg_sqs)
		torch._foreach_div_(denom, bias_correction2_sqrt)
		torch._foreach_add_(denom, eps)

		adj_lr = [lr / b for b in bias_correction1]
		if cautious:
			mask = torch._foreach_mul(device_exp_avgs, device_grads)
			mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, device_exp_avgs)]
			mask_mean = [m.mean().clamp(min=1e-3) for m in mask]
			device_exp_avgs = torch._foreach_mul(device_exp_avgs, mask)
		M_div = torch._foreach_div(device_exp_avgs, denom)

		torch._foreach_mul_(M_div, adj_lr)
		if cautious:
			torch._foreach_div_(M_div, mask_mean) # type: ignore
		torch._foreach_sub_(device_params, M_div)

class _THORNParameterGroup(TypedDict):
	orthogonalize: bool
	none_grad: bool
	params: list[Parameter]
	eps: float
	weight_decay: float
	cautious: bool
	lr: float
	betas: tuple[float, float]
	iters: int
	nesterov: bool
	lower_bound: float
	safety_factor: float
	cushion: float
	coeffs: list[tuple[float, float, float]]

class THORNOrthogonalizedParameterGroup(TypedDict, total=True):
	orthogonalize: Literal[True]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	iters: NotRequired[int]
	nesterov: NotRequired[bool]
	eps: NotRequired[float]
	none_grad: NotRequired[bool]
	lower_bound: NotRequired[float]
	safety_factor: NotRequired[float]
	cushion: NotRequired[float]
	coeffs: NotRequired[list[tuple[float, float, float]]]

class THORNNonOrthogonalizedParameterGroup(TypedDict, total=True):
	orthogonalize: Literal[False]
	params: list[Parameter]
	lr: NotRequired[float]
	betas: NotRequired[tuple[float, float]]
	weight_decay: NotRequired[float]
	cautious: NotRequired[bool]
	eps: NotRequired[float]
	none_grad: NotRequired[bool]

THORNParameterGroup = Union[THORNOrthogonalizedParameterGroup, THORNNonOrthogonalizedParameterGroup]

class THORN(Optimizer):
	param_groups: list[_THORNParameterGroup] # type: ignore
	is_distributed: bool
	rank: int | None = None
	comm_stream = cuda.Stream()
	compute_stream = cuda.Stream()

	def __init__(self, param_groups: list[THORNParameterGroup]):
		for group in param_groups:
			assert 'orthogonalize' in group
			group.setdefault('eps', 1e-7)
			group.setdefault('weight_decay', 0.01)
			group.setdefault('none_grad', True)
			if group['orthogonalize']:
				group = cast(THORNOrthogonalizedParameterGroup, group)
				group.setdefault('lr', 0.02)
				group.setdefault('betas', (0.95, 0.95))
				group.setdefault('iters', 5)
				group.setdefault('nesterov', True)
				group.setdefault('lower_bound', 1e-3)
				group.setdefault('safety_factor', 0.02)
				group.setdefault('cushion', 0.02)
				if not 'coeffs' in group:
					group = cast(dict, group)
					group['coeffs'] = _optimal_composition(
						l=group['lower_bound'],
						num_iters=group['iters'],
						safety_factor_eps=group['safety_factor'],
						cushion=group['cushion']
					)
			else:
				group = cast(THORNNonOrthogonalizedParameterGroup, group)
				group.setdefault('lr', 3e-4)
				group.setdefault('betas', (0.95, 0.98))
				group.setdefault('cautious', True)

		self.is_distributed = dist.is_initialized()
		if self.is_distributed:
			self.rank = dist.get_rank()

		super().__init__(cast(list[dict], param_groups), dict())

		for group in self.param_groups:
			for p in group['params']:
				self.state[p]['group'] = group

	def _calc_flops(self, G: torch.Tensor, steps: int):
		M, N = G.size(-2), G.size(-1)
		if M > N:
			M, N = N, M

		return steps * ((M ** 3) * 2 + (M ** 2 * N) * 4 + M * N * 2 + M ** 2 * 3)
	
	def _get_shard_mesh(self, p: DTensor, rank: int) -> tuple[torch.Tensor, dist.ProcessGroup]:
		assert isinstance(p, DTensor)

		if p.placements == (Shard(dim=0),):
			# Case for FSDP
			return p.device_mesh.mesh, p.device_mesh.get_group(mesh_dim=0)
		elif p.placements == (Replicate(), Shard(dim=0)):
			# Case for HSDP
			for shard_mesh in p.device_mesh.mesh:
				if rank in shard_mesh:
					return shard_mesh, p.device_mesh.get_group(mesh_dim=1)
			raise ValueError('shouldn\'t happen')
		else:
			raise ValueError(f'Unsupported placements {p.placements}')
		
	def _assign_params(self, params, group: _THORNParameterGroup):
		assert self.rank is not None

		param_to_state = {}
		param_to_flops = {}

		total_flops = 0
		for p in params:
			g = p.grad
			if g is None:
				continue

			g = _resize(g)

			flops = self._calc_flops(g, group['iters'])
			param_to_flops[id(p)] = flops
			total_flops += flops

		ordered_params = sorted(params, key=lambda p: param_to_flops[id(p)], reverse=True)

		round_robin = 0
		mesh = None
		shard_mesh = None
		process_group = None
		for p in ordered_params:
			if mesh is None:
				mesh = p.device_mesh
				shard_mesh, process_group = self._get_shard_mesh(p, self.rank)
			elif mesh != p.device_mesh:
				raise ValueError('All parameters must be on the same mesh.')

			assert shard_mesh is not None and process_group is not None

			param_to_state[id(p)] = _THORNState(
				worker_rank=int(shard_mesh[round_robin].item()),
				process_group=process_group
			)

			round_robin = (round_robin + 1) % len(shard_mesh)

		return param_to_state, ordered_params

	@torch.no_grad()
	def _update_momentum(self, p: Parameter, g: torch.Tensor, group: _THORNParameterGroup):
		state = self.state[p]
		momentum, beta1, _ = state['moment'], *group['betas']
		momentum.mul_(beta1).add_(g)
		update = g.add(momentum, alpha=beta1) if group['nesterov'] else momentum
		return update

	def _lr_scale_ortho(self, p: Parameter):
		# Match RMS update of AdamW, per formula 4: https://arxiv.org/pdf/2502.16982
		return 5. / (max(p.shape[:2]) ** 0.5)

	def _base_ortho_step(self, p: Parameter, group: _THORNParameterGroup):
		assert p.grad is not None
		state = self.state[p]
		g = _resize(p.grad)

		u = self._update_momentum(p, g, group)
		u = _polar_decomp(
			u.float(),
			coeffs=group['coeffs'],
			steps=group['iters'],
			eps=group['eps'],
			safety_factor_eps=group['safety_factor']
		).to(dtype=p.dtype)
		if group['betas'][1] > 0.0:
			u = _apply_per_neuron_norm(u, state['variance'], beta2=group['betas'][1], eps=group['eps'])
		
		scale = self._lr_scale_ortho(p)
		_apply_param_update(p, u.view_as(p), group['lr'], group['weight_decay'], scale=scale)

	def _sharded_ortho_step(self, params: list[torch.nn.Parameter], group: _THORNParameterGroup):
		for p in params:
			g = p.grad
			if g is None:
				continue

			g = _resize(g)
			g = self._update_momentum(p, g, group)
			p.grad = g.view_as(p)

		param_to_state, ordered_params = self._assign_params(params, group)

		def enqueue_gathers(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				_gather(p, state, self.rank, self.comm_stream, none_grad=group['none_grad'])

		def enqueue_computes(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				_compute_u(
					p,
					self.state[p]['variance'],
					state,
					rank=self.rank,
					compute_stream=self.compute_stream,
					coeffs=group['coeffs'],
					steps=group['iters'],
					beta2=group['betas'][1],
					eps=group['eps'],
					safety_factor_eps=group['safety_factor']
				)

		def enqueue_scatters(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				_scatter(p, state, self.rank, self.comm_stream)

		def enqueue_updates(start_idx: int, chunk_size: int):
			assert self.rank is not None
			for p in ordered_params[start_idx:start_idx + chunk_size]:
				state = param_to_state[id(p)]
				lr_scale = self._lr_scale_ortho(p)
				_update_param(
					p,
					state,
					lr=group['lr'],
					update_scale=lr_scale,
					weight_decay=group['weight_decay'],
					rank=self.rank,
					compute_stream=self.compute_stream
				)

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
		ortho = group['orthogonalize']

		distributed_params = []
		regular_params = []
		for p in params:
			g = p.grad
			if p is None or g is None:
				continue

			state = self.state[p]
			if ortho:
				g = _resize(g)
				if 'moment' not in state:
					state['moment'] = torch.zeros_like(g)
					state['variance'] = torch.zeros((g.shape[0], 1), dtype=g.dtype, device=g.device)
			else:
				if 'step' not in state:
					state['step'] = torch.zeros((), dtype=torch.float32, device=p.device)
					state['moment'] = torch.zeros_like(g)
					state['variance'] = torch.zeros_like(g)

			if isinstance(p.data, DTensor):
				if all(isinstance(placement, Replicate) for placement in cast(DTensor, p).placements) or not ortho:
					regular_params.append(p)
				else:
					distributed_params.append(p)
			else:
				regular_params.append(p)

		if len(distributed_params) > 0:
			self._sharded_ortho_step(distributed_params, group)

		if ortho:
			for p in regular_params:
				self._base_ortho_step(p, group)
		else:
			grads = []
			moments = []
			variances = []
			steps = []
			for p in regular_params:
				state = self.state[p]
				g = p.grad
				grads.append(g)
				moments.append(state['moment'])
				variances.append(state['variance'])
				steps.append(state['step'])
			_update_adamw(
				regular_params,
				grads,
				moments,
				variances,
				steps,
				beta1=group['betas'][0],
				beta2=group['betas'][1],
				lr=group['lr'],
				weight_decay=group['weight_decay'],
				eps=group['eps'],
				cautious=group['cautious']
			)

	@overload
	def step(self, *, param: Optional[torch.nn.Parameter] = None, closure: None = None) -> None: ...
	@overload
	def step(self, *, param: Optional[torch.nn.Parameter] = None, closure: Callable[[], float]) -> float: ...
	@torch.no_grad()
	def step(self, *, param: Optional[torch.nn.Parameter] = None, closure: Optional[Callable[[], float]] = None) -> Optional[float]: # type: ignore
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		if param is None:
			for group in self.param_groups:
				params = group['params']
				self._step_params(params, group)
		else:
			state = self.state[param]
			group = state['group']
			self._step_params([param], group)

		return loss

def thorn_gradient_release(model: torch.nn.Module, optimizer: THORN, ignore_existing_hooks: bool = False):
	def _gradient_release_hook(param: torch.Tensor, optimizer: THORN):
		optimizer.step(param=cast(torch.nn.Parameter, param))
		param.grad = None

	hooks = []
	for p in model.parameters():
		if p.requires_grad:
			if p._post_accumulate_grad_hooks is not None and len(p._post_accumulate_grad_hooks) > 0 and not ignore_existing_hooks:
				for hook in hooks:
					if hasattr(hook, 'remove'):
						hook.remove()
				raise ValueError('Model already has post_accumulate_grad_hooks. If this is expected, pass `ignore_existing_hooks=True`.')
			hooks.append(p.register_post_accumulate_grad_hook(partial(_gradient_release_hook, optimizer=optimizer)))
	model._gradient_release_hooks = hooks # type: ignore
