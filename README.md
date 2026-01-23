# THORN 🌹
THORN is an optimizer for PyTorch.

It's often a little better than [Muon](https://kellerjordan.github.io/posts/muon/) & supports FSDP.

## Usage
Requires PyTorch >= 2.6. [Triton](https://triton-lang.org/main/index.html) is optional but provides a decent speed boost.

```python
optimizer = THORN([
	{
		'orthogonalize': True,
		# Enable `orthogonalize` for matrix parameters.
		'params': [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad],
		'lr': 0.001,
		'betas': (0.95, 0.95), # First- and second-order momentum betas
		'none_grad': True, # Automatically performs `zero_grad(set_to_none=True)` after each step.
		'nesterov': True,  # Best not to touch
		'iters': 5,        # Best not to touch
		'lower_bound': 1e-3, # Lower bound of singular value, probably don't touch
		'safety_factor': 0.02, # For numerical stability when computing coefficients
		'cushion': 0.02
	},
	{
		'orthogonalize': False,
		# Use non-`orthogonalize` group (AdamW) for everything else (bias/norm parameters)
		'params': [p for p in model.parameters() if p.ndim < 2 and p.requires_grad],
		'lr': 0.001,
		'betas': (0.95, 0.98),
		'weight_decay': 0.01,
		'cautious': True, # Cautious AdamW; results are a tiny bit better
		'none_grad': True # Automatically performs `zero_grad(set_to_none=True)` after each step.
	}
])
```

### Gradient release mode
For memory savings, you can limit gradients to one layer at a time with `thorn_gradient_release`. This slows down FSDP significantly, so it is only recommended on single-GPU setups.

Note that gradient release is not compatible with gradient accumulation or float16 mixed precision (but bfloat16 works). You might be able to get most of the same effect of gradient accumulation by tuning the LR & betas.

```python
from thorn import THORN, thorn_gradient_release

model = MyModel().to(dtype=torch.bfloat16)

optimizer = THORN(...)
# enable gradient release mode
thorn_gradient_release(model, optimizer)

scheduler = CosineAnnealingLR(optimizer, ...) # optional

for item in dataset:
	loss = model(item)
	loss.backward() # <- optimization is done here...

	# ...so do not manually step THORN when gradient release is used!
	#optimizer.step()
	#optimizer.zero_grad()

	# but do step the scheduler if you're using one
	scheduler.step()
```

## Based on
- Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). [*Muon: An optimizer for hidden layers in neural networks.*](https://kellerjordan.github.io/posts/muon/)
- Lim, J., Lee, S., Kim, D., Kim, T., Park, E., Lee, J., … Weon, D. (2025). [*Motif 2 12.7B technical report.*](http://arxiv.org/abs/2511.07464)
- Li, Z., Liu, L., Liang, C., Chen, W., & Zhao, T. (2025). [*NorMuon: Making Muon more efficient and scalable.*](http://arxiv.org/abs/2510.05491)
- Delattre, B., Barthélemy, Q., Araujo, A., & Allauzen, A. (2023). [*Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration.*](http://arxiv.org/abs/2305.16173)
- Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., … Yang, Z. (2025). [*Muon is Scalable for LLM Training.*](http://arxiv.org/abs/2502.16982)
- Liang, K., Chen, L., Liu, B., & Liu, Q. (2025). [*Cautious Optimizers: Improving Training with One Line of Code.*](http://arxiv.org/abs/2411.16085)
- Pudipeddi, B., Mesmakhosroshahi, M., Xi, J., & Bharadwaj, S. (2020). [*Training Large Neural Networks with Constant Memory using a New Execution Algorithm.*](http://arxiv.org/abs/2002.05645)
- Chen, L., Li, J., Liang, K., Su, B., Xie, C., Pierse, N. W., … Liu, Q. (2025). [*Cautious Weight Decay.*](http://arxiv.org/abs/2510.12402)
- Amsel, N., Persson, D., Musco, C., & Gower, R. M. (2025). [*The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm.*](http://arxiv.org/abs/2505.16932)
- [Flash-Muon](https://github.com/nil0x9/flash-muon) by Tianyang Lin
- [optimī](https://github.com/warner-benjamin/optimi) by Benjamin Warner
