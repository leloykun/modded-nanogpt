# UNet-pattern Attention Scales

This record, by [Franz Cesista](@leloykun), makes the attention scale follow the UNet pattern instead of just being fixed at $1/\sqrt{d}$.

At first, we made the attention scale learnable. It worked well, reducing training steps by 20. However, the extra overhead made it not so worth it. We then observed that the learned attention scales generally follow a UNet pattern somewhat consistently. So, we decided to hardcode that pattern instead. Overall, this change reduced the training steps by 15 and wallclock time by ~2.4 secs (on my machine).

![](unet_attn_scales_pattern_val_losses.png)
![](unet_attn_scales_pattern_wallclock.png)

---

Diff:

```diff
# For the layer_id-th layer
# In CausalSelfAttention.__init__
+ self.attn_scale = 0.13 + 0.01 * min(layer_id, 11 - layer_id)
...
# In CausalSelfAttention.forward
- y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
+ y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale)
```

---

```python
val_losses = [3.2765, 3.2802, 3.2775, 3.2783, 3.2793, 3.279]

import scipy.stats
print('p=%.4f' % scipy.stats.ttest_1samp(val_losses, 3.28, alternative='less').pvalue)
# p=0.0184
```

---

Wallclock time improvement: ~2 secs on Franz' machine

Record | Wallclock time (ms)
--- | ---
Jan 4, 2025 record runtime on Franz' machine | 202898 ms
This record on Franz' machine | 200871 ms

---

Having learnable attention scales was originally recommended by the [OG paper on QK-Normalization](https://arxiv.org/abs/2010.04245) and more recently by the [Cosmos team at NVidia](https://arxiv.org/abs/2501.03575v1).

Also thanks to @YouJiacheng and @Grad62304977 for the discussions on this topic.
