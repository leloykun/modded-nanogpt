# 11/26/25 - Misc Tweaks

Changelogs:

1. Reduced training per-device sequence length from `64*1024` to `48*1024`. See [Critical Batch Size](https://arxiv.org/abs/2410.21676) literature.
2. Increased eval per-device sequence length from `64*1024` to `4*64*1024` (decreases `val_loss` by `~0.0015`)
3. Modified scales for `fp8` training of LM Head. Saves `1 sec` and improves `val_loss` by as much as `~0.01` after reducing training sequence length down to `48*1024`. I don't know wtf is causing this and I'm NOT going crazy about this. I have evidence. See `records/012625_MiscTweaks/no-autocast-same-fp8-scales`.
    - `w_s = 2.0**9` (from `2.0**5`)
    - `grad_s = 2.0**19` (from `2.0**29`)
4. Upgrade PyTorch to 2.7.0 nightly version (20250125) for CUDA 12.6
  - `pip install --pre torch==2.7.0.dev20250125+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126`

![](val_losses.png)
![](wallclock.png)

![](ablations.png)

```python
accs = [3.2806, 3.2771, 3.2829, 3.2813, 3.2789, 3.2774, 3.2798, 3.2759, 3.2794, 3.2775, 3.2768, 3.2793, 3.2838, 3.2779, 3.2782, 3.277, 3.2775, 3.2784, 3.2782, 3.2776, 3.2814, 3.2785, 3.2793, 3.2797, 3.2782, 3.2789, 3.2759, 3.2803, 3.278, 3.2782]

import scipy.stats
print('p=0.00' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0007 (statistically significant)

import torch
print(torch.std_mean(torch.tensor(accs)))
# (tensor(0.0019), tensor(3.2788))
```
