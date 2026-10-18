"""
Test GPT model accounting. Example run:

python -m pytest tests/test_gpt.py -v
"""

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.fp8 import convert_to_float8_training


def test_estimate_flops_unchanged_by_fp8_conversion():
    # base_train.py converts to Float8Linear *before* estimating FLOPs (MFU, --target-flops),
    # so the matmul param count must not depend on which Linear class does the matmul
    config = GPTConfig(sequence_len=256, vocab_size=32768, n_layer=4, n_head=4, n_kv_head=4, n_embd=512)
    with torch.device("meta"):
        model = GPT(config)
    flops_before = model.estimate_flops()
    matmul_params_before = model.num_matmul_params()
    convert_to_float8_training(model, module_filter_fn=lambda mod, fqn: min(mod.in_features, mod.out_features) >= 128)
    assert any("Float8" in type(m).__name__ for m in model.modules())
    assert model.num_matmul_params() == matmul_params_before
    assert model.estimate_flops() == flops_before
