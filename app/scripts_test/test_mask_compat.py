"""Regression: writing lyrics AFTER a song must work.

Loading YuE2 patches transformers.masking_utils.create_causal_mask so that
transformers-5-style callers work. The patch used to rename the argument
blindly, which broke every model loaded afterwards on transformers 4.x (the
lyrics writer): "create_causal_mask() got an unexpected keyword argument
'inputs_embeds'". No GPU needed.

    python scripts_test/test_mask_compat.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "shims"), os.path.join(ROOT, "engine"), ROOT):
    sys.path.insert(0, p)

import bootstrap  # noqa: E402

bootstrap.bootstrap()
import torch  # noqa: E402
from transformers import PretrainedConfig, masking_utils  # noqa: E402
from yue2_repo.compat import transformers5  # noqa: E402

transformers5.patch_mask_utils()          # what loading the song model does
embeds = torch.zeros(1, 4, 8)
common = dict(config=PretrainedConfig(), attention_mask=None,
              cache_position=torch.arange(4), past_key_values=None)
failed = []
for name in ("input_embeds", "inputs_embeds"):      # 4.x spelling, 5.x spelling
    try:
        masking_utils.create_causal_mask(**common, **{name: embeds})
        print(f"ok    {name}")
    except TypeError as exc:
        failed.append(name)
        print(f"FAIL  {name}: {exc}")
sys.exit(1 if failed else 0)
