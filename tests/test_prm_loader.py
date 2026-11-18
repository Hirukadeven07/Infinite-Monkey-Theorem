"""The PRM must load with transformers' own Qwen2 code, not the checkpoint's remote code (which
breaks on transformers 5.x: 'Qwen2RMConfig' object has no attribute 'pad_token_id')."""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")


def make_tiny_prm_checkpoint(path):
    """A miniature checkpoint laid out like Qwen/Qwen2.5-Math-PRM-7B: model.*, lm_head, score.0/score.2,
    and a config.json that points at remote code (which must NOT be used)."""
    from transformers import Qwen2Config, Qwen2Model

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    backbone = Qwen2Model(cfg).eval()
    head = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)).eval()
    tensors = {f"model.{k}": v.contiguous() for k, v in backbone.state_dict().items()}
    tensors.update({f"score.{k}": v.contiguous() for k, v in head.state_dict().items()})
    tensors["lm_head.weight"] = torch.randn(64, 32)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    cfg_json = cfg.to_dict()
    cfg_json.update({"architectures": ["Qwen2ForProcessRewardModel"],
                     "auto_map": {"AutoConfig": "configuration_qwen2_rm.Qwen2RMConfig",
                                  "AutoModel": "modeling_qwen2_rm.Qwen2ForProcessRewardModel"}})
    cfg_json.pop("pad_token_id", None)  # like the real config: no pad_token_id
    (path / "config.json").write_text(json.dumps(cfg_json))
    return backbone, head


def test_native_loader_matches_reference(tmp_path):
    from colab.gateway import load_prm_model

    backbone, head = make_tiny_prm_checkpoint(tmp_path)
    prm_model = load_prm_model(str(tmp_path), quant="fp32")
    ids = torch.tensor([[1, 5, 9, 7, 3], [2, 4, 6, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.no_grad():
        got = prm_model(input_ids=ids, attention_mask=attn)[0]
        want = head(backbone(input_ids=ids, attention_mask=attn).last_hidden_state)
    assert got.shape == (2, 5, 2)
    assert torch.allclose(got[0], want[0], atol=1e-5)          # full row
    assert torch.allclose(got[1, :3], want[1, :3], atol=1e-5)  # real tokens of the padded row


def test_loader_never_trusts_remote_code(tmp_path, monkeypatch):
    import transformers as tf
    from colab.gateway import load_prm_model

    make_tiny_prm_checkpoint(tmp_path)
    calls = []
    real = tf.AutoModel.from_pretrained
    monkeypatch.setattr(tf.AutoModel, "from_pretrained", lambda *a, **k: calls.append(k) or real(*a, **k))
    load_prm_model(str(tmp_path), quant="fp32")
    assert not any(k.get("trust_remote_code") for k in calls)
