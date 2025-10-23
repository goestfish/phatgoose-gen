import os

import gin

#add
import torch
#add
@gin.configurable(allowlist=["model_name_or_path", "model_class", "from_pretrained_kwargs"])
def hf_torch_model(model_name_or_path, model_class="", from_pretrained_kwargs=None):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    model_class = os.path.expandvars(model_class)

    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    from_pretrained_kwargs = dict(from_pretrained_kwargs or {})

    if "torch_dtype" in from_pretrained_kwargs and isinstance(from_pretrained_kwargs["torch_dtype"], str):
        s = from_pretrained_kwargs["torch_dtype"].lower()
        if s in ("bf16", "bfloat16"):   from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        elif s in ("fp16", "float16"):  from_pretrained_kwargs["torch_dtype"] = torch.float16
        elif s in ("fp32", "float32"):  from_pretrained_kwargs["torch_dtype"] = torch.float32

    cls_map = {
        "": AutoModel,
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "seq_cls": AutoModelForSequenceClassification,
        "token_cls": AutoModelForTokenClassification,
        "qa": AutoModelForQuestionAnswering,
    }
    model_ctor = cls_map[model_class]
    torch_model = model_ctor.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
    return torch_model


@gin.configurable(allowlist=["model_name_or_path"])
def hf_tokenizer(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if model_name_or_path.startswith("EleutherAI/pythia"):
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    assert tokenizer.pad_token_id is not None

    test_tokens = tokenizer.build_inputs_with_special_tokens([-100])
    if test_tokens[0] != -100:
        tokenizer.bos_token_id = test_tokens[0]
    else:
        tokenizer.bos_token_id = None
    if test_tokens[-1] != -100:
        tokenizer.eos_token_id = test_tokens[-1]
    else:
        tokenizer.eos_token_id = None

    return tokenizer
