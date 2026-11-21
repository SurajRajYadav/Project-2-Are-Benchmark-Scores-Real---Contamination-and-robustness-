"""Thin wrapper around a Hugging Face causal LM with the three operations we need:
  1. choice_logprobs: next-token log-probs over the letters A/B/C/D  (MMLU scoring, no generation)
  2. token_logprobs : per-token log-probs of a text                  (contamination signals)
  3. generate       : greedy generation                              (GSM8K)
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LM:
    def __init__(self, model_name, load_4bit=False):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16   # newer transformers may call this `dtype`
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        # the token the model would emit for " A", " B", ... right after "Answer:"
        self.letter_ids = [self.tok.encode("Answer: " + L, add_special_tokens=False)[-1] for L in "ABCD"]
        assert len(set(self.letter_ids)) == 4, "letter tokens are not distinct for this tokenizer"

    def _encode(self, texts, side):
        self.tok.padding_side = side
        return self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)

    @torch.no_grad()
    def choice_logprobs(self, prompts):
        """Log-softmax over the 4 letter tokens at the end of each prompt -> list of 4-lists."""
        enc = self._encode(prompts, "left")
        try:
            out = self.model(**enc, logits_to_keep=1)      # saves memory on recent versions
        except TypeError:
            out = self.model(**enc)
        last = out.logits[:, -1, :].float()
        return torch.log_softmax(last[:, self.letter_ids], dim=-1).cpu().tolist()

    @torch.no_grad()
    def token_logprobs(self, texts):
        """Per-token log-probs (token t given tokens < t) for each text; first token has none."""
        enc = self._encode(texts, "right")
        logits = self.model(**enc).logits
        results = []
        for i in range(len(texts)):
            n = int(enc.attention_mask[i].sum())
            lp = torch.log_softmax(logits[i, : n - 1].float(), dim=-1)
            tgt = enc.input_ids[i, 1:n]
            results.append(lp.gather(-1, tgt[:, None]).squeeze(-1).cpu().tolist())
        return results

    def chat(self, user_text):
        """Use the chat template when the model has one (instruct models), else plain text."""
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
        return user_text + "\n"

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=320):
        enc = self._encode(prompts, "left")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


def min_k_prob(token_lps, k=0.2):
    """Min-K% Prob (Shi et al., 2023): mean of the lowest k fraction of token log-probs.
    Texts seen in training tend to have fewer very-unlikely tokens -> HIGHER (less negative) score."""
    if not token_lps:
        return 0.0
    m = max(1, int(len(token_lps) * k))
    return float(np.mean(sorted(token_lps)[:m]))
