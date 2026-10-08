"""LLM backends: local Hugging Face transformers, HF Inference API, or 'extractive' (no LLM)."""
import os


class LLMClient:
    def __init__(self, backend="transformers", model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=400,
                 temperature=0.0, hf_api_model=None, load_in_4bit=False):
        self.backend, self.model_name = backend, model_name
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.hf_api_model, self.load_in_4bit = hf_api_model or model_name, load_in_4bit
        self._tok = self._model = self._client = None

    @classmethod
    def from_config(cls, cfg):
        l = cfg["llm"]
        return cls(l["backend"], l["model_name"], l["max_new_tokens"], l["temperature"],
                   l.get("hf_api_model"), l.get("load_in_4bit", False))

    def _load_local(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kw["torch_dtype"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "transformers":
            if self._model is None:
                self._load_local()
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tok(text, return_tensors="pt").to(self._model.device)
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=self.temperature > 0,
                                       temperature=self.temperature if self.temperature > 0 else None,
                                       pad_token_id=self._tok.eos_token_id)
            return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if self.backend == "hf_api":
            if self._client is None:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(model=self.hf_api_model, token=os.getenv("HF_TOKEN"))
            r = self._client.chat_completion(messages=messages, max_tokens=self.max_new_tokens,
                                             temperature=max(self.temperature, 0.01))
            return r.choices[0].message.content.strip()
        if self.backend == "extractive":
            return "[No generative LLM configured: backend is 'extractive'.]"
        raise ValueError(f"Unknown LLM backend: {self.backend}")
