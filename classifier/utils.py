"""Shared helpers for the latent-probe training scripts.

Deduplicates the model registry, dataset loading, device selection, and the
per-layer LinearSVC procedure used by classifier/train_latent.py and
classifier/train_latent_generated.py. Figure-specific helpers (cosine matrices,
plotting) live in the analysis/ package (analysis/_shared.py and the plot_*.py).
"""

import json

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from models import (
    DeepSeekR1_8b, Gemma3_12b, GPT20b, Llama2_7b, Llama3_8b, Llama32_3b,
    Ministral3_14b, Mistral7b_Instruct, Mistral7B_RR, Mixtral8x7b_Instruct,
    Olmo3_7b, Phi4_15b, Phi35Mini, Qwen2_32b, Qwen35_9b,
)


MODEL_REGISTRY = {
    "llama2-7b": Llama2_7b, "llama3-8b": Llama3_8b,
    "qwen2-32b": Qwen2_32b,
    "mistral-7b-rr": Mistral7B_RR,
    "gemma3-12b": Gemma3_12b, "phi35mini": Phi35Mini, "mixtral-8x7b": Mixtral8x7b_Instruct,
    "llama32-3b": Llama32_3b, "mistral-7bv3": Mistral7b_Instruct, "ministral3-14b": Ministral3_14b,
    "gpt-oss-20b": GPT20b, "olmo3-7b": Olmo3_7b, "phi4-15b": Phi4_15b,
    "qwen35-9b": Qwen35_9b, "deepseek-r1-8b": DeepSeekR1_8b,
}


def pick_device(requested: str) -> str:
    """Resolve 'auto' to the best available device; pass any explicit device through."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(model_name: str, device: str):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_name '{model_name}'. Options: {sorted(MODEL_REGISTRY)}")
    print(f"Loading model: {model_name} on device: {device}")
    return MODEL_REGISTRY[model_name](device=device)


def load_arditi(model_name: str):
    """Load the per-model harmful/harmless instruction splits (labels 1 / 0)."""
    harmful_path = f"./dataset/splits/{model_name}/harmful_train_filtered.json"
    harmless_path = f"./dataset/splits/{model_name}/harmless_train_filtered.json"
    print("Loading Arditi dataset...")
    with open(harmful_path, "r", encoding="utf-8") as f:
        harmful_prompts = json.load(f)
    with open(harmless_path, "r", encoding="utf-8") as f:
        harmless_prompts = json.load(f)
    return harmful_prompts, harmless_prompts


def train_layer_svm(X_layer, Y, seed: int = 42, test_size: float = 0.1):
    """Fit the standard per-layer LinearSVC and score it on a held-out split.

    Returns (clf, acc, report), mirroring the procedure in
    classifier/train_latent.py.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_layer, Y, test_size=test_size, random_state=seed, stratify=Y
    )
    clf = LinearSVC(C=0.1, dual="auto", max_iter=1000000, random_state=seed)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    report = classification_report(
        y_test, clf.predict(X_test), target_names=["Harmless", "Harmful"], output_dict=False
    )
    return clf, acc, report


def generate_and_extract(model, prompt: str, n_gen_tokens: int):
    """Greedily generate `n_gen_tokens`, then extract hidden states at each
    generated token position for all transformer layers.

    Returns (reps, num_gen, gen_ids, gen_text) where reps has shape
    (num_gen, num_layers, dim), matching the layer ordering of
    LanguageModel.get_representations (hidden_states[1:], embedding skipped).
    """
    formatted = model._get_prompt(prompt)
    enc = model.tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        full_ids = model.model.generate(
            **enc,
            max_new_tokens=n_gen_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.pad_token_id,
        )

    num_gen = int(full_ids.shape[1] - prompt_len)
    gen_ids = full_ids[0, prompt_len:].tolist()
    gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if num_gen == 0:
        return None, 0, gen_ids, gen_text

    with torch.no_grad():
        outputs = model.model(input_ids=full_ids, output_hidden_states=True)

    sl = slice(prompt_len, prompt_len + num_gen)
    # each h[:, sl, :] -> (1, num_gen, dim); stack over layers -> (num_layers, num_gen, dim)
    layers = torch.cat([h[:, sl, :] for h in outputs.hidden_states[1:]], dim=0)
    reps = layers.permute(1, 0, 2).to(dtype=torch.float32).cpu()  # (num_gen, num_layers, dim)
    return reps, num_gen, gen_ids, gen_text
