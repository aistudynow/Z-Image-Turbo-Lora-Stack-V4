import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch  # type: ignore

import comfy.sd  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore

logger = logging.getLogger(__name__)

_SUFFIX_KIND = {
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
    ".lora_down.weight": "A",
    ".lora_up.weight": "B",
    ".alpha": "ALPHA",
}


@dataclass
class ComponentEntry:
    tensor: torch.Tensor
    suffix: str


@dataclass
class MergeGroup:
    components: Dict[str, Dict[str, ComponentEntry]]
    original_keys: Dict[Tuple[str, str], str]

    def __init__(self) -> None:
        self.components = {}
        self.original_keys = {}


def _split_lora_key(key: str) -> Optional[Tuple[str, str, str]]:
    for suffix, kind in _SUFFIX_KIND.items():
        if key.endswith(suffix):
            return key[:-len(suffix)], suffix, kind  # type: ignore
    return None


def _replace_to_qkv_token(base: str) -> str:
    replaced = base
    replaced = replaced.replace(".attention.to_qkv", ".attention.qkv")
    replaced = replaced.replace("_attention_to_qkv", "_attention_qkv")
    replaced = re.sub(r"([._])to([._])qkv(?=$|[._])", r"\1qkv", replaced)
    return replaced


def _maybe_group_qkv(base: str) -> Optional[Tuple[str, str]]:
    patterns = [
        (r"(.*(?:[._]attention[._]))to[._](q|k|v)$", r"\1qkv"),
        (r"(.*(?:[._]attn[._]))to[._](q|k|v)$", r"\1qkv"),
    ]
    for pattern, replacement in patterns:
        match = re.match(pattern, base)
        if match:
            component = match.group(2)
            merged = re.sub(pattern, replacement, base)
            return component, merged
    return None


def _maybe_group_w13(base: str) -> Optional[Tuple[str, str]]:
    patterns = [
        (r"(.*(?:[._]feed_forward[._]))(w1|w3)$", r"\1w13"),
        (r"(.*(?:[._]ff[._]))(w1|w3)$", r"\1w13"),
    ]
    for pattern, replacement in patterns:
        match = re.match(pattern, base)
        if match:
            component = match.group(2)
            merged = re.sub(pattern, replacement, base)
            return component, merged
    return None


def _block_diag_merge(
    ordered_components: Tuple[str, ...],
    group: Dict[str, Dict[str, ComponentEntry]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, str, str]]:
    try:
        a_entries = [group[name]["A"] for name in ordered_components]
        b_entries = [group[name]["B"] for name in ordered_components]
    except KeyError:
        return None

    if not all(isinstance(entry.tensor, torch.Tensor) and entry.tensor.ndim == 2 for entry in a_entries + b_entries):
        return None

    in_features = a_entries[0].tensor.shape[1]
    if any(entry.tensor.shape[1] != in_features for entry in a_entries):
        return None

    device = a_entries[0].tensor.device
    dtype = a_entries[0].tensor.dtype

    merged_a = torch.cat([entry.tensor.to(device=device, dtype=dtype) for entry in a_entries], dim=0)
    merged_b = torch.block_diag(*[entry.tensor.to(device=device, dtype=dtype) for entry in b_entries])
    return merged_a, merged_b, a_entries[0].suffix, b_entries[0].suffix


def _set_alpha(translated: Dict[str, Any], merged_base: str, merged_a: torch.Tensor, suffix: str) -> None:
    if suffix.endswith(".weight"):
        translated[f"{merged_base}.alpha"] = int(merged_a.shape[0])


def _extract_transformer(model: Any) -> Any:
    try:
        return model.model.diffusion_model
    except Exception:
        return None


def _has_module_name(transformer: Any, needle: str) -> bool:
    if transformer is None:
        return False

    try:
        for name, _ in transformer.named_modules():
            if needle in name:
                return True
    except Exception:
        return False

    return False


def _translate_lora_state_dict(lora_state: Dict[str, Any], transformer: Any) -> Dict[str, Any]:
    model_has_qkv = _has_module_name(transformer, "qkv")
    model_has_w13 = _has_module_name(transformer, "w13")

    translated: Dict[str, Any] = {}
    qkv_groups: Dict[str, MergeGroup] = {}
    w13_groups: Dict[str, MergeGroup] = {}

    for key, value in lora_state.items():
        if not isinstance(key, str):
            translated[key] = value
            continue

        split = _split_lora_key(key)
        if split is None:
            translated[_replace_to_qkv_token(key)] = value
            continue

        base, suffix, kind = split

        direct_base = _replace_to_qkv_token(base)
        direct_key = f"{direct_base}{suffix}"

        qkv_grouped = _maybe_group_qkv(base) if model_has_qkv else None
        if qkv_grouped is not None:
            component, merged_base = qkv_grouped
            group = qkv_groups.setdefault(merged_base, MergeGroup())
            component_dict = group.components.setdefault(component, {})
            if isinstance(value, torch.Tensor) and kind in ("A", "B"):
                component_dict[kind] = ComponentEntry(value, suffix)
                group.original_keys[(component, kind)] = key
                continue

        w13_grouped = _maybe_group_w13(base) if model_has_w13 else None
        if w13_grouped is not None:
            component, merged_base = w13_grouped
            group = w13_groups.setdefault(merged_base, MergeGroup())
            component_dict = group.components.setdefault(component, {})
            if isinstance(value, torch.Tensor) and kind in ("A", "B"):
                component_dict[kind] = ComponentEntry(value, suffix)
                group.original_keys[(component, kind)] = key
                continue

        translated[direct_key] = value

    for merged_base, group in qkv_groups.items():
        merged = _block_diag_merge(("q", "k", "v"), group.components)
        if merged is None:
            for component in ("q", "k", "v"):
                for kind in ("A", "B"):
                    original_key = group.original_keys.get((component, kind))
                    if original_key is not None:
                        translated[original_key] = lora_state[original_key]  # type: ignore
            continue

        merged_a, merged_b, a_suffix, b_suffix = merged
        translated[f"{merged_base}{a_suffix}"] = merged_a
        translated[f"{merged_base}{b_suffix}"] = merged_b
        _set_alpha(translated, merged_base, merged_a, a_suffix)

    for merged_base, group in w13_groups.items():
        merged = _block_diag_merge(("w1", "w3"), group.components)
        if merged is None:
            for component in ("w1", "w3"):
                for kind in ("A", "B"):
                    original_key = group.original_keys.get((component, kind))
                    if original_key is not None:
                        translated[original_key] = lora_state[original_key]  # type: ignore
            continue

        merged_a, merged_b, a_suffix, b_suffix = merged
        translated[f"{merged_base}{a_suffix}"] = merged_a
        translated[f"{merged_base}{b_suffix}"] = merged_b
        _set_alpha(translated, merged_base, merged_a, a_suffix)

    return translated


class ZImageTurboLoraStackV4:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            "toggle_all": ("BOOLEAN", {"default": True}),
        }

        optional: Dict[str, Any] = {}
        for i in range(1, 11):
            optional[f"enabled_{i}"] = ("BOOLEAN", {"default": True})
            optional[f"lora_name_{i}"] = (lora_list,)
            optional[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "apply_lora_stack"
    CATEGORY = "loaders/lora"

    def _load_lora_state(self, lora_name: str) -> Dict[str, Any]:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return comfy.utils.load_torch_file(lora_path, safe_load=True)

    def _apply_standard(self, model: Any, clip: Any, lora_state: Dict[str, Any], strength: float) -> Any:
        patched_model, _ = comfy.sd.load_lora_for_models(model, clip, lora_state, strength, 0.0)
        return patched_model

    def _apply_single(self, model: Any, clip: Any, lora_name: str, strength: float) -> Any:
        if abs(strength) < 1e-6:
            return model

        lora_state = self._load_lora_state(lora_name)
        transformer = _extract_transformer(model)

        try:
            translated_state = _translate_lora_state_dict(lora_state, transformer)
            return self._apply_standard(model, clip, translated_state, strength)
        except Exception as exc:
            logger.warning(
                "Custom ZImageTurboLoraStackV4 translation failed for %s (%s). Falling back to default loader.",
                lora_name,
                exc,
            )
            return self._apply_standard(model, clip, lora_state, strength)

    def apply_lora_stack(self, model: Any, clip: Any, lora_count: int, toggle_all: bool, **kwargs: Any):
        if not toggle_all:
            return model, clip

        patched_model = model
        count = max(1, min(10, int(lora_count)))

        for i in range(1, count + 1):
            enabled = bool(kwargs.get(f"enabled_{i}", True))
            lora_name = kwargs.get(f"lora_name_{i}")
            strength = float(kwargs.get(f"strength_{i}", 1.0))

            if not enabled:
                continue
            if not lora_name:
                continue
            if abs(strength) < 1e-6:
                continue

            patched_model = self._apply_single(patched_model, clip, lora_name, strength)  # type: ignore

        return patched_model, clip


NODE_CLASS_MAPPINGS = {
    "ZImageTurboLoraStackV4": ZImageTurboLoraStackV4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboLoraStackV4": "ZImage Turbo LoRA Stack V4",
}
