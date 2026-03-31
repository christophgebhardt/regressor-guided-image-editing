from dataclasses import dataclass

@dataclass
class AdaptConfig:
    num_inversion_steps: int = 50
    num_inference_steps: int = 50
    end_iteration: int = num_inversion_steps
    normalize_gradient: bool = True
    scheduler_type: str = "ddim"
    save_orig: bool = False
    is_xl: bool = True

@dataclass
class GuidanceConfig:
    clf_scale: float = 0.1
    reference_value: float | None = None
    prompt: str = ""
    negative_prompt: str = ""
    cfg_scale: float = 2.0
    use_caption: bool = True
    is_nto: bool = True
    max: bool = True
    label: str = f"CG_CFG_2_{clf_scale}"
