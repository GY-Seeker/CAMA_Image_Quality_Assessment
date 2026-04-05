import os
import json
from safetensors.torch import load_file, save_file
from ...extras.logging import get_logger
logger = get_logger(__name__)

def export_for_vllm(dst_dir: str, base_arch: str = "Qwen3VLForConditionalGeneration"):
    """
    src_dir: 训练输出目录（里面有 config.json 和 model-0000x-of-0000y.safetensors）
    dst_dir: 导出后的部署目录
    base_arch: vLLM 支持的 architectures 名字。qwen3_vl 用 Qwen3VLForConditionalGeneration
    """
    os.makedirs(dst_dir, exist_ok=True)
    logger.info("looking for "+dst_dir)
    # 2) patch config.json architectures
    cfg_path = os.path.join(dst_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["architectures"] = [base_arch]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger.info("base_arch changed " + base_arch)
    # 3) strip extra_fc.* from ALL safetensors shards
    for fn in os.listdir(dst_dir):
        if fn.endswith(".safetensors") and fn.startswith("model-"):
            p = os.path.join(dst_dir, fn)
            sd = load_file(p)
            new_sd = {k: v for k, v in sd.items() if not k.startswith("extra_fc.")}
            logger.info("model changed")
            if len(new_sd) != len(sd):
                save_file(new_sd, p)
                logger.info(f"[export_for_vllm] stripped extra_fc from {fn}: {len(sd)} -> {len(new_sd)}")
    logger.info(f"[export_for_vllm] done. dst_dir={dst_dir}")