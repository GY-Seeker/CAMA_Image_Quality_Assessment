import os
import re
import time
from transformers import TrainerCallback
from .export_vllm import export_for_vllm
from ...extras.logging import get_logger
logger = get_logger(__name__)
_pat = re.compile(r"^checkpoint-(\d+)$")

def latest_ckpt(output_dir: str):
    cands = []
    for n in os.listdir(output_dir):
        m = _pat.match(n)
        if m:
            cands.append((int(m.group(1)), os.path.join(output_dir, n)))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]

def _list_shards(ckpt_dir: str):
    return sorted([
        os.path.join(ckpt_dir, fn)
        for fn in os.listdir(ckpt_dir)
        if fn.startswith("model-") and fn.endswith(".safetensors")
    ])

def _filesizes(paths):
    return {p: os.path.getsize(p) for p in paths if os.path.exists(p)}

def _wait_ckpt_complete(ckpt_dir: str, timeout_s: int = 1800, stable_rounds: int = 12, interval_s: float = 5.0):
    """
    等待 checkpoint 写完：
    - 先等 index 文件出现（分片 safetensors 常见）
    - 再等所有 shard 文件大小连续 stable_rounds 次不变
    """
    t0 = time.time()

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")

    # 1) 等 index 出现（若你们保存不生成 index，这步会一直等到timeout；可按需关掉）
    while not os.path.exists(index_path):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timeout waiting index: {index_path}")
        logger.info("wait for model.safetensors.index.json")
        time.sleep(interval_s)
    logger.info("Here is model.safetensors.index.json")
    # 2) 等 shards 大小稳定
    shards = _list_shards(ckpt_dir)
    if not shards:
        # 有些情况是单文件 model.safetensors
        single = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(single):
            shards = [single]
        else:
            raise FileNotFoundError(f"no safetensors found in {ckpt_dir}")

    stable = 0
    last = None
    while stable < stable_rounds:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timeout waiting ckpt stable: {ckpt_dir}")
        cur = _filesizes(shards)
        if last is not None and cur == last:
            stable += 1
        else:
            stable = 0
        last = cur
        logger.info("cur changed")
        time.sleep(interval_s)
    logger.info("all file has writed")

class ExportVLLMOnSaveCallback(TrainerCallback):
    def __init__(self, base_arch="Qwen3VLForConditionalGeneration", timeout_s: int = 1800):
        self.base_arch = base_arch
        self.timeout_s = timeout_s

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = latest_ckpt(args.output_dir)
        if ckpt_dir is None:
            return control

        # 等待保存完成（不靠固定sleep）
        _wait_ckpt_complete(ckpt_dir, timeout_s=self.timeout_s)

        # dst_dir = ckpt_dir + "_vllm"
        export_for_vllm( dst_dir=ckpt_dir, base_arch=self.base_arch)
        return control