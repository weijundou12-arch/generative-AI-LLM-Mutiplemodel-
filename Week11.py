Week11 全覆盖代码工程（可跑、具体场景、含注释）
场景：医疗/科研助手（图像 + 文本）

输入：一张“医学图片或截图”（比如术后 X 光、MRI 截图、检验报告截图）
输出：

结构化摘要（Key findings / Uncertainty / Next steps）

JSON 抽取（如果是报告截图）

支持微调数据生成（image + messages）

支持 LoRA 微调骨架（可在小显存机器上跑通的最小闭环）

说明：你可以先用任意图片放到 images/ 目录测试。
代码对 Qwen2-VL / LLaVA 类模型做了“尽量通用”的加载方式（trust_remote_code=True），不保证你本机没有 GPU 也能跑大模型，但工程结构与关键函数完全齐全（面试/工作/研究都够用）。

文件结构
week11_vlm_lab/
  images/                       # 放测试图片
  data/
    make_vl_instruction_jsonl.py
    vl_sft.jsonl                # 生成的数据
  inference/
    vlm_infer.py
  finetune/
    train_vl_lora.py
    vl_lora_config.json
  serve/
    vlm_fastapi.py
  eval/
    eval_structured_output.py
  README.md

1）推理脚本：单图多任务（摘要 + JSON 抽取）

保存为：inference/vlm_infer.py

"""
VLM 推理：给一张图片 + 指令，输出结构化结果
- 支持 Qwen2-VL / LLaVA 系列（尽量通用）
- 关键：AutoProcessor/AutoTokenizer + trust_remote_code
"""

import os
import json
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM


def load_vlm(model_name: str, device: str = "cuda"):
    """
    WHY：多模态模型通常需要自定义 processor（处理图像+文本）；
         trust_remote_code=True 用于加载模型仓库的自定义类（很多 VLM 都这样）
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model, processor, tokenizer


def build_messages(task: str) -> List[Dict[str, str]]:
    """
    统一把任务写成 system/user messages
    WHY：多模态 instruction tuning 依赖稳定模板，推理与训练一致会更稳
    """
    system = (
        "You are a medical multimodal assistant.\n"
        "Rules:\n"
        "1) Use only information visible in the image.\n"
        "2) If image is unclear, say 'insufficient visual evidence'.\n"
        "3) Output must be concise.\n"
    )

    if task == "summary":
        user = (
            "Task: Summarize the image for clinical/research use.\n"
            "Output format:\n"
            "Key findings:\n- ...\n"
            "Uncertainty:\n- ...\n"
            "Next steps:\n- ...\n"
        )
    elif task == "json_extract":
        user = (
            "Task: If the image contains a report/table, extract key fields as JSON.\n"
            "JSON keys: {patient_info: {...}, measurements: [...], notes: ...}\n"
            "If not applicable, output {\"not_applicable\": true}."
        )
    else:
        user = "Task: Describe what you see."

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@torch.inference_mode()
def run_infer(model, processor, tokenizer, image_path: str, task: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(task)

    # 关键：processor 同时处理图像+文本
    # 不同 VLM 的 processor 接口略不同，尽量写成通用形式
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return out


def main():
    model_name = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")  # 你可换成更小/更大
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor, tokenizer = load_vlm(model_name, device=device)

    img = os.getenv("IMG", "images/demo.png")
    for task in ["summary", "json_extract"]:
        print(f"\n=== TASK: {task} ===")
        print(run_infer(model, processor, tokenizer, img, task))


if __name__ == "__main__":
    main()

2）生成多模态 SFT 数据（image + messages → JSONL）

保存为：data/make_vl_instruction_jsonl.py

"""
生成多模态指令微调数据（JSONL）
每条样本：
{
  "image": "images/xxx.png",
  "messages": [{"role":"system","content":...}, {"role":"user","content":...}, {"role":"assistant","content":...}]
}

WHY：
- 多模态 SFT 的关键不是“文本多”，而是“任务覆盖 + 输出格式稳定 + 含拒答样本”
"""

import os
import json
from pathlib import Path
from typing import List, Dict


def sample(system: str, user: str, assistant: str, image_path: str) -> Dict:
    return {
        "image": image_path,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def build_dataset(images_dir: str) -> List[Dict]:
    sys = (
        "You are a medical multimodal assistant.\n"
        "Rules:\n"
        "1) Use only information visible in the image.\n"
        "2) If unclear, say 'insufficient visual evidence'.\n"
        "3) Follow the requested output format strictly.\n"
    )

    # 你自己放几张图片到 images/ 里，然后用文件名构建任务
    imgs = [str(p) for p in Path(images_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
    if not imgs:
        raise RuntimeError(f"No images found in {images_dir}. Put some images there first.")

    data = []
    # 样例 1：摘要任务（模板化）
    data.append(sample(
        sys,
        "Summarize the image.\nFormat:\nKey findings:\n- ...\nUncertainty:\n- ...\nNext steps:\n- ...\n",
        "Key findings:\n- (example) visible text suggests a clinical report screenshot.\nUncertainty:\n- insufficient visual evidence for diagnosis.\nNext steps:\n- provide higher-resolution image or original report PDF.\n",
        imgs[0]
    ))

    # 样例 2：结构化 JSON 抽取（强制可解析）
    data.append(sample(
        sys,
        "If the image is a report/table, extract fields as JSON with keys: measurements(list), notes(string). If not, output {\"not_applicable\": true}.",
        "{\"not_applicable\": true}",
        imgs[min(1, len(imgs)-1)]
    ))

    # 样例 3：拒答样本（非常重要：降低胡说）
    data.append(sample(
        sys,
        "What is the exact numeric value of the lab marker 'CRP' shown in the image? If not clearly readable, say 'insufficient visual evidence'.",
        "insufficient visual evidence",
        imgs[min(2, len(imgs)-1)]
    ))

    return data


def write_jsonl(path: str, rows: List[Dict]):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    out = "data/vl_sft.jsonl"
    rows = build_dataset("images")
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} samples -> {out}")

3）LoRA 微调骨架（PEFT + Transformers）

保存为：finetune/train_vl_lora.py

"""
多模态 LoRA 微调（骨架）
- 使用 transformers + peft (LoRA)
- 数据：data/vl_sft.jsonl
- 模型：Qwen2-VL / LLaVA 类（依赖各自 processor 实现）

注意：不同 VLM 的 forward 输入键可能不同（pixel_values/image_grid 等）。
这里给出“工程骨架 + 关键函数”，你换具体模型时只需在 collator 里对齐字段。
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from peft import LoraConfig, get_peft_model


MODEL_NAME = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DATA_PATH = os.getenv("VL_DATA", "data/vl_sft.jsonl")
OUT_DIR = os.getenv("OUT_DIR", "saves/week11_vl_lora")


def build_prompt(messages: List[Dict[str, str]]) -> str:
    # 简单模板：SYSTEM/USER/ASSISTANT
    # WHY：训练与推理一致；可替换为官方推荐 template
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


@dataclass
class VLDatasetCollator:
    processor: Any
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        texts = []
        labels_text = []

        for ex in batch:
            img = Image.open(ex["image"]).convert("RGB")
            images.append(img)

            # 把 assistant 作为目标输出：训练时让模型学会生成 assistant 内容
            msgs = ex["messages"]
            prompt = build_prompt(msgs[:-1])  # system+user
            target = msgs[-1]["content"]      # assistant

            # 常用做法：把 prompt + target 拼接，labels 只监督 target 区间（需要更复杂的 mask）
            # 这里给一个简化：全监督（演示用）。正式研究建议做 label mask。
            texts.append(prompt + "\nASSISTANT: ")
            labels_text.append(target)

        # 处理图像+文本
        model_inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)

        # 处理 labels（纯文本 labels）
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else nullcontext():
            labels = self.tokenizer(labels_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def main():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # LoRA：通常作用在 LLM 的 attention/MLP 投影层
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,  # 多模态必须关掉，否则 image 字段可能被丢
    )

    collator = VLDatasetCollator(processor=processor, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

4）FastAPI 服务：多模态推理接口（对接你 Week4/Week8 体系）

保存为：serve/vlm_fastapi.py

"""
多模态推理 API（FastAPI）
- POST /infer  {image_path, task}
- 实战要点：trace_id、超时、错误可观测
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Week11 VLM API", version="0.1")

MODEL_NAME = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if device == "cuda" else None
)
model.eval()


class InferReq(BaseModel):
    image_path: str
    task: str = "summary"
    max_new_tokens: int = 256


@app.post("/infer")
def infer(req: InferReq):
    trace_id = str(uuid.uuid4())
    try:
        img = Image.open(req.image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_path: {e}")

    prompt = (
        "SYSTEM: You are a medical multimodal assistant. Use only what is visible.\n"
        f"USER: Task={req.task}. Output concise.\nASSISTANT: "
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=True, temperature=0.2, top_p=0.9)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    return {"trace_id": trace_id, "text": text}


启动：

pip install fastapi uvicorn transformers peft datasets accelerate pillow
uvicorn serve.vlm_fastapi:app --host 0.0.0.0 --port 8003

5）结构化输出评测：检查 JSON 可解析率（工程必备）

保存为：eval/eval_structured_output.py

"""
评测：模型输出是否满足 JSON schema（简化版）
WHY：多模态在“报告截图抽取”里，最关键指标之一就是可解析率（生产可用性）
"""

import json
import re
from typing import Tuple

def extract_json(text: str) -> Tuple[bool, str]:
    # 粗暴截取第一个 {...} 区间（演示）
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False, ""
    return True, m.group(0)

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except:
        return False

if __name__ == "__main__":
    sample_output = """Some text... {"measurements":[{"name":"CRP","value":"12"}], "notes":"..."} trailing"""
    ok, js = extract_json(sample_output)
    print("found_json:", ok)
    print("json_valid:", is_valid_json(js))

Week11 你现在就能做的“最小闭环任务”

往 images/ 放 1–3 张截图（任意：报告、表格、医学影像截图）

python data/make_vl_instruction_jsonl.py 生成 data/vl_sft.jsonl

python inference/vlm_infer.py 跑推理看看摘要/抽取效果

有 GPU 再跑 python finetune/train_vl_lora.py 做 LoRA 微调

uvicorn serve/vlm_fastapi:app ... 把它接入你 Week8 的 MCP/Agent 工具链
