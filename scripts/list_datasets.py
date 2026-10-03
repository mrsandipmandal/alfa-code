"""Best HuggingFace datasets for multimodal coding model (code+reasoning+image+video)."""
DATASETS = {
    # code text - gated large, needs HF_TOKEN
    "code_pretrain": "bigcode/the-stack-v2",
    # public test - no login needed
    "code_small_test": "HuggingFaceH4/CodeAlpaca_20K",
    # reasoning / instruction
    "code_instruct": "bigcode/self-oss-instruct",
    "reasoning_general": "Open-Orca/OpenOrca",
    # image -> code (screenshot + html)
    "image_to_code": "HuggingFaceM4/websight",
}

print("Recommended HF datasets:")
for k, v in DATASETS.items():
    print(f"  {k:18s} -> {v}")
print("\nRun: python scripts/download_data.py --dataset code_small_test --max-rows 500")
