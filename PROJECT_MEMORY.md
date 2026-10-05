# PROJECT_MEMORY.md — Alfa-Code

> নিয়ম: ব্যবহারকারী ভাঙা ইংরেজি/বাংলায় বলবেন। প্রতিবার assistant করবে:
> 1) English correction, 2) বাংলায় বুঝিয়ে বলা, 3) কী করতে হবে + approval চাওয়া,
> 4) approve/apply হলে এই ফাইল আপডেট করা।

## 1. Datasets ব্যবহৃত (scripts/download_data.py MAP)

| Key (আমাদের নাম) | HuggingFace ID | Modality | কাজ |
|---|---|---|---|
| `code_small_test` | `HuggingFaceH4/CodeAlpaca_20K` | code | ছোট instruction: prompt → completion |
| `code_long` | `bigcode/the-stack-smol` (subset `data`) | code, high-context | বড় ফাইল, 24k chars পর্যন্ত রাখা হয় |
| `code_instruct` | `bigcode/self-oss-instruct` | code+reasoning | instruction/output মিক্স |
| `reasoning` | `Open-Orca/OpenOrca` | code+reasoning | reasoning ডেটা |
| `image_to_code` | `HuggingFaceM4/websight` | image+code | screenshot → HTML (`<image>` placeholder, URL শুধু রাখা হয়, PIL bytes নয়) |
| `video_to_code` | `microsoft/MSR-VTT` | video+code | clip → caption/code (`<video> x4` placeholder) |
| `multimodal_mix` | (উপরের 3টার combo) | code+image+video | code_long + image_to_code + video_to_code সমান ভাগে sample + shuffle |

Unified row schema (`data/processed/train.jsonl`):
`{modality, prompt, think, answer, images[], video, context_len, source}`

High-context setting:
- `--max-prompt-chars 24000` + `--max-answer-chars 24000` (~8k tokens each)
- `configs/tiny-50M.yaml`: `max_seq_len: 8192`, `rope_theta: 500000.0`, `batch_size: 1`, `grad_accum: 8`, `gradient_checkpointing: true`
- Colab T4-তে batch 1; RTX 3050 4GB-তে `--max-seq-len 2048 --batch-size 2` দিয়ে override

## 2. English correction log

ব্যবহারকারীর মূল বাক্য (2026-10-05):
> "what you undastand and currect my english i have told you english and what need to do explain and ask for me i can apply this plan every time in bangali and PROJECT_MEMORY.md update it when approve/apply"

Corrected English:
> "Tell me what you understood, correct my English above, explain what needs to be done and ask for my approval. I want to apply this plan every time in Bengali, and update PROJECT_MEMORY.md when I approve/apply."

## 3. Reusable plan (প্রতিবার এটাই চলবে)

1. ব্যবহারকারী যা বললেন → আমি কী বুঝলাম বাংলায় লিখবো।
2. ভুল ইংরেজি থাকলে → Corrected English দেবো।
3. কী কী dataset/config/script বদলাবে → বাংলায় step-by-step বুঝিয়ে approval চাইবো।
4. ব্যবহারকারী "হ্যাঁ/approve/apply" বললে → কোড বদলে + এই ফাইলের `Approvals` সেকশনে তারিখসহ লিখবো।

## 4. Approvals

- [2026-10-05] Multimodal dataset (code_long + image_to_code + video_to_code + multimodal_mix) + 8k context — approved, implemented.
- [2026-10-05] এই PROJECT_MEMORY.md ফাইল তৈরি — approved ("হ্যাঁ, তৈরি করো")।
- [2026-10-05] CI/CD push enable (all + direct push to main) — approved ("apply")।
  Implemented: repo `default_workflow_permissions` read→write; `main` unprotected (GITHUB_TOKEN push OK, PAT লাগে না);
  `.github/workflows/train-sync.yml` (dispatch+weekly, CPU-safe defaults 300 rows/2048 ctx, `[skip ci]` push, Hub upload, tag+release);
  বড় binary (`*.bin`, `data/processed/`) git-এ নয় — Hub + Release asset হিসেবে। `HF_TOKEN` secret এখনো বাকি।
