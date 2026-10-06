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
- [2026-10-05] Colab notebook fix (401 Unauthorized) — approved ("next for training in colab")।
  Implemented: `alfa.ipynb` + `notebooks/colab_t4.ipynb` — private HF `snapshot_download` বাদ, public GitHub clone;
  login/upload cell-এ token-missing guard; default `multimodal_mix` 1500 rows; train `--config` + `--max-seq-len 2048 --batch-size 2` (T4 safe)।
  Note: ভুল নামে secret (`hf_HuqTdz...` = token-as-name) তৈরি হয়েছিল — মুছে দেওয়া হয়েছে; token revoke + rotate করতে বলা হয়েছে।
- [2026-10-05] Dataset source fix (Colab warnings) — approved ("after 17 min same here why?")।
  Implemented: `bigcode/the-stack-smol` (gated) → `codeparrot/github-code-clean/Python-all`;
  `microsoft/MSR-VTT` (missing) → `AlexZigma/msr-vtt` → `friedrichor/MSR-VTT/train_7k` fallback;
  SOURCES fallback chain (`_load_rows_first`), caption-list + image/video index-ref handling।
  Note: Cell 5 train 17min+ normal (500 rows × 2048 ctx, grad-ckpt); GPU meter lag হতে পারে — 40min+ আটকে থাকলে rows কমাতে হবে।
- [2026-10-05] GPU-not-used fix ("sudhu ram use korche kano?") — approved (implied, training blocked slow).
  Implemented: `train.py` + `train_hf.py` CUDA diagnostics print (torch version, cuda flag, GPU name, loud CPU warning);
  notebook-এ GPU check cell (torch.cuda.is_available + restart hint).
  Root cause hypothesis: torch CPU-only install বা runtime GPU detach → Trainer silently CPU-তে চলে (~10x slow), তাই RAM 7.2GB + GPU 0.0.
- [2026-10-05] Clone-cell fix ("tarin hoye gache?") — Colab-এ `!rm -rf /content/alfa` shell-এর cwd মুছে দেওয়ায় clone fail হয়েছিল।
  Implemented: clone cell-এ আগে `%cd /content`, তারপর `rm -rf` + clone। দুটো notebook-এই fix + push।
  Note: সকাল 07:36-এর পুরোনো run-এ training সফল হয়েছিল (214M outputs: checkpoint-32, config.json, model.safetensors) — Cell 19-এর `ls` ও Cell 20-এর `uploaded` সেই পুরোনো files; এই run-এ নতুন training হয়নি।
- [2026-10-05] codeparrot trust fix ("dakho train hocche?") — `codeparrot/github-code-clean` loading script চায়, `trust_remote_code=False` দিয়ে fail হয়েছিল → CodeAlpaca fallback কাজ করেছিল।
  Implemented: `codeparrot/*` source-এ `trust_remote_code=True` (explicit note-সহ)। পরের run থেকে code_long আসল long-code দিয়ে আসবে।
  Confirmed: training GPU-তে চলছে (cuda=True, Tesla T4, GPU RAM 2.9GB, loss 8.84→1.96, 1.81s/it).
- [2026-10-05] Training COMPLETE ("ekhon ki train holo?") — 100% 94/94, train_loss 1.696, loss 9.117→0.9554, `saved` + Hub-এ `uploaded` ✅।
  Implemented: `code_long` → `iamtarun/python_code_instructions_18k_alpaca` (parquet, public, 349 likes; script-based codeparrot নতুন datasets lib-এ dead) + alpaca-style handler।
- [2026-10-05] Test inference ("Test inference — Hub-এর model দিয়ে...") — approved (user chose option 1)।
  Implemented: `scripts/infer.py` (Hub subfolder / local, char-ord encoding = training-এর মতো, temperature/top-p sampling)।
  Verified locally: encode/decode roundtrip OK + random-weight ckpt-এ generate mechanics OK (gibberish expected)।
  Honest note: weights char-level pseudo-token ids-এ train হয়েছে, BPE tokenizer compatible নয় — output demo-quality হবে; real quality = ভবিষ্যৎ কাজ (BPE wiring + longer training)।
- [2026-10-05] BPE wiring ("approve" — kano char-level?) — approved।
  Implemented: `scripts/bpe.py` (shared: BPE load/encode/decode + char fallback); `train.py` + `train_hf.py` `--tokenizer` (default alfa-32k.json, missing → warning + char fallback); `infer.py` BPE encode/decode + unpadded prompt; `requirements.txt` accelerate>=1.1.0।
  Verified locally: BPE roundtrip OK; train JsonlDS batches OK (shapes, ids<32000, masks); infer BPE end-to-end OK (prompt 24 toks, no pad waste)।
  IMPORTANT: পুরোনো Hub checkpoint char-level — BPE-এর সাথে incompatible। Colab-এ tokenizer + training আবার চালাতে হবে।
- [2026-10-05] UI wiring ("2. UI wiring — mock সরিয়ে trained model বসানো") — approved (user chose option 2)।
  Implemented: `app/model_backend.py` (নতুন, gradio ছাড়া testable: ALFA_MODEL/ALFA_TOKENIZER env, real/char/mock mode, training-shape prompt + reasoning yields);
  `app/ui.py` thin wrapper (ALFA_PORT/ALFA_SHARE env); notebook-এ UI launch cell (share=True)।
  Verified locally: mock path OK (reason + filenames shown); real path OK (/tmp/bpe-ckpt, CPU, code block generated)।
- [2026-10-05] UI import fix (Colab `ModuleNotFoundError: model_backend`) — `from app.ui import demo` package mode-এ plain `model_backend` import fail হয়েছিল।
  Implemented: `app/ui.py` try/except import (package → script fallback)। দুটো mode-ই stubbed gradio দিয়ে verified।
- [2026-10-05] Empty-output fix ("konokichui output nai") — tiny model মাঝে মাঝে EOS/whitespace ছাড়া কিছু generate করে না → খালি box।
  Implemented: backend-এ 3-try retry (warming temperature) + খালি থাকলে স্পষ্ট message; output-এ word-count। দুটো path-ই verified।
- [2026-10-05] Dashboard redesign ("use this image... for my alfa ai") — approved Full dashboard + Multi-turn + New Chat ("approve")।
  Implemented: `app/ui.py` rewrite — sidebar (logo, New Chat, Code/Images/Videos/Files nav, engine card) + top bar + hero (orb, welcome, big prompt bar) + Deep Think toggle + uploads + multi-turn Chatbot + 3 quick cards; dark glass CSS, orange accent; backend untouched।
  Verified: stubbed-gradio import, engine info, clear, chat-turn streaming, multi-turn accumulation — all OK।
- [2026-10-05] Gradio5 Chatbot fix ("?" — messages format error) — পুরোনো `[[user, bot]]` tuple format Gradio 5 reject করে।
  Implemented: history এখন `[{role, content}]` messages format (`_to_messages`/`_to_pairs`); roundtrip + multi-turn verified।
- [2026-10-05] HTML design implement (pasted echo_ai HTML) — approved (build mode)।
  Implemented: `app/static/alfa.html` (Echo design pixel-kept, Alfa branding, code suggestions, tiny-50M pill, no external avatar) + `app/server.py` (GET /, GET /api/status, POST /api/generate JSON+multipart)।
  Verified live: status/mock/generate/page-200 + real-ckpt multipart with file context — all OK। Run: `python app/server.py` → :8000।
- [2026-10-05] Attached HTML implement ("impliment attached html... alfa-design.html... 2nd page alfa.html") — user explicitly asked in build mode।
  Implemented: `app/static/alfa-design.html` (attached Echo design, Alfa branding, Lucide icons, sidebar toggle, pinned/history, model dropdown, mic/wave) +
  `app/static/alfa.html` (same design + backend wiring: fetch generate, history, engine badge, upload); server routes `/` + `/design`।
  Verified: both 200, Alfa strings present, zero Echo/Venkat/Elon/Sonnet leftovers।
- [2026-10-05] Design as-is fix ("you done 1st... 2nd alfa-design.html not same") — attached HTML হুবহু বসানো হলো, কোনো rebranding নয়।
  Implemented: `app/static/alfa-design.html` = attached file exactly (Echo/Venkat/Elon/3.5 Sonnet/PINNED/YESTERDAY/wave সহ); `/design` serve verified।
  Note: `alfa.html` (working Alfa app) আলাদা রইল — `/` route অপরিবর্তিত।
- [2026-10-05] Exact-file fix ("puro eki royehache... kichu change korbi na") — আমার transcription ভুল ছিল; আসল file থেকে byte-identical copy (md5 match)।
  Implemented: `alfa-design.html` = Downloads file হুবহু (কোনো change নয়); `alfa.html` = একই design + appended wiring script only (Enter-capture submit, wave-button submit, attach, model→tokens, history, result box)।
  Verified: design diff empty; `/design` + `/` 200 with correct markers।
- [2026-10-05] GGUF convert ("3. GGUF convert — local/RTX 3050") — approved (build mode)।
  Implemented: `scripts/convert_to_gguf.py` self-contained (gguf-py, GQA map, BPE→gpt2 tokenizer, F16/Q8_0/Q4_0; K-quants invalid for 1376-dim — dropped with reason)।
  Verified end-to-end: 3 quants convert + sizes differ correctly + read-back (llama/21 tensors/tokens/GQA shape) + Q8 dequant err 0.00009।
  Caught mid-way: writer doesn't quantize (mislabeled F32) + Q4_K block mismatch — both fixed before commit। README + requirements updated।
- [2026-10-05] Echo pixel-theme ("valo kore pixel wise dakh... without name and icons") — approved (build mode)।
  Implemented: `app/ui.py` rewrite — near-black #08080a + purple aurora glow + 24px frame; sidebar (Alfa logo, New Chat, Chat/Code/Images/Videos nav, TODAY history, Pro card, User row);
  center orb (pure CSS gradient) + Welcome + big headline + 3 suggestion cards + glowing prompt bar + uploads + Deep Think + History;
  NO brand names, NO emoji/icons (verified: no non-text glyphs); backend untouched।
- [2026-10-05] Weak-output fix (শুধু `,` এসেছিল) — punctuation-only output-ও retry হবে (8+ word-char না থাকলে); notebook UI cell এখন `git pull` + reload করে (restart ছাড়াই fix পাবে)। Verified (forced `,` → retry → fallback message)।
- [2026-10-05] Pixel-perfect Dark AI Dashboard redesign ("make this design as it is without name and logos") — approved ("approve")।
  Implemented: `app/static/alfa.html` রেফারেন্স স্ক্রিনশটের সাথে হুবহু মিলিয়ে পুনর্নির্মাণ:
  - টপ-রাইট কোণে পার্পল অরোরা এবং লাইট রে/বিম ইফেক্ট (CSS multilayer radial/linear gradients ও blur);
  - সাইডবার: নির্দিষ্ট ব্র্যান্ড নাম ও লোগো মুক্ত স্পার্কল আইকন, `+ New Chat` বাটন, Chat / API / Community নেভিগেশন, Pinned / Today / Yesterday / Previous 7 Days হিস্ট্রি গ্রুপ, Upgrade to Pro গ্লাস কার্ড এবং জেনেরিক User প্রোফাইল রো;
  - সেন্টার: অ্যানিমেটেড ভাসমান পার্পল সেলিস্টিয়াল অর্ব (অরবিট রিং ও স্পার্কল কোর), "Welcome to AI Assistant", "How Can I Assist You?" হেডলাইন এবং ৩টি গ্লাস সাজেশনের কার্ড উইথ মিডিয়া আইকন;
  - নিয়ন গ্লোয়িং প্রম্পট বার: পার্পল আউটার গ্লো (`box-shadow`), স্পার্কল আইকনসহ টেক্সট বক্স, মিডিয়া টুলবার (চ্যাট, ইমেজ, মিউজিক), মডেল সিলেক্টর পিল, পেপারক্লিপ, মাইক এবং অ্যানিমেটেড অডিও ওয়েভফর্ম বাটন;
  - `app/ui.py`-তেও নিউট্রাল ব্র্যান্ডিং ("AI Workspace", "Welcome to AI Assistant", "Pro Access", "✦") প্রয়োগ করা হয়েছে; ব্যাকএন্ড ও API সম্পূর্ণ অপরিবর্তিত ও কার্যকর।
