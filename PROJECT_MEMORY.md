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
- [2026-10-05] Working page behaviors ("kay kintu eita to kaj korche na...") — `/design` static showcase, কাজ করে `/` (working app)।
  Implemented in `alfa.html` only (`alfa-design.html` untouched, still byte-identical): model pill + dropdown = tiny-50M/tiny-50M-long (256/512 tokens);
  wave button = send; paperclip = file attach + image tool = image attach; TODAY panel = live history, click → full communication in main area;
  mic + music buttons removed (no backend)। Verified served + API।
- [2026-10-05] Real-engine proof ("ball model chole na to") — local checkpoint ছিল না বলে mock চলছিল।
  Implemented: mini-data (30 rows) + BPE tokenizer + torch-only manual train (accelerate নেই বলে Trainer নয়), 30 steps loss falling → `/tmp/alfa-real-ckpt`;
  server + API verify: `{"mode":"real"}` ✅। User-এর জন্য: Hub ckpt download বা full train করলে UI auto real হবে।
- [2026-10-05] Toolbar simplify ("Attach file button থাকবে, বাকি লাগছে না") — `alfa.html` থেকে chat/image/music/mic buttons removed, শুধু paperclip (সব file type); design file untouched।
- [2026-10-05] Quality fix both ("bhul val kaj korche to! ki koronio?" → দুটোই) — approved ("apply")।
  Implemented (Part B): backend temp 0.6 default + repetition_penalty 1.15 + question-style prompt nudge; Gradio temp slider; infer/server defaults aligned। Verified।
  Part A (Colab retrain): `multimodal_mix --max-rows 4000 --overwrite` + tokenizer + `train_hf.py --epochs 3` (~1-2h T4), Hub upload, infer sample check।
- [2026-10-05] Scale-up trio (user list: 5000+ rows/3-5 epochs + chat Q&A + 1B) — approved (build mode)।
  Implemented: `chat_qa` source (ultrachat_200k → alpaca fallback, modality=chat, multi-turn aware) + append recipe; `configs/base-1B.yaml` (d2048/L22/H32/KV8/FFN5632, seq4096, verified builds to 1.12B/2.2GB fp16);
  notebook scale-up cell (5000 mix + 2000 chat + 1B train) in both notebooks; workflow choice + README।
  Verified: chat handler both shapes, notebooks valid, workflow YAML OK। Note: 1B needs 16GB+ GPU; CI stays smoke-size।
- [2026-10-05] Tiny-50M release ("Tiny-50M er ekta release dau... version 0.0.1...") — approved (build mode)।
  Implemented: tag v0.0.1 + GitHub release (Q8_0 59.5MB + Q4_0 31.5MB + demo tokenizer from local smoke ckpt, honestly labeled preview);
  README version → 0.0.1। Version scheme: 0.0.x tiny fixes, 1B next as v0.1.0।
- [2026-10-05] OOM fix (Colab `torch.OutOfMemoryError` at first optimizer step on 1B/T4) — root cause: 1.1B Adam fp32 states (~9GB) + weights/grads exceed T4.
  Implemented: `--optim adamw_8bit` (halves optimizer VRAM) in both train scripts + `optim: adamw_8bit` in base-1B.yaml; `--grad-accum` override; OOM catch with 4-step fix hints; bitsandbytes in notebook pip cell।
  Verified: argparse flags, config load both yamls, OptimizerNames valid, notebooks valid। Colab: restart runtime first (old processes hold VRAM), then scale-up cell rerun।
- [2026-10-05] Kaggle production verify — ultrachat split fix কাজ করেছে (train_sft → 2000 rows, tokenizer vocab 32000) ✅। কিন্তু session No Accelerator → torch+cpu, 1B CPU-তে অসম্ভব। Fix: session GPU T4x2 enable করতে হবে।
- [2026-10-05] T4x2 vs TPU ("konta batter?") — **GPU T4x2**। TPU-তে চলবে না (PyTorch+bitsandbytes+fp16 CUDA-only stack; TPU-তে torch-xla rewrite লাগতো)।
  Implemented: `cuda_hint()` in bpe.py (Kaggle vs Colab aware), used by both train scripts। Verified both messages।
- [2026-10-05] Kaggle 2 errors ("why?" + OOM hints + 404s) — (1) self_learn tried Hub download of local path `outputs/tiny-50M` → 404; (2) OOM with DDP frames (accelerate used BOTH T4s, doubling memory).
  Implemented: ckpt fast-fail with actionable message (no more confusing 404); all train cells `CUDA_VISIBLE_DEVICES=0` (single GPU, no DDP buckets);
  cycle cells pass `--ckpt {OUT}`; kaggle cycle retrain also got missing `--resume-from {OUT}`। Verified cells + clean exit।
- [2026-10-05] FSDP both-GPUs ("why single T4... time kom lagto na?" → Recommended bs2/seq2048/accum8 → "apply")।
  Implemented: `--fsdp full_shard` (+FULL_STATE_DICT save, Llama wrap) in both train scripts; adamw_8bit auto-fallback (incompatible); kaggle setup USE_FSDP/LAUNCH/ACCUM/FSDP_FLAG, train+cycle cells use them।
  Verified: compile, flags, resolve matrix, cfg keys, notebook cells, TrainingArguments signature। First real test = Kaggle smoke (quota-safe), then full run।
- [2026-10-05] Kaggle clone getcwd bug ("ball korcho bara... dakh ki holo") — shell cwd (/kaggle/working/alfa) মুছে ফেলায় clone fail → সব cell cascade fail। Colab-এ আগের fix Kaggle notebook-এ ছিল না।
  Implemented: kaggle clone cell-এ `%cd /content`→`%cd /kaggle/working` guard; 3 notebook-এ guard-before-rm verified।
- [2026-10-05] MODEL switch ("tiny-50M kano ami to 1B nicchi to?") — default train cell tiny-50M-এ hardcode ছিল।
  Implemented: setup cell (`MODEL = "base-1B"` / `"tiny-50M"`) থেকে CONFIG/OUT/EPOCHS/SEQ_LEN/BS/ROWS/CHAT_ROWS derive; data + train cells `{var}` interpolation;
  redundant scale-up cell removed; upload/UI cells already model-agnostic।   Verified order clone→setup→data→train→upload→ui।
- [2026-10-05] Self-learning loop 1+2 ("1+2 plan + implement করবো" → Colab manual 1B + both seeds + auto-upload/review → "apply")।
  Implemented: `data/seeds/topics.txt` + `scripts/self_learn.py` (topics+train seeds, retry gen, 4 gates: meaningful/dedup/lang-aware-parse/bounds, keep-rate abort, 30% cap, newline-safe append) + notebook cycle cell (retrain to self-cycle-N, cp to promote) + upload auto-discover।
  Verified end-to-end locally: seeds/filter/abort/positive/trim/fresh/cap/schema all OK। Found+fixed real bug: append glued rows when file lacks trailing newline।
- [2026-10-05] Kaggle notebook ("train limit nai... kaggle er jonno notebook banau") — Colab 1B train success (100% 375/375, 4.2GB) + Colab quota শেষ।
  Implemented: `notebooks/kaggle_t4.ipynb` — kaggle_secrets HF_TOKEN, /kaggle/working paths, MODEL switch, data/train/cycle/upload/infer cells, no colab imports।
  Verified: no google.colab refs, no /content paths, python syntax OK।
- [2026-10-05] Resume training ("1B ta to train hoyechilo... ki abar korte hobe?") — প্রতি cycle-এ from-scratch retrain মানে 3h ফালতু; checkpoint থেকে continue করাই সঠিক।
  Implemented: `--resume-from` in train_hf.py + train.py (shared build_model; arch must match config; clean exit on bad path) + cycle cell retrains with `--resume-from {OUT} --epochs 1` into versioned self-cycle-N।
  Verified: resume loads exact ckpt weights, fresh differs, bad path exits clean, notebooks valid।
  Review model: auto-upload versioned `outputs/self-cycle-N`, promote to base-1B only on approve (rollback = previous cycle)।
- [2026-10-05] 1B OOM in SDPA (log: ultrachat wrong split + OOM at 2% + gradio6 css warning) — 3 fixes, all verified।
  Implemented: SOURCES split support (ultrachat→train_sft, all others explicit train); scale-up 1B cell --max-seq-len 1024 (2048 OOMs on T4) + PYTORCH alloc hint;
  ui.py Gradio6 css compat (Blocks vs launch by version) + launch_share() used by notebook cell।
- [2026-10-05] Upload/UI model auto-detect ("tiny-50M kano ami to 1B nicchi to?") — upload cell শুধু tiny-50M জানত।
  Implemented: upload cell এখন tiny-50M + base-1B দুটোই detect করে upload করে (যেটা train হয়েছে); UI cell base-1B থাকলে সেটা, নইলে tiny-50M serve করে। দুটো notebook-এই, syntax verified।
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
- [2026-10-05] Disk-full crash ("kano holo?") — 15%-এ checkpoint save-এ ENOSPC (FSDP ঠিকই চলছিল, loss falling)। Checkpoint dirs empty → progress lost।
  Implemented: disk preflight (fail fast with cleanup advice) + `--save-steps/--save-total-limit` (notebook: 200/1) + `--resume-ckpt` Trainer resume + kaggle cleanup lines (checkpoints + HF/pip cache + df check)।
  Verified: estimate/check/kwargs unit tests, notebook cells, flags। Rerun: cleanup auto-runs, crash হলে --resume-ckpt দিয়ে continue।
- [2026-10-05] Disk-full at 61% ("kano holo?") — checkpoint save-এ ENOSPC (FSDP/loss ঠিক ছিল); checkpoint dirs empty → progress lost।
  Root cause: optimizer states ~9GB/checkpoint + HF cache on small root mount; preflight ভুল mount দেখেছিল (আমার bug)।
  Implemented: multi-mount preflight (outputs + HF_HOME + TMPDIR) + optimizer-aware estimate (1B ~= 29GB) + kaggle cache redirect (HF/PIP/TMP → working disk) + stronger cleanup + df gate।
  Verified: estimate/check/kwargs unit tests, notebook cells। train.py shared helpers বলে auto-covered।
- [2026-10-08] Preflight worked but blocked (20.9 < 29.2 GB) — root cause: my own cache redirect filled working disk (websight images + datasets in /kaggle/working/.hf-cache), old cleanup didn't delete it.
  Implemented: cleanup now `du` + removes working caches too (both train cells). Action for user: just re-run train cell (cleanup+preflight run first, no fresh session needed).
- [2026-10-08] Preflight blocked at 20.9<29.2GB though panel showed 2.4/57.6GB — user rightfully angry: 4GB model vs 29GB demand explained (optimizer 9GB x2 rotation + margin).
  Implemented: `--min-disk-gb` override in both train scripts (user judges, user risks). Fresh session likely has clean disk → rerun passes.
- [2026-10-08] Panel-vs-container disk mystery (2.4/57.6 panel vs 20.9 free preflight) — HF datasets cache (websight images etc.) filled the container overlay, not working dir.
  Implemented: download_data.py auto-purges HF datasets cache after writing train.jsonl (rows already safe); preflight failure now prints du breakdown of hogs. Verified purge safety + hog report.
- [2026-10-08] "what exact need?" (Disk 2.4/57.6 panel vs 20.9 free) — exact math executed: weights 2.25 + adam 8.98, x2 rotation + margin = 29.1GB.
  Fix: `--save-only-model` default True (checkpoints = weights only, no 9GB optimizer) → need 5.8GB, fits 20.9GB easily; `--no-save-only-model` restores full state; resume-ckpt+weights-only note; train.py mirrored.
  Verified: signature, kwargs both modes, estimates, help. Crash resume path = --resume-from (weights, fresh optim).
- [2026-10-08] Final-save crash at 100% ("kano holo?" + screenshots) — 657/657 done, loss 0.9458, then RuntimeError (invalid python storage) in safetensors.
  Root cause: our own `model.save_pretrained()` on FSDP-sharded params (bypasses Trainer's gather). Proof: run survived all step-N Trainer saves to 100%; installed transformers 5.17 `save_model()` uses `accelerator.get_state_dict()` for FULL_STATE_DICT.
  Implemented: both train scripts keep `trainer` ref + call `trainer.save_model()` (with why-comment); verified via mock-Trainer test (save_model:1, bare save:0) on CPU.
  Recovery (no retrain): `ls` OUT — if latest checkpoint-N/ has model.safetensors (~2GB+), it IS the trained model (Trainer-saved, FSDP-safe) → upload/infer/GGUF straight from it; else rerun train cell.
- [2026-10-09] Kaggle T4 training headless ("train in Kaggle_T4") — CLI দিয়ে সরাসরি চালানো, browser ছাড়াই।
  Implemented: staging `/tmp/alfa-kaggle-kernel/` = `notebooks/kaggle_t4.ipynb` + `kernel-metadata.json` (id `mrsandipmandal/alfa-train-t4`, `enable_gpu/internet: true`, `machine_shape: NvidiaTeslaT4` = T4×2) + `kaggle kernels push -p /tmp/alfa-kaggle-kernel` (kaggle CLI 2.2.4: push = upload + run)।
  Verified: status RUNNING; live log শুধু `kaggle kernels logs mrsandipmandal/alfa-train-t4 -f`-এ দেখায় (plain logs = শুধু run-এর পর) — 2× Tesla T4 15360MiB, clone+data+tokenizer done, train step 90/657, loss ~1.05-1.10, 27s/it, ETA ~4h20m (session 12h limit-এর মধ্যে)।
  Verified (15:14 IST): step-200 checkpoint save PASSED (`Writing model shards 1/1` ~3s, কোনো ENOSPC/ক্র্যাশ নেই) → training 201+ ধরে চলেছে, step 210/657, loss ~0.81-1.15, ~27s/it, ETA ~3h25m; পরবর্তী milestone step 400 save + 100% final `trainer.save_model()`।
  Note: user-এর Kaggle API token (`KGAT_…`) = `~/.kaggle/access_token`-এ আগে থেকেই সেই একই token (SAME), auth verified EXIT:0; কোনো বদল লাগেনি, token কোথাও repo-তে লেখা নেই। HF_TOKEN secret CLI-তে attach করা যায় না → Hub upload cell শেষে skip করবে; পরে web UI-তে secret attach করে upload cell চালাতে হবে, অথবা `kaggle kernels output mrsandipmandal/alfa-train-t4 -p <dir>` দিয়ে output নামাতে হবে।
- [2026-10-09] APPROVED + APPLIED: code-100k plan — text-only code+reasoning model, **min 100,000 context**, 2×T4 Kaggle primary + Colab demo.
  Approved answers: params **1.5B** (rejected 1B/3B), train seq **8192**, datasets **6** (code_mix incl. apps/TACO reasoning; OpenOrca/general removed).
  How 100K is real on T4: train @8192 plain RoPE, YaRN rope_scaling (factor 16 → 131072 eff) written into saved config.json at save time → HF inference `--context 100000` + GGUF `--ctx-size 100000` (llama.cpp -c 100000). Honest README wording: trained @8K, YaRN-scaled to 100K+.
  Applied: configs/code-100k.yaml (~1.33B, rope_theta 1e6); download_data.py MAP/SOURCES (code_evolve=Magicoder-OSS-12K, code_reason=codeparrot/apps->mbpp, code_reason2=TACO->mbpp, code_mix sampler, `_problem_row`, "reasoning" key removed + list_datasets); train_hf.py/train.py `_apply_rope_scaling()` BEFORE trainer.save_model(); convert_to_gguf.py --ctx-size + defensive yarn metadata; infer.py --context YaRN override; kaggle_t4.ipynb (default MODEL=code-100k, FSDP no longer overrides per-model SEQ=8192, data cell code_mix 40k, CTX_FLAG on infer, token-regenerate warning in header); colab_t4.ipynb (code-100k branch seq 4096 single T4, 100K demo cell w/ Hub pull fallback, UI discovery list).
  Verified: py_compile all scripts; param math ~1.33B; unit tests (problem_row, code_mix sampling, SOURCES/MAP consistency, _apply_rope_scaling, mock-trainer save-order with rope present AT save); nbformat.validate both notebooks; --help outputs.
  Security: user's Kaggle token pasted in chat → user must regenerate it; tokens only via Kaggle Secrets/HF token — never repo/notebook.
  NEXT (user): Kaggle T4x2 session → run kaggle_t4.ipynb (train) → Save Version + Hub upload → then v0.0.2 release (GGUF Q8_0+Q4_0, ctx 100000).
- [2026-10-09] Colab notebook hardened for free T4 15GB (user: "colab er notebook ta o thik koro"):
  ACCUM per model (code-100k/base-1B=16, tiny=8 → effective batch 16 = Kaggle parity), train+self-learn cells get `--grad-accum`, PYTORCH_CUDA_ALLOC_CONF=expandable_segments, `--save-steps 100 --save-total-limit 1`, OOM/resume hints (SEQ_LEN 2048 fallback + `--resume-from {OUT}`), 100K-demo cell now guards missing Hub model with Bengali message. Fixed IPython bug I introduced (bare shell lines without `!` → would SyntaxError). Verified: nbformat.validate, no bare-shell lines in either notebook.
- [2026-10-09] APPROVED + APPLIED: Option A — Colab single-T4 1.39B train infeasible; Kaggle 2×T4 = primary path.
  Evidence (3 empirical OOMs @ seq 4096/2048/1024, identical point): HF AMP keeps fp32 weights+grads (11.1GB) + 8-bit opt states (2.8GB) = 13.9GB FIXED on 14.56GiB T4 → no seq fits; not a bug, capacity. Colab session `alfa-colab` stopped; Colab kept for later GGUF convert/download.
  Kaggle notebook: added multi-session auto-resume (cell 7 → numeric-sorted latest `checkpoint-*` → `--resume-from`, weights+fresh optimizer = documented train_hf path). Needed: 40000 rows × 2 ep @8192 est. 26-52h >> 12h session; checkpoints live in `/kaggle/working/outputs/code-100k/` which survives the fresh-clone (cell 4 only wipes `alfa/`).
  Restaged `/tmp/alfa-kaggle-kernel/alfa-train-t4.ipynb` + resume-logic unit test (numeric sort: 200/400/1000 → 1000; empty → fresh) + nbformat.validate + metadata assert (T4×2 GPU). Colab scripts (colab_setup.py ran end-to-end: clone+deps+code_mix 5000+tokenizer; colab_train.py) committed too.
  PENDING user: (1) stop old RUNNING kernel `alfa-train-t4` (08:05, pre-code-100k notebook) — required before push on same id; (2) rows/epochs decision: keep 40000×2 (~3-5 sessions, weeks of quota) vs trim to 40000×1 (~3 sessions) vs 15000×1 (~1 session pilot).
- [2026-10-09] APPROVED + APPLIED: Option B (40000 rows × 1 epoch) with user's 30h quota + kernel v2 launched.
  Notebook tuple → EPOCHS 1 (40000×1 ≈ 14-28h fits 30h; A=2ep rejected as 26-55h). `kaggle kernels push -p /tmp/alfa-kaggle-kernel` accepted WHILE v1(old notebook) was RUNNING → v2 replaced it (CLI has no stop command; push = push+run, Kaggle one-session-per-kernel). Verified live log: fresh clone → `MODEL: code-100k | epochs: 1 | seq: 8192 | rows: 40000` + `FSDP full_shard` + `--context 100000` + code_mix streaming (CodeAlpaca_20K first source).
  User Q: PC needed? — NO during training (Kaggle cloud runs 12h sessions; PC only for push/monitor/output-pull). If session dies at 12h: re-push same staging → cell-7 RES_FROM auto-loads `/kaggle/working/outputs/code-100k/checkpoint-N` (survives in working dir).
- [2026-10-10] Kaggle run failed at step 3 (user: "last kaggle model training is failed — check reason and retry") — diagnosed from `kaggle kernels logs mrsandipmandal/alfa-train-t4`.
  Status said COMPLETE although training died (a `!cmd` failure inside a notebook cell does not fail the run) → cells 8/10 then cascaded with `--resume-from has no config.json` / `no config.json`.
  **Cause: `torch.OutOfMemoryError` on BOTH T4s at step 3/2500 (14.14/14.56 GiB in use, 426 MiB free).** FSDP `full_shard` fixed block per GPU = sharded fp32 params 2.59 + grads 2.59 + **AdamW states 5.18 = 10.36 GiB before any activation**, + seq-8192 activations & 32000-vocab logits ≈ 14.1 GiB. AdamW states are allocated lazily at the first `optimizer.step()` → steps 1-2 survived, step 3 died. `fp16=True` only autocasts compute; FSDP sharded weights/grads/optimizer stay fp32 (accelerate never sets `mixed_precision_policy` from `Accelerator.mixed_precision`), so it does not shrink that block.
  **Two more blockers in the same log:** (1) 117-122 s/step × 2500 = **83h vs 30h quota** — every row was padded to a full 8192 tokens while real content averages **115 tokens (71× waste)**; (2) pad ids sat in `labels` (cross_entropy ignores only -100) → the model was trained to predict pads, which is also why loss "fell" to 0.95 fast. The 2026-10-09 14-28h estimate was wrong for exactly this reason.
  Implemented: train_hf.py + train.py — `JsonlDS` returns UNPADDED ragged samples + module-level `PadCollator` (batch-max padding, -100 labels) wired into Trainer; configs/code-100k.yaml `optim: adafactor` (5.18 → ~0 B/param optimizer state = the OOM fix; adamw_8bit is FSDP-incompatible, adamw_torch too big); kaggle_t4.ipynb — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, cell 7 no longer pretends to delete checkpoints (they are the resume source), cells 8/10 skip cleanly when `OUT` has no config.json; train_hf.py prints `optim=` + `mixed_precision=` for diagnosis; colab_train.py docstring.
  Verified locally (RTX 3050): py_compile both trainers, smoke runs EXIT 0 (loss finite, grad_norm 3-6 — no NaN, model saved), collator unit test (`-100` pads), 71× waste measured on real train.jsonl, IPython transform of every notebook code cell + nbformat.validate. Expected: new loss curve starts higher (no pad tokens to copy) and ETA drops 83h → ~1-2h.
- [2026-10-10] Retry (Kaggle run 2, kernel v3) died at step 2/2500 with a NEW error after the OOM fix worked: `RuntimeError: aten.add_.Tensor got mixed torch.Tensor and DTensor` at `transformers/optimization.py:1266` (Adafactor.step `exp_avg_sq_row.add_(update.mean(dim=-1))`), both ranks, exit 1 (kernel status again misleadingly COMPLETE; cells 8/10 correctly SKIPped this time).
  What the same log confirmed FIXED: clone pulled da38d41, `optim=adafactor` + `mixed_precision=fp16`, disk preflight 20.9GB >= 7.2GB, and the padding fix: **5.7-7 s/it (was 117 s/it), ETA ~4h (was 83h)**.
  Root cause: transformers 5.18 Adafactor inits factored state as `torch.zeros(shape).to(grad)` = **plain torch.Tensor**, while FSDP2 (what accelerate/transformers now use for `fsdp full_shard` on 2 GPUs) makes every param/grad a **DTensor**; AdamW survives because it uses `torch.zeros_like(p)` which propagates DTensor. First `optimizer.step()` that actually runs (step 2; step 1 was skipped by the fp16 GradScaler after an init overflow = the logged `grad_norm: nan`) then crashes on the mixed-type `add_`.
  Fix: `train_hf.py::patch_adafactor_dtensor()` called from `resolve_fsdp_optim()` whenever `optim == "adafactor"` (covers train.py too) — pre-creates `exp_avg_sq_row/col` (and `exp_avg_sq`/`exp_avg`) via `torch.zeros_like(<grad-derived tensor>)` so state inherits the grad's exact DTensor spec (row=Shard(0), col matches `update.mean(dim=-2)`); the original `step()` then skips its own broken init and its `.to(grad)` resume-casts are no-ops. On plain tensors the result is bit-identical, single-GPU unaffected; idempotent.
  Verified: DTensor unit test (gloo world1 + device mesh) reproduces Kaggle's EXACT error unpatched, then PASSes 6 checks (patch idempotent, plain path bit-identical, DTensor step matches plain math, state all DTensor); py_compile both trainers; wiring prints once; end-to-end smoke on RTX 3050 (40 rows, 10 steps, --optim adafactor) EXIT 0, loss 10.36 -> 9.02, grad_norm 9.5 -> 2.6 finite, model saved.
  Still to watch next run: `grad_norm: nan` at steps 1-2 (also in run 1) — if loss stays ~10.7 for 50+ steps, fp16 stability needs a look; expected normal = GradScaler skips early steps then loss descends (baseline now genuinely high, no pad tokens to copy).
  Retry outcome (kernel v4, pushed da38d41..8728e24 first): VERIFIED PAST CRASH ZONE — `patched Adafactor.step` printed on both ranks, step 2 (old crash point) passed, 8 loss logs through step 16: loss 10.8 -> 10.17 -> 9.65 -> 8.00 -> **7.512**, grad_norm finite (4.4-15.9) after a single expected step-2 GradScaler skip (`nan` only at step 2, then never again), 5.6-5.7 s/it, ETA ~3h53m (fits 12h session + 30h quota). Long watch armed for `train_runtime`/failure patterns.
