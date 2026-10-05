"""Alfa-Code dashboard UI (Syntra-style dark theme).

Layout: sidebar (logo, New Chat, features, engine card) + top bar +
hero (orb, welcome, big prompt bar) + uploads + multi-turn history +
quick cards. Generation served by app/model_backend.py (real checkpoint
when available, else mock mode with reason).
"""
import os

import gradio as gr

try:
    # package mode: `from app.ui import demo` (Colab cell, python -m app.ui)
    from app.model_backend import generate, get_backend
except ImportError:
    # script mode: `python app/ui.py` (script dir on sys.path)
    from model_backend import generate, get_backend


CSS = """
.alfa-app { background: radial-gradient(1200px 600px at 70% -10%, #12303a 0%, transparent 60%),
  radial-gradient(1000px 500px at 110% 110%, #0e3a2f 0%, transparent 55%),
  linear-gradient(180deg, #0b1220 0%, #070b14 100%) !important; }
.alfa-sidebar { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 12px; }
.alfa-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 14px; padding: 12px; }
.alfa-orb { width: 72px; height: 72px; margin: 6px auto;
  border-radius: 50%; background: radial-gradient(circle at 35% 30%, #7dd3fc 0%, #22d3ee 25%, #0ea5e9 55%, #052e3b 100%);
  box-shadow: 0 0 40px rgba(34,211,238,0.45); }
.alfa-hero-title { text-align: center; color: #e2e8f0; font-size: 13px; letter-spacing: 3px; }
.alfa-hero-sub { text-align: center; color: #f1f5f9; font-size: 22px; font-weight: 700; }
.alfa-topbar { color: #94a3b8; font-size: 13px; }
.alfa-topbar b { color: #e2e8f0; }
.alfa-prompt textarea { background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 14px !important;
  color: #f1f5f9 !important; font-size: 15px !important; }
.alfa-send { background: #f97316 !important; color: #fff !important;
  border-radius: 12px !important; font-weight: 700 !important; }
.alfa-nav button { text-align: left !important; }
.alfa-engine { font-size: 12px; color: #94a3b8; }
.alfa-engine b { color: #4ade80; }
"""


def _engine_info():
    b = get_backend()
    if b["mode"] == "mock":
        return (f"<div class='alfa-engine'>Engine: <b>mock</b><br>{b['reason']}<br>"
                "Train + set ALFA_MODEL for real.</div>")
    return (f"<div class='alfa-engine'>Engine: <b>{b['mode']}</b><br>"
            f"device: {b['device']}</div>")


def _chat_turn(prompt, history, image, file, video, deep_think):
    """One Generate click -> streams (status, updated_history)."""
    history = list(history or [])
    user_label = (prompt or "")[:200]
    n_tokens = 512 if deep_think else 256
    final_md = ""
    status = "[Thinking...]"
    for status, chat_md in generate(prompt, image, file, video,
                                    max_new_tokens=n_tokens):
        final_md = chat_md
        yield status, history + [[user_label, chat_md + "\n\n_...generating..._"]]
    yield "[Done]", history + [[user_label, final_md]]


def _clear():
    return [], "", "[Idle]"


TEMPLATES = {
    "code": "Write a Python function that ",
    "image": "<image>\nBuild this UI as a single HTML file: ",
    "video": "<video> <video> <video> <video>\nDescribe this clip, then write code to render it: ",
    "file": "Read the attached file and explain + improve the code:\n",
}


with gr.Blocks(title="Alfa-Code", css=CSS, elem_classes="alfa-app") as demo:
    with gr.Row():
        # ---------- sidebar ----------
        with gr.Column(scale=1, elem_classes="alfa-sidebar"):
            gr.Markdown("## 🅰️ Alfa-Code\n`tiny-50M` →")
            new_chat = gr.Button("+ New Chat", variant="primary")
            gr.Markdown("**FEATURES**")
            nav_code = gr.Button("🧩 Code", elem_classes="alfa-nav")
            nav_image = gr.Button("🖼️ Images", elem_classes="alfa-nav")
            nav_video = gr.Button("🎬 Videos", elem_classes="alfa-nav")
            nav_files = gr.Button("📁 Files", elem_classes="alfa-nav")
            gr.Markdown("**ENGINE**")
            engine_md = gr.Markdown("loading...")
        # ---------- main ----------
        with gr.Column(scale=4):
            gr.Markdown("<div class='alfa-topbar'><b>Alfa-Code v1</b> &nbsp;·&nbsp; "
                        "Dashboard &nbsp;·&nbsp; Help</div>")
            gr.HTML("<div class='alfa-orb'></div>")
            gr.Markdown("<div class='alfa-hero-title'>WELCOME BACK</div>")
            gr.Markdown("<div class='alfa-hero-sub'>Bring your ideas to life today</div>")
            with gr.Row():
                prompt = gr.Textbox(label="", placeholder="✨ Ask me anything...",
                                    lines=3, elem_classes="alfa-prompt", scale=5)
            with gr.Row():
                deep = gr.Checkbox(label="🧠 Deep Think (longer answer)", value=False)
                btn = gr.Button("Generate ⏎", variant="primary",
                                elem_classes="alfa-send")
            with gr.Row():
                img = gr.Image(label="Image", type="filepath")
                fil = gr.File(label="File")
                vid = gr.Video(label="Video")
            status = gr.Label(value="[Idle]", label="Reasoning status")
            chat = gr.Chatbot(label="History", height=480)
            with gr.Row():
                card_code = gr.Button("🧩 Code Generator\nWrite python code", elem_classes="alfa-card")
                card_image = gr.Button("🖼️ Image → Code\nBuild UI from screenshot",
                                       elem_classes="alfa-card")
                card_video = gr.Button("🎬 Video → Code\nDescribe clip + render code",
                                       elem_classes="alfa-card")

    demo.load(fn=_engine_info, inputs=[], outputs=[engine_md])
    btn.click(fn=_chat_turn,
              inputs=[prompt, chat, img, fil, vid, deep],
              outputs=[status, chat])
    new_chat.click(fn=_clear, inputs=[], outputs=[chat, prompt, status])
    nav_code.click(fn=lambda: TEMPLATES["code"], inputs=[], outputs=[prompt])
    nav_image.click(fn=lambda: TEMPLATES["image"], inputs=[], outputs=[prompt])
    nav_video.click(fn=lambda: TEMPLATES["video"], inputs=[], outputs=[prompt])
    nav_files.click(fn=lambda: TEMPLATES["file"], inputs=[], outputs=[prompt])
    card_code.click(fn=lambda: TEMPLATES["code"], inputs=[], outputs=[prompt])
    card_image.click(fn=lambda: TEMPLATES["image"], inputs=[], outputs=[prompt])
    card_video.click(fn=lambda: TEMPLATES["video"], inputs=[], outputs=[prompt])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1",
                server_port=int(os.getenv("ALFA_PORT", "7860")),
                share=os.getenv("ALFA_SHARE", "0") == "1")
