"""Alfa-Code dashboard UI — dark purple-black theme (Echo-style).

Layout: left sidebar (logo, New Chat, nav, history groups, upgrade card,
user row) + center (orb, welcome, headline, suggestion cards, prompt bar
with uploads, status, history). No brand names, no icon glyphs — text only.
Generation served by app/model_backend.py (real checkpoint when available,
else mock mode with reason).
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
.alfa-app { background: #08080a !important; }
.alfa-frame { background: #08080a; border: 1px solid rgba(255,255,255,0.10);
  border-radius: 24px; overflow: hidden; }
.alfa-side { background: #0d0d10 !important; border-right: 1px solid rgba(255,255,255,0.07);
  border-radius: 0px; padding: 14px 12px; min-height: 100%; }
.alfa-logo { color: #f2f2f4; font-size: 17px; font-weight: 700; }
.alfa-newchat button { width: 100%; background: transparent !important;
  border: 1px solid rgba(255,255,255,0.22) !important; color: #f2f2f4 !important;
  border-radius: 10px !important; font-weight: 600 !important; }
.alfa-nav button { background: transparent !important; border: none !important;
  color: #8a8a93 !important; text-align: left !important; font-size: 14px !important; }
.alfa-nav-active button { background: rgba(255,255,255,0.07) !important;
  border: none !important; color: #f2f2f4 !important; text-align: left !important;
  border-radius: 9px !important; font-size: 14px !important; }
.alfa-group { color: #6d6d76; font-size: 11px; letter-spacing: 1px; margin: 14px 0 4px 4px; }
.alfa-hist { color: #8a8a93; font-size: 13px; line-height: 2.0; padding-left: 4px; }
.alfa-pro { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 12px; padding: 10px; color: #b9b9c0; font-size: 12px; }
.alfa-pro button { width: 100%; background: transparent !important;
  border: 1px solid rgba(192,132,252,0.55) !important; color: #e9d5ff !important;
  border-radius: 9px !important; }
.alfa-user { color: #c9c9d1; font-size: 14px; padding: 8px 4px; }
.alfa-main { background:
  radial-gradient(900px 480px at 85% -5%, rgba(168,85,247,0.20) 0%, transparent 60%),
  radial-gradient(700px 420px at 100% 20%, rgba(192,132,252,0.12) 0%, transparent 55%),
  #08080a; padding: 28px 40px; }
.alfa-orb { width: 64px; height: 64px; margin: 34px auto 10px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #e9d5ff 0%, #c084fc 30%, #7e22ce 62%, #1a0b2e 100%);
  box-shadow: 0 0 46px rgba(168,85,247,0.55); }
.alfa-welcome { text-align: center; color: #8a8a93; font-size: 17px; }
.alfa-headline { text-align: center; color: #f4f4f6; font-size: 40px; font-weight: 700; margin: 2px 0 18px 0; }
.alfa-suggest button { background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 14px !important;
  color: #b9b9c0 !important; text-align: left !important; font-size: 13px !important;
  min-height: 96px !important; }
.alfa-bar textarea { background: #101014 !important;
  border: 1px solid rgba(168,85,247,0.55) !important; border-radius: 16px !important;
  color: #f2f2f4 !important; font-size: 14px !important;
  box-shadow: 0 0 26px rgba(168,85,247,0.22) !important; }
.alfa-send button { background: #18181d !important;
  border: 1px solid rgba(168,85,247,0.65) !important; color: #e9d5ff !important;
  border-radius: 10px !important; font-weight: 700 !important; }
.alfa-deep label { color: #8a8a93 !important; font-size: 13px !important; }
.alfa-status { color: #8a8a93; font-size: 12px; }
"""


def _engine_line():
    b = get_backend()
    if b["mode"] == "mock":
        return f"Engine: mock — {b['reason']}"
    return f"Engine: {b['mode']} — {b['device']}"


def _history_md(history):
    pairs = _to_pairs(history)
    if not pairs:
        return "<div class='alfa-hist'>No chats yet.</div>"
    lines = "".join(f"<div>{(u or '')[:34]}</div>" for u, _ in pairs[-8:])
    return f"<div class='alfa-hist'>{lines}</div>"


def _to_messages(pairs):
    """Gradio 5 Chatbot format: [{'role':..,'content':..}, ...]."""
    msgs = []
    for u, b in pairs:
        msgs.append({"role": "user", "content": u or ""})
        msgs.append({"role": "assistant", "content": b or ""})
    return msgs


def _to_pairs(messages):
    """Parse messages-format history back into [[user, bot], ...]."""
    messages = messages or []
    pairs = []
    i = 0
    while i < len(messages):
        m = messages[i]
        u = m.get("content", "") if isinstance(m, dict) else ""
        b = ""
        if i + 1 < len(messages) and isinstance(messages[i + 1], dict):
            b = messages[i + 1].get("content", "")
            i += 1
        pairs.append([u, b])
        i += 1
    return pairs


def _chat_turn(prompt, history, image, file, video, deep_think):
    pairs = _to_pairs(history)
    user_label = (prompt or "")[:200]
    n_tokens = 512 if deep_think else 256
    final_md = ""
    status = "[Thinking...]"
    for status, chat_md in generate(prompt, image, file, video,
                                    max_new_tokens=n_tokens):
        final_md = chat_md
        yield status, _to_messages(pairs + [[user_label, chat_md + "\n\n_...generating..._"]]), _history_md(pairs)
    done_pairs = pairs + [[user_label, final_md]]
    yield "[Done]", _to_messages(done_pairs), _history_md(done_pairs)


def _clear():
    return [], "", "[Idle]", "<div class='alfa-hist'>No chats yet.</div>"


TEMPLATES = {
    "code": "Write a Python function that ",
    "image": "<image>\nBuild this UI as a single HTML file: ",
    "video": "<video> <video> <video> <video>\nDescribe this clip, then write code to render it: ",
    "debug": "Find and fix the bug in this code:\n",
}


with gr.Blocks(title="AI Workspace", css=CSS, elem_classes="alfa-app") as demo:
    with gr.Row(elem_classes="alfa-frame"):
        # ---------- sidebar ----------
        with gr.Column(scale=1, elem_classes="alfa-side"):
            gr.Markdown("<div class='alfa-logo'>✦</div>")
            new_chat = gr.Button("+ New Chat", elem_classes="alfa-newchat")
            nav_chat = gr.Button("Chat", elem_classes="alfa-nav-active")
            nav_code = gr.Button("Code", elem_classes="alfa-nav")
            nav_image = gr.Button("Images", elem_classes="alfa-nav")
            nav_video = gr.Button("Videos", elem_classes="alfa-nav")
            gr.Markdown("<div class='alfa-group'>TODAY</div>")
            hist_md = gr.Markdown("<div class='alfa-hist'>No chats yet.</div>")
            gr.Markdown("<div class='alfa-pro'>Upgrade to Pro<br>"
                        "Get premium features, latest models and unlimited usage.<br><br>"
                        "<b>Pro Access</b></div>")
            gr.Markdown("<div class='alfa-user'>User</div>")
        # ---------- main ----------
        with gr.Column(scale=4, elem_classes="alfa-main"):
            gr.HTML("<div class='alfa-orb'></div>")
            gr.Markdown("<div class='alfa-welcome'>Welcome to AI Assistant</div>")
            gr.Markdown("<div class='alfa-headline'>How Can I Assist You?</div>")
            with gr.Row():
                sug1 = gr.Button("Write a todo list for my day",
                                 elem_classes="alfa-suggest")
                sug2 = gr.Button("Write a Python function for sorting",
                                 elem_classes="alfa-suggest")
                sug3 = gr.Button("Explain a code snippet",
                                 elem_classes="alfa-suggest")
            prompt = gr.Textbox(label="", placeholder="Ask AI anything or write your request...",
                                lines=3, elem_classes="alfa-bar")
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Image", type="filepath")
                with gr.Column(scale=1):
                    fil = gr.File(label="File")
                with gr.Column(scale=1):
                    vid = gr.Video(label="Video")
            with gr.Row():
                deep = gr.Checkbox(label="Deep Think (longer answer)", value=False,
                                   elem_classes="alfa-deep")
                btn = gr.Button("Generate", elem_classes="alfa-send")
            status = gr.Label(value="[Idle]", label="Reasoning status")
            gr.Markdown(f"<div class='alfa-status'>{_engine_line()}</div>")
            chat = gr.Chatbot(label="History", height=420)

    btn.click(fn=_chat_turn,
              inputs=[prompt, chat, img, fil, vid, deep],
              outputs=[status, chat, hist_md])
    new_chat.click(fn=_clear, inputs=[], outputs=[chat, prompt, status, hist_md])
    nav_chat.click(fn=lambda: "", inputs=[], outputs=[prompt])
    nav_code.click(fn=lambda: TEMPLATES["code"], inputs=[], outputs=[prompt])
    nav_image.click(fn=lambda: TEMPLATES["image"], inputs=[], outputs=[prompt])
    nav_video.click(fn=lambda: TEMPLATES["video"], inputs=[], outputs=[prompt])
    sug1.click(fn=lambda: "Write a todo list for my day", inputs=[], outputs=[prompt])
    sug2.click(fn=lambda: "Write a Python function for sorting", inputs=[], outputs=[prompt])
    sug3.click(fn=lambda: "Explain this code:\n", inputs=[], outputs=[prompt])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1",
                server_port=int(os.getenv("ALFA_PORT", "7860")),
                share=os.getenv("ALFA_SHARE", "0") == "1")
