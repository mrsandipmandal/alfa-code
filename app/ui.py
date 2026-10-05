"""Coding AI mode UI: reasoning (live->collapsed) + image + file + video.

Generation is served by app/model_backend.py: real trained checkpoint
when ALFA_MODEL is available, otherwise mock mode with the reason shown.
"""
import gradio as gr

try:
    # package mode: `from app.ui import demo` (Colab cell, python -m app.ui)
    from app.model_backend import generate
except ImportError:
    # script mode: `python app/ui.py` (script dir on sys.path)
    from model_backend import generate

with gr.Blocks(title="Alfa-Code") as demo:
    gr.Markdown("# Alfa-Code: reasoning + image + file + video")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Markdown()
            prompt = gr.Textbox(label="Prompt", placeholder="Describe code task...")
            with gr.Row():
                img = gr.Image(label="Image", type="filepath")
                fil = gr.File(label="File")
                vid = gr.Video(label="Video")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            status = gr.Label(value="Idle", label="Reasoning status")
    btn.click(fn=generate, inputs=[prompt, img, fil, vid], outputs=[status, chat])

if __name__ == "__main__":
    import os

    demo.launch(server_name="127.0.0.1",
                server_port=int(os.getenv("ALFA_PORT", "7860")),
                share=os.getenv("ALFA_SHARE", "0") == "1")
