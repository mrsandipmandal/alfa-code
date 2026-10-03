"""Coding AI mode UI: reasoning (live->collapsed) + image + file + video."""
import time, gradio as gr
from pathlib import Path

def mock_reason(prompt, image, file, video):
    steps = []
    ctx = []
    if image: ctx.append(f"image: {Path(image).name}")
    if file: ctx.append(f"file: {Path(file).name if isinstance(file,str) else 'uploaded'}")
    if video: ctx.append(f"video: {Path(video).name} (4 frames extracted)")
    header = f"Context: {', '.join(ctx)}\n" if ctx else ""
    yield "[Thinking...]", f"{header}Understanding: {prompt[:120]}..."
    time.sleep(0.4)
    yield "[Planning...]", f"{header}Plan: 1.parse 2.generate 3.verify"
    time.sleep(0.4)
    code = f"```python\n# generated for: {prompt[:60]}\ndef solve():\n    return 'todo: wire to outputs/tiny-50M.gguf'\n```"
    yield "[Coding...]", f"{header}{code}"
    time.sleep(0.3)
    yield "[Done]", f"{header}<details><summary>Reasoning (3 steps)</summary>Understand > Plan > Code</details>\n\n{code}"

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
    btn.click(fn=mock_reason, inputs=[prompt, img, fil, vid], outputs=[status, chat])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
