import gradio as gr

from lesionnet.config import CLASS_NAMES
from lesionnet.predict import predict_full


def build_ui() -> gr.Blocks:
    cache = {"composite": None}

    with gr.Blocks(title="LesionNet") as demo:
        gr.Markdown("# LesionNet — Skin Lesion Classifier with Grad-CAM")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Upload lesion image")
                gender = gr.Radio(
                    choices=["Male", "Female", "Prefer not to say"],
                    label="Gender",
                    value="Prefer not to say",
                )
                age = gr.Number(label="Age", minimum=0, maximum=130, step=1)
                submit = gr.Button("Submit", variant="primary")
                download = gr.DownloadButton("Download result")
            with gr.Column():
                label = gr.Label(label="Prediction", num_top_classes=7)
                overlay = gr.Image(type="pil", label="Grad-CAM overlay")

        def on_submit(image, gender, age):
            if image is None:
                raise gr.Error("Please upload a lesion image first.")
            if age is not None and not 0 <= age <= 130:
                raise gr.Error("Age must be between 0 and 130.")
            result = predict_full(image, gender, age)
            cache["composite"] = result["composite_path"]
            probs = {c: float(p) for c, p in zip(CLASS_NAMES, result["probs"])}
            return probs, result["overlay"]

        def on_download():
            path = cache["composite"]
            if path is None:
                raise gr.Error("Submit a prediction first before downloading.")
            return path

        submit.click(on_submit, inputs=[image, gender, age], outputs=[label, overlay])
        download.click(on_download, inputs=[], outputs=[download])

    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port)