import gradio as gr

from lesionnet.config import CLASS_NAMES
from lesionnet.predict import predict_full


APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

APP_CSS = r"""
:root {
  --void:#0A0A0C; --panel:#131417; --panel-raised:#191A1E;
  --hairline:#26272C; --hairline-bright:#38393F;
  --text:#EDEDEF; --text-dim:#8A8B92; --text-faint:#55565C;
  --amber:#FFB000; --amber-dim:#5C4210; --amber-glow:rgba(255,176,0,0.14);
}

* { box-sizing:border-box; }
body, .gradio-container {
  background-color:var(--void) !important;
  background-image:
    radial-gradient(ellipse 900px 500px at 15% -10%, rgba(255,176,0,0.05), transparent 60%),
    radial-gradient(ellipse 700px 500px at 100% 10%, rgba(255,176,0,0.03), transparent 60%);
  color:var(--text) !important;
  font-family:'Space Grotesk',sans-serif !important;
}
.gradio-container { max-width:none !important; padding:28px 32px 60px !important; }
.gradio-container button, .gradio-container input { font-family:'Space Grotesk',sans-serif !important; }
.main-shell { width:100%; max-width:1240px; margin:0 auto; }

.les-header {
  display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
  padding-bottom:18px; margin-bottom:22px; border-bottom:1px solid var(--hairline);
}
.brand { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.brand h1 { margin:0; color:var(--text); font-size:22px; font-weight:700; letter-spacing:.04em; }
.brand .tag, .status, .panel-head, .field-label, .headline-sub, .meta-row, .les-footer {
  font-family:'IBM Plex Mono',monospace !important;
}
.brand .tag { color:var(--text-faint); font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; }
.status { display:flex; align-items:center; gap:8px; color:var(--text-dim); font-size:11px; letter-spacing:.03em; white-space:nowrap; }
.status .dot { width:7px; height:7px; border-radius:50%; background:#33D17A; box-shadow:0 0 8px #33D17A; animation:pulse 2.2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }

.main-grid { width:100%; max-width:1240px; margin:0 auto; gap:20px !important; align-items:start; }
.left-column, .right-column { min-width:0 !important; }
.panel { background:var(--panel) !important; border:1px solid var(--hairline) !important; border-radius:3px !important; overflow:hidden !important; padding:0 !important; gap:0 !important; box-shadow:none !important; }
.panel + .panel { margin-top:20px; }
.panel-head {
  display:flex; justify-content:space-between; align-items:center; gap:18px;
  min-height:42px; padding:12px 16px; border-bottom:1px solid var(--hairline);
  color:var(--text-dim); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
}
.panel-head b { color:var(--text); font-weight:500; letter-spacing:.12em; }
.panel-body { padding:20px !important; gap:0 !important; }

.reticle { position:relative !important; overflow:hidden !important; background:#000 !important; border:1px solid #000 !important; border-radius:2px !important; padding:0 !important; min-height:0 !important; }
.reticle .image-container { position:relative !important; width:100% !important; min-height:0 !important; aspect-ratio:1/1 !important; background:#000 !important; border:0 !important; border-radius:2px !important; overflow:hidden !important; }
.overlay-reticle .image-container { aspect-ratio:16/9 !important; }
.reticle img { width:100% !important; height:100% !important; object-fit:cover !important; display:block !important; }
.reticle::before, .reticle::after, .reticle .image-container::before, .reticle .image-container::after {
  content:""; position:absolute; width:18px; height:18px; z-index:8; pointer-events:none;
  border-color:var(--amber); opacity:.85;
}
.reticle::before { top:8px; left:8px; border-top:2px solid var(--amber); border-left:2px solid var(--amber); }
.reticle::after { top:8px; right:8px; border-top:2px solid var(--amber); border-right:2px solid var(--amber); }
.reticle .image-container::before { bottom:8px; left:8px; border-bottom:2px solid var(--amber); border-left:2px solid var(--amber); }
.reticle .image-container::after { bottom:8px; right:8px; border-bottom:2px solid var(--amber); border-right:2px solid var(--amber); }
.reticle .image-container button, .reticle button { border-radius:2px !important; border-color:var(--hairline) !important; background:rgba(19,20,23,.92) !important; color:var(--text-dim) !important; box-shadow:none !important; }
.reticle .image-container button:hover, .reticle button:hover { border-color:var(--hairline-bright) !important; color:var(--text) !important; }
.reticle-tag {
  margin-top:8px; color:var(--amber); background:transparent; font-family:'IBM Plex Mono',monospace !important;
  font-size:9.5px; letter-spacing:.08em; text-transform:uppercase;
}

.input-fields { margin-top:20px !important; gap:20px !important; }
.field-label { display:block; margin-bottom:8px; color:var(--text-faint) !important; font-size:10px !important; letter-spacing:.12em; text-transform:uppercase; }
.seg fieldset, .seg .wrap, .seg [role="radiogroup"] { display:flex !important; width:100% !important; gap:0 !important; }
.seg fieldset { border:1px solid var(--hairline) !important; border-radius:2px !important; overflow:hidden !important; background:var(--panel-raised) !important; }
.seg label { flex:1 1 0; min-height:40px; margin:0 !important; padding:0 8px !important; border:0 !important; border-right:1px solid var(--hairline) !important; border-radius:0 !important; background:var(--panel-raised) !important; color:var(--text-dim) !important; font-size:13px !important; display:flex !important; align-items:center; justify-content:center; cursor:pointer; }
.seg label:last-child { border-right:0 !important; }
.seg label:hover { color:var(--text) !important; }
.seg label:has(input:checked) { background:var(--amber-glow) !important; color:var(--amber) !important; font-weight:600 !important; }
.seg input { accent-color:var(--amber) !important; }

.stepper { min-height:40px; border:1px solid var(--hairline) !important; border-radius:2px !important; background:var(--panel-raised) !important; overflow:hidden !important; }
.stepper input { min-height:40px !important; background:transparent !important; border:0 !important; color:var(--text) !important; font-family:'IBM Plex Mono',monospace !important; font-size:15px !important; padding:9px 12px !important; box-shadow:none !important; }
.stepper button { min-width:34px !important; min-height:40px !important; background:transparent !important; border:0 !important; border-left:1px solid var(--hairline) !important; border-radius:0 !important; color:var(--text-dim) !important; box-shadow:none !important; }
.stepper button:hover { color:var(--amber) !important; background:transparent !important; }

.run-btn { width:100%; min-height:48px !important; margin-top:26px !important; padding:14px !important; border:0 !important; border-radius:2px !important; background:var(--amber) !important; color:#1A1200 !important; font-weight:700 !important; font-size:14px !important; letter-spacing:.04em !important; box-shadow:0 0 0 1px rgba(255,176,0,.4), 0 8px 24px -8px rgba(255,176,0,.35) !important; transition:transform .12s, box-shadow .12s !important; }
.run-btn:hover { transform:translateY(-1px) !important; box-shadow:0 0 0 1px rgba(255,176,0,.6), 0 10px 28px -6px rgba(255,176,0,.45) !important; }
.run-btn:active { transform:translateY(0) !important; }
.dl-btn { width:100%; min-height:44px !important; margin-top:12px !important; padding:12px !important; border-radius:2px !important; background:transparent !important; border:1px solid var(--hairline) !important; color:var(--text-dim) !important; font-family:'IBM Plex Mono',monospace !important; font-size:11.5px !important; letter-spacing:.06em !important; text-transform:uppercase; box-shadow:none !important; }
.dl-btn:hover { border-color:var(--hairline-bright) !important; color:var(--text) !important; background:transparent !important; }

.readout { min-height:0; }
.headline { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; }
.headline .cls { color:var(--text); font-size:44px; line-height:.95; font-weight:700; letter-spacing:.01em; }
.headline .conf { color:var(--amber); font-family:'IBM Plex Mono',monospace; font-size:30px; line-height:1; font-weight:600; white-space:nowrap; }
.headline-sub { margin-top:8px; color:var(--text-faint); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; }
.bars { margin-top:26px; display:flex; flex-direction:column; gap:12px; }
.bar-row { display:grid; grid-template-columns:52px 1fr 48px; align-items:center; gap:12px; }
.bar-row .code, .bar-row .pct { font-family:'IBM Plex Mono',monospace; font-size:12px; }
.bar-row .code { color:var(--text-dim); }
.bar-row.top .code { color:var(--amber); font-weight:600; }
.bar-track { height:7px; background:var(--panel-raised); border:1px solid var(--hairline); border-radius:1px; overflow:hidden; position:relative; }
.bar-fill { height:100%; width:0; background:var(--text-faint); border-radius:1px; animation:bar-grow 1.1s cubic-bezier(.2,.8,.2,1) forwards; }
.bar-row.top .bar-fill { background:linear-gradient(90deg,#7A5200,var(--amber)); box-shadow:0 0 10px rgba(255,176,0,.4); }
.bar-row .pct { color:var(--text-dim); text-align:right; }
.bar-row.top .pct { color:var(--amber); font-weight:600; }
@keyframes bar-grow { from { width:0 !important; } }
.divider { height:1px; margin:22px 0; background:var(--hairline); }
.meta-row { display:flex; flex-wrap:wrap; gap:12px 24px; color:var(--text-faint); font-size:10.5px; letter-spacing:.05em; }
.meta-row b { color:var(--text-dim); font-weight:500; }

.les-footer { max-width:1240px; margin:26px auto 0; padding-top:16px; border-top:1px solid var(--hairline); display:flex; justify-content:space-between; gap:20px; color:var(--text-faint); font-size:10px; letter-spacing:.06em; }
button:focus-visible, input:focus-visible, [tabindex]:focus-visible { outline:2px solid var(--amber) !important; outline-offset:2px !important; }

@media (max-width:920px) {
  .gradio-container { padding:20px 16px 40px !important; }
  .les-header { align-items:flex-start; flex-direction:column; }
  .status { white-space:normal; }
  .main-grid { flex-direction:column !important; }
  .left-column, .right-column { width:100% !important; flex:none !important; }
}

@media (prefers-reduced-motion:reduce) {
  *, *::before, *::after { animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; }
  .status .dot { animation:none !important; }
}
"""

APP_JS = r"""
(() => {
  const animateConfidence = () => {
    document.querySelectorAll('.confidence-value[data-target]').forEach((el) => {
      if (el.dataset.animated === '1') return;
      el.dataset.animated = '1';
      const target = Number(el.dataset.target || '0');
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        el.textContent = `${target.toFixed(1)}%`;
        return;
      }
      let value = 0;
      const step = Math.max(target / 35, 0.5);
      const timer = window.setInterval(() => {
        value = Math.min(target, value + step);
        el.textContent = `${value.toFixed(1)}%`;
        if (value >= target) window.clearInterval(timer);
      }, 30);
    });
  };
  const observer = new MutationObserver(animateConfidence);
  observer.observe(document.body, {subtree:true, childList:true});
  animateConfidence();
})();
"""

APP_THEME = gr.themes.Base(
    font=gr.themes.GoogleFont("Space Grotesk", weights=(400, 500, 600, 700)),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono", weights=(400, 500, 600)),
).set(
    body_background_fill="#0A0A0C",
    body_background_fill_dark="#0A0A0C",
    body_text_color="#EDEDEF",
    body_text_color_dark="#EDEDEF",
    body_text_color_subdued="#8A8B92",
    body_text_color_subdued_dark="#8A8B92",
    background_fill_primary="#131417",
    background_fill_primary_dark="#131417",
    background_fill_secondary="#191A1E",
    background_fill_secondary_dark="#191A1E",
    border_color_primary="#26272C",
    border_color_primary_dark="#26272C",
    border_color_accent="#38393F",
    border_color_accent_dark="#38393F",
    block_background_fill="#131417",
    block_background_fill_dark="#131417",
    block_border_color="#26272C",
    block_border_color_dark="#26272C",
    block_radius="3px",
    block_shadow="none",
    block_shadow_dark="none",
    input_background_fill="#191A1E",
    input_background_fill_dark="#191A1E",
    input_border_color="#26272C",
    input_border_color_dark="#26272C",
    input_border_color_focus="#FFB000",
    input_border_color_focus_dark="#FFB000",
    button_primary_background_fill="#FFB000",
    button_primary_background_fill_dark="#FFB000",
    button_primary_background_fill_hover="#FFB000",
    button_primary_background_fill_hover_dark="#FFB000",
    button_primary_border_color="#FFB000",
    button_primary_border_color_dark="#FFB000",
    button_primary_text_color="#1A1200",
    button_primary_text_color_dark="#1A1200",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_border_color="#26272C",
    button_secondary_border_color_dark="#26272C",
    button_secondary_text_color="#8A8B92",
    button_secondary_text_color_dark="#8A8B92",
)


def _render_readout(probs: dict[str, float]) -> str:
    top_class = max(probs, key=probs.get)
    top_conf = probs[top_class] * 100.0
    ordered = sorted(
        ((c, float(probs.get(c, 0.0))) for c in CLASS_NAMES),
        key=lambda item: (-item[1], item[0]),
    )
    rows = []
    for class_name, prob in ordered:
        confidence = prob * 100.0
        modifier = " top" if class_name == top_class else ""
        rows.append(
            f'''<div class="bar-row{modifier}">
                <span class="code">{class_name}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{confidence:.1f}%;"></div></div>
                <span class="pct">{confidence:.1f}%</span>
            </div>'''
        )
    return f'''<div class="readout">
        <div class="headline">
          <div class="cls">{top_class}</div>
          <div class="conf confidence-value" data-target="{top_conf:.1f}">0.0%</div>
        </div>
        <div class="headline-sub">Top prediction · classification confidence</div>
        <div class="bars">{"".join(rows)}</div>
        <div class="divider"></div>
        <div class="meta-row">
          <span><b>MODEL</b> EfficientNet-B4</span>
          <span><b>CLASSES</b> {len(CLASS_NAMES)}</span>
          <span><b>XAI</b> Grad-CAM</span>
        </div>
    </div>'''


def build_ui() -> gr.Blocks:
    cache = {"composite": None}

    with gr.Blocks(title="LesionNet") as demo:
        with gr.Column(elem_classes=["main-shell"]):
            gr.HTML('''<header class="les-header">
                <div class="brand">
                  <h1>LESIONNET</h1>
                  <span class="tag">Dermoscopic Classification · Grad-CAM Explainability</span>
                </div>
                <div class="status"><span class="dot"></span> EFFICIENTNET-B4 · MODEL LOADED</div>
            </header>''')

            with gr.Row(elem_classes=["main-grid"]):
                with gr.Column(elem_classes=["left-column"]):
                    with gr.Column(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-head"><b>01 · Specimen Input</b><span>224×224 · RGB</span></div>')
                        with gr.Column(elem_classes=["panel-body"]):
                            image = gr.Image(
                                type="pil",
                                label="Upload lesion image",
                                show_label=False,
                                sources=["upload", "webcam", "clipboard"],
                                elem_classes=["reticle", "specimen-reticle"],
                            )
                            gr.HTML('<div class="reticle-tag">SPECIMEN INPUT</div>')

                            with gr.Column(elem_classes=["input-fields"]):
                                gr.HTML('<span class="field-label">Sex</span>')
                                gender = gr.Radio(
                                    choices=["Male", "Female", "Prefer not to say"],
                                    label="Sex",
                                    value="Prefer not to say",
                                    show_label=False,
                                    elem_classes=["seg"],
                                )
                                gr.HTML('<span class="field-label">Age</span>')
                                age = gr.Number(
                                    label="Age",
                                    minimum=0,
                                    maximum=130,
                                    step=1,
                                    show_label=False,
                                    elem_classes=["stepper"],
                                )

                            submit = gr.Button("▸ RUN ANALYSIS", variant="primary", elem_classes=["run-btn"])

                with gr.Column(elem_classes=["right-column"]):
                    with gr.Column(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-head"><b>02 · Diagnostic Readout</b><span>7-class probability readout</span></div>')
                        with gr.Column(elem_classes=["panel-body"]):
                            label = gr.HTML(
                                value='''<div class="readout"><div class="headline"><div class="cls">—</div><div class="conf">—</div></div><div class="headline-sub">Run an analysis to populate the diagnostic readout</div></div>''',
                                elem_classes=["readout-output"],
                            )

                    with gr.Column(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-head"><b>03 · Grad-CAM Overlay</b><span>explainability map</span></div>')
                        with gr.Column(elem_classes=["panel-body"]):
                            overlay = gr.Image(
                                type="pil",
                                label="Grad-CAM overlay",
                                show_label=False,
                                interactive=False,
                                elem_classes=["reticle", "overlay-reticle"],
                            )
                            download = gr.DownloadButton(
                                "⬇ DOWNLOAD FULL REPORT (.PNG)",
                                variant="secondary",
                                elem_classes=["dl-btn"],
                            )

            gr.HTML('''<footer class="les-footer">
                <span>LESIONNET · DIAGNOSTIC INSTRUMENT</span>
                <span>PRELIMINARY SCREENING · NOT A MEDICAL DIAGNOSIS</span>
            </footer>''')

        def on_submit(image, gender, age):
            if image is None:
                raise gr.Error("Please upload a lesion image first.")
            if age is not None and not 0 <= age <= 130:
                raise gr.Error("Age must be between 0 and 130.")
            result = predict_full(image, gender, age)
            cache["composite"] = result["composite_path"]
            probs = {c: float(p) for c, p in zip(CLASS_NAMES, result["probs"])}
            return _render_readout(probs), result["overlay"]

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
    demo.launch(
        theme=APP_THEME,
        css=APP_CSS,
        head=APP_HEAD,
        js=APP_JS,
        server_name=server_name,
        server_port=server_port,
    )
