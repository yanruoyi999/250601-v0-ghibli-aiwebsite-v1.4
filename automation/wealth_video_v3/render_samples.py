from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

NEG = (
    "text, subtitles, captions, readable numbers, logos, watermark, collage, split screen, storyboard, "
    "multi-panel layout, cyberpunk neon, giant glass containers, fantasy portals, floating UI, surreal machinery, "
    "black empty background, plastic 3D render, deformed hands, extra fingers, duplicate people, oversaturated neon"
)

print("loading model...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "Lykon/dreamshaper-8-lcm",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cpu")
torch.set_num_threads(2)

items = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
paths = []
for item in items:
    sid = int(item["shot_id"])
    generator = torch.Generator(device="cpu").manual_seed(int(item["seed"]))
    print("rendering shot", sid, flush=True)
    image = pipe(
        prompt=item["prompt"],
        negative_prompt=NEG,
        width=640,
        height=360,
        num_inference_steps=4,
        guidance_scale=2.0,
        generator=generator,
    ).images[0].convert("RGB")
    # Painterly content tolerates high-quality Lanczos enlargement well for review.
    image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    path = OUT / f"shot_{sid:03d}_sample.png"
    image.save(path, optimize=True)
    paths.append(path)

# Make one contact sheet for rapid visual review.
thumb_w, thumb_h = 640, 360
sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 2), (245, 242, 235))
d = ImageDraw.Draw(sheet)
for i, path in enumerate(paths):
    im = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    x = (i % 2) * thumb_w
    y = (i // 2) * thumb_h
    sheet.paste(im, (x, y))
    d.rectangle((x + 12, y + 12, x + 70, y + 54), fill=(20, 20, 20))
    d.text((x + 27, y + 19), str(items[i]["shot_id"]), fill="white")
sheet.save(OUT / "samples_contact.jpg", quality=94)
print("done", [str(p) for p in paths])
