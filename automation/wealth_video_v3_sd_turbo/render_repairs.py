from pathlib import Path
import json,sys,torch
from diffusers import AutoPipelineForText2Image
from PIL import Image,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out_repairs'; OUT.mkdir(parents=True,exist_ok=True)
chunk=int(sys.argv[1]); items=json.loads((ROOT/'repair_prompts.json').read_text(encoding='utf-8'))
items=[x for n,x in enumerate(items) if n%10==chunk]
print('loading sd-turbo',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
STYLE='Cinematic narrative oil painting on canvas, visible impasto brush strokes, realistic anatomy and believable everyday space, luminous saturated pigments, colored shadows, rich cobalt blue, honey gold, emerald and olive green, terracotta, cream, small lavender accents. No collage, no split screen, no fantasy portal, no surreal machinery, no watermark.'
for x in items:
 i=int(x['id']); prompt=x['scene']+'. '+STYLE
 g=torch.Generator(device='cpu').manual_seed(990000+i*104729)
 im=pipe(prompt=prompt,width=768,height=432,num_inference_steps=2,guidance_scale=0.0,generator=g).images[0].convert('RGB')
 im=im.resize((1920,1080),Image.Resampling.LANCZOS); im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Contrast(im).enhance(1.03); im=ImageEnhance.Brightness(im).enhance(1.02); im=im.filter(ImageFilter.UnsharpMask(1.1,105,2))
 p=OUT/f'shot_{i:03d}.jpg'; im.save(p,quality=94,subsampling=0); print('saved',p.name,flush=True)
