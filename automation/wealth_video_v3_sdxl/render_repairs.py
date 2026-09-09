from pathlib import Path
import json,sys,torch
from diffusers import AutoPipelineForText2Image
from PIL import Image,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parent
PROMPTS=ROOT.parent/'wealth_video_v3_sd_turbo'/'repair_prompts.json'
OUT=ROOT/'out_repairs'; OUT.mkdir(parents=True,exist_ok=True)
chunk=int(sys.argv[1]); items=json.loads(PROMPTS.read_text(encoding='utf-8')); items=[x for n,x in enumerate(items) if n%10==chunk]
print('loading sdxl-turbo',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe.enable_vae_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
STYLE='Cinematic narrative oil painting on canvas, visible impasto brush strokes, believable everyday architecture and realistic adult anatomy. Rich luminous pigments with cobalt and ultramarine blue, honey gold, emerald and olive green, terracotta, cream, small lavender or cyan accents; colored shadows, bright midtones, vivid but controlled saturation. No collage, no split screen, no fantasy portal, no surreal machinery, no watermark, no readable words or numbers.'
for x in items:
 i=int(x['id']); prompt=x['scene']+'. '+STYLE
 g=torch.Generator(device='cpu').manual_seed(770000+i*65537)
 im=pipe(prompt=prompt,width=640,height=384,num_inference_steps=2,guidance_scale=0.0,generator=g).images[0].convert('RGB')
 top=(im.height-360)//2; im=im.crop((0,top,640,top+360)).resize((1920,1080),Image.Resampling.LANCZOS)
 im=ImageEnhance.Color(im).enhance(1.12); im=ImageEnhance.Contrast(im).enhance(1.04); im=ImageEnhance.Brightness(im).enhance(1.03); im=im.filter(ImageFilter.UnsharpMask(1.0,95,2))
 p=OUT/f'shot_{i:03d}.jpg'; im.save(p,quality=94,subsampling=0); print('saved',p.name,flush=True)
