from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out'; OUT.mkdir(parents=True,exist_ok=True)
NEG='photo, photorealistic, anime, manga, cartoon, 3d, child, teenager, text, captions, watermark, collage, split screen, neon, fantasy, deformed hands, extra fingers, smooth plastic skin'
items=[
(2,102018,'Traditional oil painting on canvas, thick visible impasto bristle strokes, luminous Renaissance pigments, not photo. Mature 38-year-old Western man, short curly dark hair, light stubble, business casual, at an ordinary apartment kitchen table checking his phone after a raise; groceries, paid envelope, keys. Cobalt, honey gold, emerald, terracotta, cream. Cinematic 16:9.'),
(18,118162,'Traditional oil painting on canvas, thick visible impasto bristle strokes, luminous jewel colors, not photo. Mature 38-year-old Western man, short curly dark hair, light stubble, at a real hospital billing counter holding wallet and medical folder; ordinary waiting chairs behind him, restrained concern. Ultramarine, vermilion, emerald, gold, cream, warm flesh. Cinematic 16:9.'),
(74,174666,'Traditional oil painting on canvas, textured bristle strokes and impasto highlights, not photo. Mature 38-year-old Western man driving a believable everyday car over a rough city street toward work; ordinary traffic and storefronts, subtle financial-cushion metaphor. Cobalt against warm yellow, burnt orange, viridian, turquoise, amber. Cinematic 16:9.'),
(140,241260,'Traditional oil painting on canvas, visible layered brushwork, luminous Impressionist daylight, not photo. Mature 38-year-old Western man with short curly dark hair and stubble sitting in a sunlit neighborhood cafe with notebook and coffee, calmly considering real choices. Sky blue, spring green, butter yellow, lavender, terracotta, ultramarine. Cinematic 16:9.')]
print('loading',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('Lykon/dreamshaper-8',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type='dpmsolver++',use_karras_sigmas=True)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
paths=[]
for sid,seed,prompt in items:
 ids=pipe.tokenizer(prompt,truncation=True,max_length=77,return_tensors=None)['input_ids']; prompt=pipe.tokenizer.decode(ids,skip_special_tokens=True)
 print('render',sid,'tokens',len(ids),flush=True)
 g=torch.Generator(device='cpu').manual_seed(seed)
 im=pipe(prompt=prompt,negative_prompt=NEG,width=640,height=360,num_inference_steps=12,guidance_scale=6.5,generator=g).images[0].convert('RGB')
 im=Image.blend(im,ImageOps.posterize(im,7),0.05)
 im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Brightness(im).enhance(1.04); im=ImageEnhance.Contrast(im).enhance(1.02)
 im=im.resize((1920,1080),Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.1,percent=65,threshold=4))
 p=OUT/f'shot_{sid:03d}_quality.jpg'; im.save(p,quality=94,subsampling=0,optimize=True); paths.append(p)
sheet=Image.new('RGB',(960,540*2),(245,242,235));d=ImageDraw.Draw(sheet)
for k,p in enumerate(paths):
 im=Image.open(p).resize((480,270),Image.Resampling.LANCZOS);x=(k%2)*480;y=(k//2)*540
 sheet.paste(im,(x,y));d.text((x+10,y+276),p.stem,fill=(20,20,20))
sheet.save(OUT/'quality_contact.jpg',quality=94)
print('done',flush=True)
