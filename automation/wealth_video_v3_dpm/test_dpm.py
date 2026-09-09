from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(parents=True,exist_ok=True)
items=[
(1,2001,'cinematic oil painting on canvas, visible impasto brush strokes, rich luminous pigments. A dark-haired Western man in his late thirties in business-casual clothing sits at a real office desk reading a salary raise notification on his laptop, restrained relief, coffee mug, notebook, houseplant, daylight window. cobalt blue, honey gold, olive green, terracotta. believable everyday life, realistic anatomy, no text, no women, no painting easel'),
(18,2018,'cinematic oil painting on canvas, visible impasto brush strokes, rich jewel pigments. A dark-haired Western man in his late thirties stands at a hospital billing or pharmacy counter holding a medical folder and wallet, calm waiting area behind him, concerned but restrained. ultramarine, emerald green, cream, terracotta, gold. realistic healthcare expense scene, no text'),
(140,2140,'cinematic oil painting on canvas, visible impasto brush strokes, bright airy Impressionist light. A dark-haired Western man in his late thirties sits in a sunlit cafe with a notebook and phone face down, looking outside toward a tree-lined street and ordinary family life, hopeful realistic choice. sky blue, spring green, butter yellow, lavender, terracotta, gold. no text'),
]
pipe=StableDiffusionPipeline.from_pretrained('Lykon/dreamshaper-8',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
paths=[]
neg='photo, photorealistic, 3d render, anime, child, text, subtitles, readable words, logo, watermark, collage, split screen, fantasy portal, giant machinery, deformed hands, extra fingers, grayscale, red-blue duotone'
for sid,seed,prompt in items:
    g=torch.Generator(device='cpu').manual_seed(seed)
    im=pipe(prompt=prompt,negative_prompt=neg,width=768,height=432,num_inference_steps=20,guidance_scale=7.0,generator=g).images[0].convert('RGB')
    im=im.resize((1920,1080),Image.Resampling.LANCZOS)
    im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Contrast(im).enhance(1.03); im=im.filter(ImageFilter.UnsharpMask(1.0,100,2))
    p=OUT/f'shot_{sid:03d}_dpm.jpg'; im.save(p,quality=94); paths.append(p)
thumb=(640,360); sheet=Image.new('RGB',(thumb[0]*3,thumb[1]),(245,242,235)); d=ImageDraw.Draw(sheet)
for i,p in enumerate(paths):
    im=Image.open(p).resize(thumb,Image.Resampling.LANCZOS); x=i*thumb[0]; sheet.paste(im,(x,0)); d.rectangle((x+10,10,x+60,42),fill='black'); d.text((x+23,18),str(items[i][0]),fill='white')
sheet.save(OUT/'dpm_contact.jpg',quality=92)
print('done')