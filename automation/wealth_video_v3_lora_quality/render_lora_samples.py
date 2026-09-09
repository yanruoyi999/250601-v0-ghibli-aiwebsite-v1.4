from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out'; OUT.mkdir(parents=True,exist_ok=True)
NEG='photo, photorealistic, anime, manga, cartoon, 3d, child, teenager, text, caption, watermark, collage, split screen, neon, fantasy, deformed hands, extra fingers, smooth plastic skin'
items=[
(2,102018,'oil painting, mature 38-year-old Western man, short curly dark hair, light stubble, business casual, ordinary apartment kitchen table, checking phone after a raise, groceries paid envelope keys, cobalt honey gold emerald terracotta cream, cinematic everyday life'),
(18,118162,'oil painting, mature 38-year-old Western man, short curly dark hair, light stubble, real hospital billing counter, wallet and medical folder, ordinary waiting chairs, restrained concern, ultramarine vermilion emerald gold cream warm flesh, cinematic everyday life'),
(74,174666,'oil painting, mature 38-year-old Western man driving a believable everyday car over rough city street toward work, ordinary traffic storefronts, cobalt warm yellow burnt orange viridian turquoise amber, subtle real-world financial cushion metaphor'),
(140,241260,'oil painting, mature 38-year-old Western man, short curly dark hair, light stubble, sunlit neighborhood cafe, notebook and coffee, calm real choices, sky blue spring green butter yellow lavender terracotta ultramarine, luminous daylight')]
print('loading',flush=True)
pipe=StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.load_lora_weights('rocifier/painterly')
try: pipe.fuse_lora(lora_scale=0.75)
except Exception as e: print('fuse warning',e,flush=True)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type='dpmsolver++',use_karras_sigmas=True)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
paths=[]
for sid,seed,prompt in items:
 ids=pipe.tokenizer(prompt,truncation=True,max_length=77,return_tensors=None)['input_ids'];prompt=pipe.tokenizer.decode(ids,skip_special_tokens=True)
 print('render',sid,'tokens',len(ids),flush=True)
 g=torch.Generator(device='cpu').manual_seed(seed)
 im=pipe(prompt=prompt,negative_prompt=NEG,width=640,height=360,num_inference_steps=12,guidance_scale=7.0,generator=g).images[0].convert('RGB')
 im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Brightness(im).enhance(1.04); im=ImageEnhance.Contrast(im).enhance(1.02)
 im=im.resize((1920,1080),Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.1,percent=65,threshold=4))
 p=OUT/f'shot_{sid:03d}_lora.jpg'; im.save(p,quality=94,subsampling=0,optimize=True);paths.append(p)
sheet=Image.new('RGB',(960,600),(245,242,235));d=ImageDraw.Draw(sheet)
for k,p in enumerate(paths):
 im=Image.open(p).resize((480,270),Image.Resampling.LANCZOS);x=(k%2)*480;y=(k//2)*300
 sheet.paste(im,(x,y));d.text((x+10,y+274),p.stem,fill=(20,20,20))
sheet.save(OUT/'lora_quality_contact.jpg',quality=94)
print('done',flush=True)
