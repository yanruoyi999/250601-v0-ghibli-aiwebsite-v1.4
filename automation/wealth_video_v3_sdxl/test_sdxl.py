from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(parents=True,exist_ok=True)
items=[
(1,3001,'cinematic narrative oil painting, visible impasto brush strokes. ONE dark-haired Western man in his late thirties, alone at a modern office desk, looking at a salary raise notification on his laptop with restrained relief. Coffee mug, notebook, green plant, daylight window. Rich cobalt blue, honey gold, olive green, terracotta and cream. Real everyday office, realistic anatomy, no other people, no text, no easel.'),
(10,3010,'cinematic narrative oil painting, visible impasto brush strokes. ONE dark-haired Western man in his late thirties, fully dressed in a casual shirt, alone at a kitchen table opening a household bill, with plain envelopes, calendar, keys and coffee. Rich cobalt, amber, olive green, terracotta and cream. Real apartment, realistic anatomy, no other people, no readable text, no bed.'),
(40,3040,'cinematic narrative oil painting, visible impasto brush strokes. ONE well-dressed dark-haired Western professional in his late thirties, alone in a tasteful but expensive apartment, quietly checking a pile of bills while a nice financed car is visible through the window. He looks successful but trapped. Rich ultramarine, emerald, gold, terracotta and warm cream. No other people, no text.'),
(90,3090,'cinematic narrative oil painting, visible impasto brush strokes. ONE dark-haired Western man in his late thirties, alone in a normal bedroom, calmly looking at practical emergency supplies: medical folder, flashlight, small tool kit, coat and packed bag. He realizes money cannot cover every possible future. Cobalt, amber, forest green, warm wood. No second person, no text.'),
(120,3120,'cinematic narrative oil painting, visible impasto brush strokes. A dark-haired Western man in his late thirties sits at a kitchen table with his visibly older gray-haired parent, helping pay for groceries and medicine without panic. Exactly two people of clearly different ages. Sky blue, spring green, warm gold, terracotta, cream. Real family scene, no text.'),
(140,3140,'cinematic narrative oil painting, visible impasto brush strokes, bright airy Impressionist light. ONE dark-haired Western man in his late thirties sits alone at a sunlit cafe table with a notebook and phone face down, looking outside toward a tree-lined street, park and ordinary family life, imagining what financial safety would let him do. Sky blue, spring green, butter yellow, lavender, terracotta and gold. No other foreground person, no text.'),
]

print('loading sdxl-turbo',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo',torch_dtype=torch.float32,variant=None,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe.enable_vae_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
paths=[]
for sid,seed,prompt in items:
    g=torch.Generator(device='cpu').manual_seed(seed)
    im=pipe(prompt=prompt,width=640,height=384,num_inference_steps=2,guidance_scale=0.0,generator=g).images[0].convert('RGB')
    # crop to 16:9 then upscale
    h=360; top=(im.height-h)//2; im=im.crop((0,top,640,top+h)).resize((1920,1080),Image.Resampling.LANCZOS)
    im=ImageEnhance.Color(im).enhance(1.06); im=ImageEnhance.Contrast(im).enhance(1.03); im=im.filter(ImageFilter.UnsharpMask(1.0,95,2))
    p=OUT/f'shot_{sid:03d}_sdxl.jpg'; im.save(p,quality=94); paths.append(p)
thumb=(480,270); sheet=Image.new('RGB',(thumb[0]*3,thumb[1]*2),(245,242,235)); d=ImageDraw.Draw(sheet)
for k,p in enumerate(paths):
    im=Image.open(p).resize(thumb,Image.Resampling.LANCZOS); x=(k%3)*thumb[0]; y=(k//3)*thumb[1]; sheet.paste(im,(x,y)); d.rectangle((x+7,y+7,x+47,y+30),fill='black'); d.text((x+12,y+10),str(items[k][0]),fill='white')
sheet.save(OUT/'sdxl_contact.jpg',quality=92)
print('done')