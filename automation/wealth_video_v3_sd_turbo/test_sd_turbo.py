from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(parents=True,exist_ok=True)
items=[
(1,1001,'oil painting on canvas, visible brush strokes, rich luminous pigments, cinematic 16:9. A dark-haired Western man in his late thirties wearing business-casual clothes sits at a real office desk reading a salary raise notification on his laptop, restrained relief, coffee mug, notebook, green plant, daylight window, cobalt blue, honey gold, olive green, terracotta, believable everyday life, no text'),
(2,1002,'oil painting on canvas, visible brush strokes, rich luminous pigments, cinematic 16:9. The same type of dark-haired Western man in his late thirties sits on the edge of his bed late at night checking a banking app on his phone, amber bedside lamp, cobalt blue night outside the window, green houseplant, walnut wood, worried but calm, believable bedroom, no text'),
(18,1018,'oil painting on canvas, visible brush strokes, rich jewel pigments, cinematic 16:9. A dark-haired Western man in his late thirties at a hospital billing or pharmacy counter holding a medical folder and wallet, calm waiting area behind him, ultramarine, emerald green, cream, terracotta and gold, realistic healthcare expense scene, no text'),
(74,1074,'oil painting on canvas, visible brush strokes, rich complementary pigments, cinematic 16:9. A dark-haired Western man in his late thirties at a car repair garage while a mechanic shows him a worn suspension part and repair estimate folder, ordinary car behind them, cobalt blue, chrome yellow, burnt orange, viridian green, realistic practical expense scene, no text'),
(90,1090,'oil painting on canvas, visible brush strokes, rich luminous pigments, cinematic 16:9. A dark-haired Western man in his late thirties in a quiet office meeting room after being laid off, closing his laptop and packing his ID badge into a work bag, coworkers blurred far behind, cobalt, amber, forest green, terracotta, restrained emotion, no text'),
(140,1140,'oil painting on canvas, visible brush strokes, bright airy Impressionist light, cinematic 16:9. A dark-haired Western man in his late thirties in a sunlit cafe with a notebook and phone face down, looking outside toward a tree-lined street and family life, sky blue, spring green, butter yellow, lavender, terracotta and gold, hopeful realistic everyday choice, no text'),
]

pipe=AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
paths=[]
for sid,seed,prompt in items:
    g=torch.Generator(device='cpu').manual_seed(seed)
    im=pipe(prompt=prompt,width=768,height=432,num_inference_steps=2,guidance_scale=0.0,generator=g).images[0].convert('RGB')
    im=im.resize((1920,1080),Image.Resampling.LANCZOS)
    im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Contrast(im).enhance(1.03); im=im.filter(ImageFilter.UnsharpMask(1.2,110,2))
    p=OUT/f'shot_{sid:03d}_sd_turbo.jpg'; im.save(p,quality=93); paths.append(p)
thumb=(640,360); sheet=Image.new('RGB',(thumb[0]*3,thumb[1]*2),(245,242,235)); d=ImageDraw.Draw(sheet)
for i,p in enumerate(paths):
    im=Image.open(p).resize(thumb,Image.Resampling.LANCZOS); x=(i%3)*thumb[0]; y=(i//3)*thumb[1]; sheet.paste(im,(x,y)); d.rectangle((x+10,y+10,x+60,y+42),fill='black'); d.text((x+23,y+18),str(items[i][0]),fill='white')
sheet.save(OUT/'sd_turbo_contact.jpg',quality=92)
print('done',paths)
