from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageEnhance, ImageFilter

ROOT=Path(__file__).resolve().parents[1]/'wealth_video_v3_full'
OUT=Path(__file__).resolve().parent/'out_full'; OUT.mkdir(parents=True,exist_ok=True)
chunk=int(sys.argv[1])
items=json.loads((ROOT/'shot_texts_142.json').read_text(encoding='utf-8'))
items=[x for x in items if (int(x['id'])-1)%10==chunk]

PALETTES={
1:'cobalt blue, honey gold, olive green, terracotta, cream, a little turquoise',
2:'ultramarine, emerald green, warm gold, terracotta, cream, sky blue',
3:'ultramarine, vermilion, emerald, luminous gold, warm cream, soft rose',
4:'cobalt blue, chrome yellow, burnt orange, viridian green, turquoise, warm flesh',
5:'cobalt, honey yellow, olive green, terracotta, muted rose, cream',
6:'deep cobalt, amber, forest green, burnt orange, warm cream, muted rose',
7:'indigo, cobalt, amber, teal green, warm wood, muted rose',
8:'cobalt, gold, viridian, terracotta, deep violet, warm flesh',
9:'ultramarine, honey gold, emerald, olive, terracotta, soft cyan',
10:'sky blue, spring green, butter yellow, terracotta, lavender, ultramarine',
11:'sky blue, spring green, butter yellow, lavender, rose, warm cream',
12:'sky blue, fresh green, pale gold, lavender, rose, terracotta',
13:'ultramarine, emerald, gold, terracotta, sky blue, warm cream',
14:'sky blue, fresh green, warm gold, terracotta, lavender, rose',
}
COMPS=['wide eye-level view','medium three-quarter view','over-the-shoulder view','side-profile medium view','wide interior view','rear three-quarter view']

# Specific grounded scenes for the hook and high-risk concepts.
EXACT={
1:'a dark-haired Western man in his late thirties at a real office desk reading a salary raise notification on a laptop, restrained relief, coffee mug and green plant',
2:'the same type of adult man at a kitchen table reviewing a higher bank balance on his phone beside paid household envelopes and groceries',
3:'the adult man putting paid utility envelopes into a drawer after dinner in an ordinary apartment',
4:'the adult man in a calm ordinary home at night, lights on, groceries stocked, no visible crisis, but his posture still tense',
5:'the adult man alone on the edge of his bed late at night checking a banking app on his phone, amber bedside lamp and cobalt night window',
6:'the adult man alone in the quiet bedroom holding the phone even though nothing has happened, listening to the silence',
7:'the adult man alone checking the phone one more time before sleep, tired eyes and restrained anxiety',
8:'the adult man briefly relaxing after seeing the balance, shoulders dropping, phone glow fading',
9:'the adult man at a kitchen table beginning to calculate again with a notebook, plain envelopes, calendar and phone',
10:'close everyday scene of the adult man opening the next household bill at the kitchen table, all writing unreadable',
11:'the adult man reaching for car keys and a medical folder after an unexpected family phone call, realistic urgency',
12:'the adult man looking around a slightly more expensive apartment with groceries, furniture and monthly commitments accumulating',
13:'the adult man at a window believing money should make him safe, comfortable home behind him but worried expression',
14:'the adult man walking through a normal bright city street while his worried expression remains, safety still feels farther away',
16:'the adult man at a modest apartment table with rent envelope and calculator, serious but practical',
17:'the adult man buying bread, vegetables and milk at a neighborhood grocery checkout',
18:'the adult man at a hospital billing or pharmacy counter holding a medical folder and wallet',
19:'the adult man at a car repair garage counter while a mechanic shows an unexpected repair estimate folder',
25:'a modest calm household in one apartment and a wealthier tense household in another apartment across the same hallway, realistic contrast',
26:'the protagonist calmly cooking in a modest kitchen while a wealthier professional in the background hallway looks tense over paperwork',
27:'a well-dressed professional alone in a tasteful home quietly checking bills while an expensive financed car is visible outside',
41:'a powerful luxury car boxed into a tight real parking space while a modest compact car has a clear open exit, grounded visual metaphor',
44:'the protagonist in a normal office conversation noticing better watches, clothing and status cues among coworkers',
45:'parents and children at a pleasant school pickup street with family cars and backpacks, realistic status comparison',
53:'the protagonist cutting up a paid-off credit card at a kitchen table, coffee and keys nearby, no readable numbers',
54:'the protagonist touring a slightly nicer but believable apartment in daylight with a rental folder',
59:'the protagonist beside a nice but ordinary financed car in a dealership parking lot, keys and payment folder in hand',
60:'the protagonist reviewing multiple small subscription charges on a phone at home, screen text unreadable',
61:'the protagonist accepting a food delivery at an apartment door, convenience becoming routine',
62:'friends at a warm restaurant table while the protagonist discreetly checks the bill',
63:'parents at school pickup in a pleasant but costly neighborhood, backpacks and family cars visible',
66:'the protagonist at a home desk after a reduced-hours message, smaller paycheck envelope beside unchanged household folders',
67:'rent folder, car keys, insurance papers, school bag and utility envelopes still lined up on a kitchen table',
69:'the protagonist hesitating at an office elevator late at night, household folders visible in his work bag',
72:'a high-earning professional alone at a kitchen table with nearly empty cash buffer and many fixed household folders, realistic fragility',
74:'the protagonist at a car repair garage while a mechanic shows a worn suspension part, ordinary car behind them',
77:'the protagonist at a repair garage while a mechanic shows a worn tire or brake part, solvable expense',
78:'quiet office meeting room after a layoff conversation, the protagonist closes a laptop and packs an ID badge',
79:'urgent family phone call in a kitchen while the protagonist grabs a coat and car keys',
80:'quiet downtown commercial street with a few dark storefronts and normal commuters, recession implied without charts',
81:'the protagonist in a calm hospital waiting area holding a medical folder, realistic health uncertainty',
82:'the protagonist and an older parent at a kitchen table with medicine, groceries and household papers',
90:'the protagonist alone in a normal bedroom looking at practical emergency supplies, coats, medical folder and tool kit, realizing limits',
97:'the protagonist alone in a bright office late at night after coworkers have left, promotion plaque nearby, tense but controlled',
99:'the protagonist still working late in a successful office, ambition on the outside and anxiety in his face',
102:'the protagonist pausing before a conspicuous purchase in a car showroom, status temptation',
103:'the protagonist at a kitchen table with a job application folder and weekend bag, two realistic options open',
104:'parents cooking dinner while a child does homework, emergency folder quietly stored on a shelf',
105:'the protagonist walking out of a harsh fluorescent office into warm daylight, work badge in his pocket',
107:'the protagonist dividing household money among groceries, emergency savings, transit and time-off planning, labels unreadable',
108:'the protagonist closing a finance app and turning toward family and a window instead of watching a dashboard',
112:'the protagonist counting months on an unreadable calendar beside pantry staples, rent folder and emergency envelope',
113:'the protagonist sorting optional luxury items from household essentials in a closet and garage',
114:'after a raise, the protagonist puts part of the money into an emergency envelope before shopping and leaves work on time',
115:'the protagonist writing one specific private concern in a notebook beside a family photo and medical folder, words invisible',
117:'the protagonist calmly checking pantry essentials, calendar and emergency envelope in an ordinary home',
118:'the protagonist paying off and cutting an expensive credit card, filing the final statement away, details blurred',
119:'the protagonist politely declining extra work at an office doorway and walking toward daylight',
120:'the protagonist and an older parent at a kitchen table with medicine and groceries, helping without panic',
125:'payday at a kitchen table, the protagonist saves part of the increase before considering a new purchase',
126:'the protagonist enjoying a normal family dinner out and paying comfortably without guilt',
127:'bright ordinary living room with open floor space around the protagonist, bills put away and uncluttered table',
128:'the protagonist resting on a park bench in gentle morning light after leaving work early, phone in pocket',
129:'the protagonist waiting calmly at a train platform with coffee and a small bag, no urgent phone checking',
130:'the protagonist politely declining extra work at an office doorway and walking toward a bright tree-lined street',
132:'one family household using money for a repaired car, full groceries, medical care and a training course through real objects',
134:'one coherent home scene with calendar, emergency envelope, repaired appliance and family outing bag showing control and choice',
136:'the protagonist keeps an older reliable car and modest home, spending time with family in a colorful garden',
138:'the protagonist in a bright ordinary home with repaired essentials, family dinner and open schedule, no luxury display',
140:'the protagonist in a sunlit cafe with notebook and phone face down, looking outside toward a park, family life and a train station',
141:'close view of the protagonist hand writing one private sentence in a notebook beside coffee and a phone face down, words unreadable',
142:'the protagonist closes a notebook and walks from the cafe into a bright colorful morning street toward ordinary life',
}

def scene_for(i:int,text:str,seg:int)->str:
    if i in EXACT: return EXACT[i]
    t=text.lower()
    man='a dark-haired Western man in his late thirties with light stubble'
    if 'research' in t or 'consumer financial' in t or 'well-being' in t: return f'{man} reviewing a plain research report at a library desk with notebook and coffee, all text unreadable'
    if 'control over' in t or 'monthly finances' in t: return f'{man} calmly doing a monthly budget at a kitchen table with calendar, envelopes and groceries'
    if 'financial shock' in t or 'shock absorption' in t: return f'{man} handling an unexpected but manageable appliance repair at home with a technician and repair folder'
    if 'goals' in t: return f'{man} pinning a family goal photo beside a calendar and savings envelope in a home office'
    if 'freedom' in t or 'choices' in t or 'say no' in t: return f'{man} leaving an office into a bright tree-lined street with relaxed shoulders and open space'
    if 'universal salary' in t: return 'diverse adult commuters with different lifestyles waiting at one city crosswalk, no infographic or magic number'
    if 'expensive things' in t or 'look successful' in t: return f'{man}, well dressed, alone in a tasteful home quietly checking bills while a financed car sits outside'
    if 'reference point' in t or 'comparing' in t or 'people around' in t or 'feel behind' in t: return f'{man} among coworkers or friends noticing subtle status cues during ordinary conversation'
    if 'five years ago' in t: return f'{man} holding an old modest-apartment photo while unpacking in a newer ordinary home'
    if 'finish line' in t: return f'{man} walking through a real neighborhood from paid credit-card paperwork toward a home viewing, no literal race track'
    if 'apartment' in t or 'better home' in t or 'neighborhood' in t: return f'{man} touring a slightly nicer but believable apartment in daylight with a rental folder'
    if 'convenience' in t: return f'{man} accepting food delivery at an apartment door with normal household items around him'
    if 'social life' in t: return f'{man} at a warm restaurant with friends while discreetly checking the bill'
    if 'income can fall' in t: return f'{man} at a home desk after a reduced-hours message, smaller paycheck envelope beside unchanged household folders'
    if 'obligations' in t or 'fixed commitments' in t: return f'{man} sorting rent, car, insurance and school-related household folders at a kitchen table, all writing blurred'
    if 'less free' in t or 'step away' in t or 'one paycheck' in t: return f'{man} hesitating at an office elevator late at night with household folders visible in a work bag'
    if 'future' in t and 'promise' in t: return f'{man} looking from an apartment window at changing weather over the city, calendar and family photo nearby'
    if 'disaster' in t or 'threat' in t or 'undefined fear' in t: return f'{man} awake alone in a normal bedroom with no visible crisis, tension shown only through posture and light'
    if 'higher number' in t or 'another number' in t or 'cycle' in t: return f'{man} in an office after a promotion, newer laptop and nicer coat but the same worried posture'
    if 'comparison group' in t: return f'{man} at an upscale coworker dinner surrounded by subtle status cues, participating but uneasy'
    if 'relief fades' in t: return f'{man} briefly relaxed at home until another ordinary envelope arrives through the mail slot'
    if 'ambition' in t: return f'{man} alone in a bright office late at night after coworkers left, achievement plaque nearby, tense but controlled'
    if 'controlled by fear' in t: return f'{man} making an ordinary grocery choice calmly with the phone left in his pocket'
    if 'scoreboard' in t: return f'{man} closing a finance app and turning toward family or a window instead of watching a blurred dashboard'
    if 'options your money protects' in t: return f'bright home entryway with {man}, work bag, family photo, bicycle and weekend bag showing real options'
    if 'how many months' in t or 'essential life continue' in t: return f'{man} counting months on an unreadable calendar beside pantry staples and an emergency envelope'
    if 'specific event' in t and 'safe from' in t: return f'{man} writing one private concern in a notebook beside family photo and medical folder, words invisible'
    if 'six months' in t: return f'{man} calmly checking pantry essentials, calendar and emergency envelope in an ordinary home'
    if 'high-interest debt' in t: return f'{man} paying off and cutting a credit card, filing the final statement away, details unreadable'
    if 'responsibilities changed' in t or "someone else's life" in t: return f'{man} at a family table choosing a real responsibility over a flashy social-media image on a phone'
    if 'protecting part' in t: return f'payday at a kitchen table, {man} saves part of the increase before considering a new purchase'
    if 'room to recover' in t: return f'{man} resting on a park bench in gentle morning light after leaving work early, phone in pocket'
    if 'room to wait' in t: return f'{man} waiting calmly at a train platform with coffee and a small bag, no urgent phone checking'
    if 'room to say no' in t: return f'{man} politely declining extra work at an office doorway and walking toward daylight'
    if text.strip().lower()=='room.': return f'bright ordinary living room with open floor space around {man}, bills put away and uncluttered table'
    if 'money can improve' in t: return f'{man} arriving home with groceries and a repaired car outside, ordinary life visibly easier'
    if 'hardship' in t or 'opportunities' in t: return 'a family household using money for groceries, medical care, repaired transport and a training course through real objects'
    if 'control' in t and 'choice' in t: return 'one coherent home scene with calendar, emergency envelope, repaired appliance and family outing bag showing control and choice'
    if 'money is protecting' in t: return f'{man} at home storing an emergency folder on a shelf while family life continues around him'
    if 'another obligation' in t: return f'{man} declining a flashy upgrade and keeping an older reliable car in a normal driveway'
    if 'stronger' in t or 'more expensive' in t: return f'{man} in a bright ordinary home with repaired essentials, family dinner and open schedule, no luxury display'
    # Segment-specific grounded fallbacks.
    fallbacks={
      1:f'{man} alone in an ordinary apartment handling a phone, notebook and household envelopes, restrained money anxiety',
      2:f'{man} in a real kitchen or workplace dealing with ordinary rent, food, healthcare and income decisions',
      3:f'{man} calmly organizing a household budget, emergency envelope, family goal and free-time plan at home',
      4:f'{man} in an office, neighborhood or social setting noticing subtle changes in status and comparison',
      5:f'{man} in an ordinary apartment, car lot or restaurant as flexible income becomes everyday commitments',
      6:f'{man} at work and home feeling fixed household costs while job flexibility shrinks',
      7:f'{man} facing a realistic future uncertainty in a home, hospital, garage or office without fantasy symbols',
      8:f'{man} after a promotion in a nicer office or social setting, outwardly successful but still tense',
      9:f'{man} choosing between a purchase, family support, job freedom and practical options in everyday life',
      10:f'{man} at a kitchen table turning financial safety into concrete questions using ordinary household objects',
      11:f'{man} working through debt, emergency savings, job choice and family support in realistic settings',
      12:f'{man} preserving part of a raise and gaining open space, rest, time and the ability to decline pressure',
      13:f'{man} using money to reduce real hardship and protect choices in an ordinary colorful family home',
      14:f'{man} in a bright cafe, home or morning street defining what enough is for in real life',
    }
    return fallbacks[seg]

def is_multi(scene:str)->bool:
    words=('coworker','friends','family household','parents','child','mechanic','technician','older parent','diverse adult','two household','restaurant with friends','family life continues')
    return any(w in scene.lower() for w in words)

print('loading sd-turbo',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)
for x in items:
    i=int(x['id']); seg=int(x['segment']); scene=scene_for(i,x['text'],seg)
    people='believable adults, no duplicate faces' if is_multi(scene) else 'one adult man only, alone, no second person'
    comp=COMPS[(i*7+seg)%len(COMPS)]
    prompt=(f'{scene}. {people}. {comp}. Cinematic narrative oil painting on canvas, visible impasto brush strokes, realistic anatomy and believable space, luminous saturated pigments and colored shadows. Palette: {PALETTES[seg]}. No readable text, no collage, no fantasy symbols.')
    g=torch.Generator(device='cpu').manual_seed(830000+i*7919)
    im=pipe(prompt=prompt,width=768,height=432,num_inference_steps=2,guidance_scale=0.0,generator=g).images[0].convert('RGB')
    im=im.resize((1920,1080),Image.Resampling.LANCZOS)
    im=ImageEnhance.Color(im).enhance(1.08); im=ImageEnhance.Contrast(im).enhance(1.03); im=ImageEnhance.Brightness(im).enhance(1.02); im=im.filter(ImageFilter.UnsharpMask(1.1,105,2))
    p=OUT/f'shot_{i:03d}.jpg'; im.save(p,quality=93,subsampling=0)
    print('saved',p.name,flush=True)
