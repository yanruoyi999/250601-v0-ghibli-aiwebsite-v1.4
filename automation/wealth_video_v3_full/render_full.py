from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out'
OUT.mkdir(exist_ok=True)
chunk=int(sys.argv[1])
items=json.loads((ROOT/'shot_texts_142.json').read_text(encoding='utf-8'))
items=[x for x in items if (int(x['id'])-1)%5==chunk]

NEG='photograph, photorealistic, DSLR, glossy skin, 3D render, text, subtitles, captions, readable numbers, logo, watermark, collage, split screen, storyboard, multi-panel, cyberpunk neon, fantasy portal, giant glass container, floating UI, surreal machinery, black empty background, deformed hands, extra fingers, duplicated people, muddy gray, monochrome red and blue'

settings={
1:['modest apartment kitchen at dusk','quiet bedroom late at night','small living room with a side lamp','hallway outside a bedroom','kitchen counter beside a dark window'],
2:['grocery checkout line','pharmacy billing counter','ordinary rental apartment','small neighborhood cafe','home kitchen table with household receipts','medical waiting room'],
3:['home office with folders and calendar','kitchen table during monthly budgeting','living room with emergency supplies','ordinary workplace break room','parking lot beside a practical car'],
4:['open-plan office','after-work restaurant with coworkers','suburban street of different homes','school pickup area','airport departure hall','weekend brunch table'],
5:['newer apartment living room','parking garage beside a financed car','family dining table with phones and subscriptions','coffee shop convenience purchase','restaurant dinner with friends','school neighborhood sidewalk'],
6:['office after a difficult meeting','kitchen table covered with fixed bills','commuter train platform','parking lot after work','home desk with insurance and rent folders'],
7:['auto repair garage','office meeting room after layoffs','hospital corridor','family kitchen during an urgent phone call','quiet commercial street during a downturn','parents living room'],
8:['late office with promotion materials','gym locker room after work','upscale networking event','bedroom desk after midnight','elevator mirror after a long workday'],
9:['family kitchen table','small home office','park bench with family nearby','front door of an unpleasant workplace','simple desk with household envelopes'],
10:['kitchen table with monthly essentials','closet showing reversible lifestyle choices','office desk after a raise','quiet hospital parking lot','family living room'],
11:['pantry and household essentials','credit card statement beside scissors','office doorway with a folded resignation letter','parent and adult child at kitchen table','calendar beside emergency savings envelope'],
12:['sunlit room with open floor space','quiet park bench in morning light','kitchen after bills are put away','train platform with time to wait','front doorway with freedom to leave'],
13:['family breakfast table','community park on a bright day','work desk with clear space around it','repair shop where an expense is calmly handled','sunlit home office'],
14:['open window above a writing desk','quiet cafe with notebook','park path in morning light','home doorway opening to a bright street','family dinner table after a calm conversation']}

palettes={
1:'cobalt blue, ultramarine, amber tungsten, olive green, walnut brown, terracotta accents',
2:'ultramarine, warm vermilion, honey gold, deep natural green, cream and warm flesh tones',
3:'Venetian jewel colors: ultramarine, emerald green, terracotta, warm gold, cream, small rose accents',
4:'cobalt and chrome yellow, burnt orange, viridian green, turquoise accents, warm skin tones',
5:'cobalt, honey yellow, olive and emerald green, terracotta orange, muted rose, natural wood',
6:'deep cobalt, amber, forest green, burnt orange, warm cream, restrained crimson accents',
7:'indigo and cobalt, amber lamps, teal-green walls, warm wood, muted rose and cream',
8:'cobalt against gold, viridian green, terracotta, deep violet accents, warm flesh tones',
9:'ultramarine, honey gold, emerald and olive, terracotta, soft cyan, warm cream',
10:'sky blue, spring green, butter yellow, terracotta, ultramarine, soft lavender',
11:'sky blue, spring green, butter yellow, soft lavender and rose, warm cream and ultramarine',
12:'sky blue, fresh green, pale gold, lavender, rose, terracotta and colored shadows',
13:'ultramarine, emerald, gold, terracotta, sky blue, warm cream',
14:'sky blue, fresh green, warm gold, terracotta, soft lavender and rose, ultramarine details'}

types=['wide eye-level environmental shot','medium three-quarter narrative shot','over-the-shoulder composition','side-profile medium shot','intimate close narrative detail','rear three-quarter shot with environment visible','tabletop still-life with a human hand entering frame','wide domestic interior with the person small in frame','natural two-person interaction','quiet observational street-level scene']
chars=['a restrained Western man in his mid-30s with short dark curly hair and light stubble','an ordinary Western professional in his late 30s wearing a simple shirt and dark trousers','a tired but composed man in his 30s in everyday office clothes','an adult couple in their 30s dressed casually','an adult child and an older parent in a normal family setting','two coworkers in ordinary business-casual clothes']

def scene(text,seg,i):
 t=text.lower(); loc=settings[seg][(i*3+seg)%len(settings[seg])]; c=chars[(i+seg)%len(chars)]
 if 'raise' in t and 'room you stand' not in t: return f'{c} in a real office reading a salary increase message on a laptop, then setting it beside a lunch container and transit card; small relieved expression, no celebration'
 if 'bank balance' in t or 'banking app' in t or 'need to check' in t or 'number calms' in t: return f'{c} in a {loc}, quietly checking a phone banking screen whose details are unreadable; paid envelope, keys, mug and groceries make the scene lived-in'
 if 'bill' in t or 'rent' in t or 'fixed commitment' in t or 'obligation' in t or 'subscriptions' in t: return f'real household finance moment in a {loc}: envelopes, rent folder, insurance papers, keys and a mug while {c} sorts them; all writing blurred'
 if 'food' in t: return f'{c} at a neighborhood grocery checkout with vegetables, bread, milk and a reusable bag, ordinary weekly purchase'
 if 'medical' in t or 'health problem' in t: return f'{c} at a clean hospital or pharmacy billing counter holding a medical folder and wallet, realistic waiting chairs, no readable words'
 if 'unexpected expense' in t or 'financial shock' in t or 'shock absorption' in t: return f'{c} at an auto repair counter looking at a car part and paper estimate, calm enough to handle the expense'
 if 'income is a flow' in t: return f'ordinary payday scene in a {loc}: blurred deposit notification beside groceries and household receipts; money shown through real-life flow, no symbols'
 if 'flow changes' in t or 'life will not collapse' in t: return f'{c} at home during a quiet work interruption, laptop closed but rent paid, food in the kitchen and car keys by the door; life remains stable'
 if 'sixty thousand' in t or 'two hundred thousand' in t or 'bigger number' in t: return 'one continuous realistic interior with a modest relaxed household in the foreground and a more expensive tense household visible through an adjacent doorway; no split screen'
 if 'consumer financial protection' in t or 'financial well-being' in t: return f'{c} at a library-style desk reviewing a plain public research report with charts out of focus, notebook and coffee nearby'
 if 'control over' in t or 'monthly finances' in t: return f'{c} doing a calm monthly budget at a kitchen table, calendar, categorized envelopes, groceries and laptop arranged neatly, no readable labels'
 if 'goals' in t: return f'{c} pinning a simple household goal photo to a corkboard beside a calendar and savings jar, realistic long-term planning'
 if 'freedom' in t or 'choices' in t or 'say no' in t: return f'{c} leaving an office building into a bright tree-lined street with no hurry, holding a small bag; physical space expresses freedom'
 if 'universal salary' in t: return 'diverse commuters with visibly different lifestyles waiting at the same city crosswalk, none singled out by a magic number'
 if 'expensive things' in t or 'look successful' in t: return 'well-dressed professional in a tasteful apartment with a financed car visible outside, quietly checking bills at the kitchen counter; polished but fragile'
 if 'engine' in t and 'room around the car' in t: return 'real city parking situation: powerful luxury car boxed tightly between obstacles while a modest compact car nearby has an open lane and easy exit, subtle practical metaphor'
 if 'reference point' in t or 'comparing' in t or 'coworkers' in t or 'people around' in t or 'feel behind' in t or 'new room' in t: return f'{c} among coworkers or friends in a {loc}, noticing upgraded watches, homes, travel photos and school talk in ordinary interaction'
 if 'five years ago' in t: return f'{c} packing an old moving box with a modest apartment photo while standing in a newer but ordinary home, grounded comparison with the past'
 if 'finish line' in t: return f'{c} walking through a real neighborhood toward a home open house while passing an old cut-up credit card in a recycling bin; goals changing through everyday milestones'
 if 'credit card' in t: return f'close tabletop scene of {c} cutting up an old credit card beside a paid statement, coffee and keys, account details unreadable'
 if 'better home' in t or 'apartment' in t or 'neighborhood' in t: return f'{c} touring a slightly nicer but believable apartment in daylight, holding a rental folder; attractive but not luxurious'
 if 'never having to think about money' in t: return f'{c} awake in a comfortable orderly home at night, phone face down beside the bed; comfort without guaranteed mental peace'
 if 'car payment' in t: return f'{c} in a dealership parking lot beside a nice but ordinary financed car, keys in hand and payment folder under the arm'
 if 'convenience' in t: return f'{c} accepting a food delivery at an apartment door while small delivery bags and subscription boxes sit nearby'
 if 'social life' in t: return f'friends at a warm restaurant table ordering food, {c} checking the bill discreetly, convivial but expensive routine'
 if 'school' in t: return f'parents at school pickup in a pleasant neighborhood, backpacks and cars visible, {c} thinking about maintaining the area'
 if 'income can fall' in t: return f'{c} at a home desk after receiving a reduced-hours message on a blurred laptop, smaller paycheck envelope beside unchanged household folders'
 if 'obligations usually do not' in t: return f'rent folder, car keys, insurance papers, school bag and utility envelopes still lined up on a kitchen table while {c} looks at them'
 if 'step away from the job' in t or 'less free' in t or 'one paycheck' in t: return f'{c} at an office elevator late in the evening, coat on but hesitating because household folders are visible in the work bag'
 if 'suspension system' in t: return 'realistic car repair garage with worn shock absorbers on the counter while the owner reviews household expenses nearby; subtle physical analogy'
 if 'future' in t and 'promises' in t: return f'{c} looking from an apartment window at changing weather over the city, calendar and family photo on the sill; ordinary future uncertainty'
 if 'car repair' in t: return f'{c} at an auto repair garage watching a mechanic show a worn tire or brake part, wallet ready, practical solvable expense'
 if 'job loss' in t: return f'quiet office meeting room after a layoff conversation, {c} closing a laptop and putting an ID badge in a bag, restrained emotion'
 if 'family emergency' in t: return f'{c} receiving an urgent phone call in a kitchen while another family member grabs a coat and car keys, realistic urgency'
 if 'recession' in t: return 'quiet downtown commercial street with several dark storefronts at early evening, commuters passing normally, slowdown implied without charts'
 if 'parent who needs support' in t or 'helping a parent' in t: return 'adult child and older parent at a kitchen table reviewing medication, groceries and household papers together, caring realistic support'
 if 'disaster' in t or 'threat' in t or 'fear is irrational' in t or 'undefined fear' in t: return f'{c} alone in a normal bedroom at night, phone dark, mind alert despite no visible crisis; anxiety shown through posture and light'
 if 'anything that might ever happen' in t: return f'{c} looking at a home closet containing practical emergency supplies, family coats, medical folder and tool kit, recognizing limits of preparation'
 if 'cycle' in t or 'higher number' in t or 'another number' in t: return f'{c} in an office with signs of a recent promotion, newer laptop and nicer coat but the same worried posture, staying late'
 if 'lifestyle and comparison group' in t: return f'{c} at an upscale coworker dinner in a nicer part of town, surrounded by subtle status cues, participating but feeling behind'
 if 'relief fades' in t: return f'{c} at home after a good financial month, brief smile disappearing as another ordinary envelope arrives through the mail slot'
 if 'ambition' in t: return f'{c} alone in a bright office late at night after colleagues left, award plaque and running shoes nearby, working harder with a tense jaw'
 if 'buy status' in t: return f'{c} in a car showroom considering a conspicuous status purchase, then pausing before paying'
 if 'create options' in t: return f'{c} at a kitchen table with two realistic choices available, a job application folder and a travel bag, calm posture'
 if 'protect your family' in t: return 'parents calmly cooking dinner while a child does homework nearby, emergency folder and insurance card tucked on a shelf'
 if 'leave bad work' in t or 'terrible job' in t: return f'{c} walking out of a harsh fluorescent office into warm daylight, work badge in a pocket, relaxed shoulders'
 if 'decisions controlled by fear' in t: return f'{c} at a grocery aisle making ordinary choices calmly instead of compulsively checking a phone'
 if 'assigned a job' in t: return f'{c} dividing household funds across an emergency envelope, groceries, transit pass and time-off calendar, no readable labels'
 if 'scoreboard' in t: return f'{c} closing a finance app and turning toward family or a window, ignoring a blurred performance dashboard'
 if 'options your money protects' in t: return f'{c} in a bright home entryway with work bag, family photo, bicycle and packed weekend bag, several real options open'
 if 'how many months' in t or 'essential life continue' in t: return f'{c} at a kitchen table counting months on an unreadable calendar beside pantry staples, rent folder and emergency envelope'
 if 'choices' in t and 'obligations' in t: return f'closet and garage scene where {c} sorts optional luxury items from essential household items, deciding what could be reversed'
 if 'raises increasing your freedom' in t: return f'{c} after receiving a raise, placing part into an emergency envelope before shopping, then closing the laptop and leaving work on time'
 if 'specific event' in t and 'safe from' in t: return f'{c} writing one concrete concern in a private notebook whose words are invisible, family photo and medical folder nearby'
 if 'six months' in t: return f'calm pantry and household scene with several months of essentials stocked sensibly, {c} checking a calendar and emergency envelope'
 if 'high-interest debt' in t: return f'close tabletop moment: {c} paying off and cutting an expensive credit card, then filing the final statement away, details blurred'
 if 'responsibilities changed' in t or "someone else's life" in t: return f'{c} at a family table comparing a real new responsibility with a flashy social media post blurred on a phone, choosing the real responsibility'
 if 'protecting part of the increase' in t: return f'payday at a kitchen table: {c} moves part of a deposit into an emergency envelope before considering a new purchase'
 if 'spending is bad' in t: return f'{c} enjoying a normal dinner out with family, paying comfortably without guilt'
 if 'room to recover' in t: return f'{c} resting on a park bench in gentle morning light after leaving work early, phone in pocket, unhurried'
 if 'room to wait' in t: return f'{c} waiting calmly at a train platform with coffee and a small bag, no urgent phone checking'
 if 'room to say no' in t: return f'{c} at an office doorway politely declining extra work and walking toward daylight'
 if 'uncommitted money' in t or t.strip()=='room.': return f'bright ordinary living room with open floor space around {c}, bills put away, chair by window and uncluttered table; literal breathing space'
 if 'reduce real hardship' in t or 'opportunities' in t: return 'family household using money for practical improvements: repaired car, full groceries, medical care and a training course represented through real objects in one coherent home scene'
 if 'larger number' in t and 'safety' in t: return f'{c} looking at a high bank balance on a blurred screen, then turning to practical household buffers and family needs around the room'
 if 'control, shock absorption' in t or 'progress, and choice' in t: return f'single coherent home scene showing control, buffer, progress and choice through a calendar, emergency envelope, repaired appliance and family outing bag, with {c} calmly present'
 if 'money is protecting' in t or 'another obligation' in t: return f'{c} choosing to keep an older reliable car and modest home rather than upgrading, spending time with family in a colorful driveway and garden'
 if 'life stronger' in t or 'not merely more expensive' in t: return f'{c} in a bright ordinary home with repaired essentials, family dinner and open schedule, visibly stronger life without luxury display'
 if 'what would financial safety' in t: return f'{c} in a sunlit cafe with a notebook, looking through the window toward ordinary possibilities like a park, family and train station'
 if 'write the answer' in t: return 'close narrative shot of a hand writing one private sentence in a notebook beside coffee and a phone face down, words unreadable'
 if 'enough has a purpose' in t: return f'{c} closing the notebook and walking from a cafe into a bright colorful morning street toward family and ordinary life, calm chosen direction'
 return f'{c} in a {loc}, performing a concrete everyday action that reflects this idea: {text}; believable household or workplace objects, no fantasy'

BASE='Wide 16:9 narrative oil painting on canvas, visibly painted rather than photographed. Thick controlled bristle strokes, layered pigment, subtle impasto highlights, painterly edges, tactile canvas texture. Realistic anatomy and believable everyday architecture, but not photorealistic and not a digital photo. Emotionally restrained adult acting. Luminous midtones, clear highlights, colored shadows instead of dead black. '

print('loading model',flush=True)
pipe=AutoPipelineForText2Image.from_pretrained('Lykon/dreamshaper-8-lcm',torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing(); pipe=pipe.to('cpu'); torch.set_num_threads(2)

for x in items:
 i=int(x['id']); seg=int(x['segment']); text=x['text']
 prompt=BASE+types[(i*7+seg)%len(types)]+'. '+scene(text,seg,i)+'. Color design: '+palettes[seg]+'. Keep saturation rich and painterly like a vivid traditional oil painting, never fluorescent. Use four to six naturally occurring hues from clothing, walls, plants, food, sky, lamps, wood and skin. Renaissance jewel-color structure, Impressionist luminous light where appropriate, and small complementary-color accents. Everyday life first. No text in image.'
 g=torch.Generator(device='cpu').manual_seed(2026090900+i*7919)
 print('render',i,flush=True)
 im=pipe(prompt=prompt,negative_prompt=NEG,width=640,height=360,num_inference_steps=4,guidance_scale=2.2,generator=g).images[0].convert('RGB')
 # Subtle painterly finish without turning it into cartoon art.
 poster=ImageOps.posterize(im,7)
 im=Image.blend(im,poster,0.08)
 im=ImageEnhance.Color(im).enhance(1.10)
 im=ImageEnhance.Brightness(im).enhance(1.04)
 im=ImageEnhance.Contrast(im).enhance(1.02)
 im=im.resize((1920,1080),Image.Resampling.LANCZOS)
 im=im.filter(ImageFilter.UnsharpMask(radius=1.2,percent=75,threshold=4))
 im.save(OUT/f'shot_{i:03d}.jpg',quality=92,subsampling=0,optimize=True)
print('done',len(items),flush=True)
