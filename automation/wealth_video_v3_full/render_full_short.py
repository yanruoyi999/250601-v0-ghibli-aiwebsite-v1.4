from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_short"
OUT.mkdir(exist_ok=True)
chunk = int(sys.argv[1])
items = json.loads((ROOT / "shot_texts_142.json").read_text(encoding="utf-8"))
items = [x for x in items if (int(x["id"]) - 1) % 10 == chunk]

NEG = (
    "photo, photorealistic, 3d render, text, caption, readable numbers, logo, watermark, "
    "collage, split screen, cyberpunk, fantasy portal, giant machinery, duplicate people, "
    "deformed hands, gray monochrome, red blue duotone"
)

PALETTES = {
    1: "cobalt, ultramarine, amber, olive green, walnut, terracotta",
    2: "ultramarine, vermilion, honey gold, emerald, cream, sky blue",
    3: "ultramarine, emerald, terracotta, luminous gold, cream, soft rose",
    4: "cobalt, chrome yellow, burnt orange, viridian, turquoise, warm flesh",
    5: "cobalt, honey yellow, olive, emerald, terracotta, muted rose",
    6: "deep cobalt, amber, forest green, burnt orange, warm cream",
    7: "indigo, cobalt, amber, teal green, warm wood, muted rose",
    8: "cobalt, gold, viridian, terracotta, deep violet, warm flesh",
    9: "ultramarine, honey gold, emerald, olive, terracotta, soft cyan",
    10: "sky blue, spring green, butter yellow, terracotta, lavender, ultramarine",
    11: "sky blue, spring green, butter yellow, lavender, rose, warm cream",
    12: "sky blue, fresh green, pale gold, lavender, rose, terracotta",
    13: "ultramarine, emerald, gold, terracotta, sky blue, warm cream",
    14: "sky blue, fresh green, warm gold, terracotta, lavender, rose",
}

LOCATIONS = [
    "ordinary apartment kitchen", "quiet bedroom", "home office", "open-plan office",
    "neighborhood grocery", "pharmacy counter", "hospital waiting area", "car repair garage",
    "commuter train platform", "small cafe", "family living room", "restaurant with friends",
    "school pickup street", "parking garage", "sunlit city sidewalk", "modest apartment entryway",
]
COMPOSITIONS = [
    "wide eye-level scene", "medium three-quarter view", "over-shoulder view",
    "side-profile medium shot", "intimate narrative detail", "rear three-quarter view",
    "wide interior with person small in frame", "natural two-person interaction",
]
PEOPLE = [
    "restrained dark-haired man in his late thirties", "ordinary professional in business-casual clothes",
    "tired but composed office worker", "adult couple in casual clothes",
    "adult child with an older parent", "two coworkers in ordinary work clothes",
]
PROPS = [
    "keys, mug and groceries", "calendar, envelopes and a houseplant", "lunch container and transit card",
    "family photo and folded papers", "shopping bag and reusable bottle", "coat, notebook and car keys",
    "bread, vegetables and milk", "medicine folder and wallet", "work bag and coffee cup",
    "school backpack and jacket", "repair receipt and worn car part", "plain notebook and phone face down",
]


def scene_for(text: str, seg: int, i: int) -> str:
    t = text.lower()
    person = PEOPLE[(i + seg) % len(PEOPLE)]
    loc = LOCATIONS[(i * 5 + seg) % len(LOCATIONS)]
    props = PROPS[(i * 7 + seg) % len(PROPS)]

    rules = [
        (("raise" in t or "bank balance" in t), f"{person} reading a salary or balance update at a real desk, then returning attention to {props}"),
        (("banking app" in t or "need to check" in t or "number calms" in t), f"{person} checking a phone late at night in an ordinary bedroom, amber lamp, window, plant and {props}"),
        (("bill" in t or "rent" in t or "obligation" in t or "subscription" in t), f"{person} sorting household bills at a kitchen table with {props}; all writing blurred"),
        (("food" in t), f"{person} buying ordinary groceries at a neighborhood checkout with bread, produce, milk and a reusable bag"),
        (("medical" in t or "health problem" in t), f"{person} at a calm hospital or pharmacy counter holding a medical folder and wallet"),
        (("unexpected expense" in t or "financial shock" in t or "shock absorption" in t), f"{person} calmly handling an unexpected car repair at a garage counter with a worn part and wallet"),
        (("income is a flow" in t), f"payday in an ordinary kitchen: {person}, groceries, rent folder and a blurred deposit notification"),
        (("flow changes" in t or "life will not collapse" in t), f"{person} at home after work slows down, but food, rent folder and daily routines remain stable"),
        (("sixty thousand" in t or "two hundred thousand" in t or "bigger number" in t), "two believable households in one continuous apartment building hallway: modest calm life beside expensive tense life"),
        (("consumer financial protection" in t or "financial well-being" in t or "research" in t), f"{person} reviewing a plain research report at a library-style desk with notebook and coffee; text unreadable"),
        (("control over" in t or "monthly finances" in t), f"{person} doing a calm monthly budget at a kitchen table with calendar, envelopes, groceries and {props}"),
        (("goals" in t), f"{person} pinning a family goal photo beside a calendar and savings jar in a home office"),
        (("freedom" in t or "choices" in t or "say no" in t), f"{person} leaving an office into a bright tree-lined street with relaxed shoulders and open space"),
        (("universal salary" in t), "diverse commuters with different lifestyles waiting at one city crosswalk, no magic number or infographic"),
        (("expensive things" in t or "look successful" in t), f"well-dressed professional in a tasteful home quietly checking bills while a financed car sits outside"),
        (("engine" in t and "car" in t), "powerful luxury car boxed into a tight real parking space while a modest compact car has an easy open exit"),
        (("reference point" in t or "comparing" in t or "coworkers" in t or "people around" in t or "feel behind" in t), f"{person} among coworkers or friends noticing subtle status cues during ordinary conversation"),
        (("five years ago" in t), f"{person} holding an old modest-apartment photo while unpacking in a newer ordinary home"),
        (("finish line" in t), f"{person} walking through a real neighborhood from paid credit-card paperwork toward a home viewing"),
        (("credit card" in t), f"close tabletop moment of {person} cutting up a paid credit card beside coffee and keys; details unreadable"),
        (("better home" in t or "apartment" in t or "neighborhood" in t), f"{person} touring a slightly nicer but believable apartment in daylight with a rental folder"),
        (("car payment" in t), f"{person} beside a nice ordinary financed car in a dealership parking lot, keys and payment folder in hand"),
        (("convenience" in t), f"{person} accepting a food delivery at an apartment door with normal delivery bags and household items"),
        (("social life" in t), f"friends at a warm restaurant table while {person} discreetly checks the bill"),
        (("school" in t), f"parents at school pickup with backpacks, family cars and a pleasant but costly neighborhood"),
        (("income can fall" in t), f"{person} at a home desk after a reduced-hours message, smaller paycheck envelope beside unchanged household folders"),
        (("obligations usually do not" in t), "rent folder, car keys, insurance papers, school bag and utility envelopes still lined up on a kitchen table"),
        (("step away from the job" in t or "less free" in t or "one paycheck" in t), f"{person} hesitating at an office elevator late at night with household folders visible in a work bag"),
        (("suspension system" in t), "real car repair garage with worn shock absorbers on the counter beside ordinary household expense papers"),
        (("future" in t and "promise" in t), f"{person} looking from an apartment window at changing weather over the city, calendar and family photo nearby"),
        (("car repair" in t), f"{person} at a repair garage while a mechanic shows a worn tire or brake part, practical solvable expense"),
        (("job loss" in t), f"quiet office meeting room after a layoff conversation; {person} closes a laptop and packs an ID badge"),
        (("family emergency" in t), f"urgent family phone call in a kitchen while coats and car keys are grabbed, realistic restrained urgency"),
        (("recession" in t), "quiet downtown commercial street with a few dark storefronts and normal commuters, slowdown implied without charts"),
        (("parent who needs support" in t or "helping a parent" in t), "adult child and older parent at a kitchen table with medicine, groceries and household papers"),
        (("disaster" in t or "threat" in t or "undefined fear" in t), f"{person} awake in a normal bedroom with no visible crisis, tension shown only through posture and light"),
        (("anything that might ever happen" in t), f"{person} looking at practical home emergency supplies, coats, medical folder and tool kit, recognizing limits"),
        (("cycle" in t or "higher number" in t or "another number" in t), f"{person} in an office after a promotion, newer laptop and nicer coat but the same worried posture"),
        (("lifestyle and comparison group" in t), f"{person} at an upscale coworker dinner surrounded by subtle status cues, participating but uneasy"),
        (("relief fades" in t), f"{person} briefly relaxed at home until another ordinary envelope arrives through the mail slot"),
        (("ambition" in t), f"{person} alone in a bright office late at night after coworkers left, award plaque nearby, tense but controlled"),
        (("buy status" in t), f"{person} pausing before a conspicuous purchase in a car showroom"),
        (("create options" in t), f"{person} at a kitchen table with a job application folder and weekend bag, two realistic choices open"),
        (("protect your family" in t), "parents cooking dinner while a child does homework, emergency folder quietly stored on a shelf"),
        (("leave bad work" in t or "terrible job" in t), f"{person} walking out of a harsh fluorescent office into warm daylight, work badge in pocket"),
        (("controlled by fear" in t), f"{person} making an ordinary grocery choice calmly with a phone left in a pocket"),
        (("assigned a job" in t), f"{person} dividing household money among groceries, emergency savings, transit and time-off planning; labels unreadable"),
        (("scoreboard" in t), f"{person} closing a finance app and turning toward family or a window instead of watching a blurred dashboard"),
        (("options your money protects" in t), f"bright home entryway with {person}, work bag, family photo, bicycle and weekend bag showing real options"),
        (("how many months" in t or "essential life continue" in t), f"{person} counting months on an unreadable calendar beside pantry staples, rent folder and emergency envelope"),
        (("choices" in t and "obligations" in t), f"{person} sorting optional luxury items from household essentials in a closet and garage"),
        (("raises increasing your freedom" in t), f"after a raise, {person} puts money into an emergency envelope before shopping and leaves work on time"),
        (("specific event" in t and "safe from" in t), f"{person} writing one private concern in a notebook beside family photo and medical folder; words invisible"),
        (("six months" in t), f"{person} calmly checking pantry essentials, calendar and emergency envelope in an ordinary home"),
        (("high-interest debt" in t), f"{person} paying off and cutting an expensive credit card, filing the final statement away; details blurred"),
        (("responsibilities changed" in t or "someone else's life" in t), f"{person} at a family table choosing a real responsibility over a flashy blurred social-media image"),
        (("protecting part of the increase" in t), f"payday at a kitchen table: {person} saves part of the increase before considering a new purchase"),
        (("spending is bad" in t), f"{person} enjoying a normal family dinner out and paying comfortably without guilt"),
        (("room to recover" in t), f"{person} resting on a park bench in gentle morning light after leaving work early, phone in pocket"),
        (("room to wait" in t), f"{person} waiting calmly at a train platform with coffee and a small bag, no urgent phone checking"),
        (("room to say no" in t), f"{person} politely declining extra work at an office doorway and walking toward daylight"),
        (("uncommitted money" in t or t.strip() == "room."), f"bright ordinary living room with open floor space around {person}, bills put away and uncluttered table"),
        (("reduce real hardship" in t or "opportunities" in t), "family household using money for repaired car, full groceries, medical care and a training course through real objects"),
        (("control, shock absorption" in t or "progress, and choice" in t), "one coherent home scene: calendar, emergency envelope, repaired appliance and family outing bag show control and choice"),
        (("money is protecting" in t or "another obligation" in t), f"{person} keeps an older reliable car and modest home, spending time with family in a colorful garden"),
        (("life stronger" in t or "more expensive" in t), f"{person} in a bright ordinary home with repaired essentials, family dinner and open schedule, no luxury display"),
        (("what would financial safety" in t), f"{person} in a sunlit cafe with notebook, looking outside toward a park, family and train station"),
        (("write the answer" in t), "close view of a hand writing one private sentence in a notebook beside coffee and a phone face down; words unreadable"),
        (("enough has a purpose" in t), f"{person} closes a notebook and walks from a cafe into a bright colorful morning street toward ordinary life"),
    ]
    for condition, scene in rules:
        if condition:
            return scene
    return f"{person} in a {loc}, doing a concrete everyday action about money and safety with {props}, natural restrained emotion"

print("loading model", flush=True)
pipe = AutoPipelineForText2Image.from_pretrained(
    "Lykon/dreamshaper-8-lcm", torch_dtype=torch.float32,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cpu")
torch.set_num_threads(2)

for x in items:
    i = int(x["id"])
    seg = int(x["segment"])
    scene = scene_for(x["text"], seg, i)
    comp = COMPOSITIONS[(i * 7 + seg) % len(COMPOSITIONS)]
    prompt = (
        f"Oil painting on canvas, visible thick impasto bristle strokes, painterly not photo, "
        f"rich luminous saturated pigments: {PALETTES[seg]}. {comp}: {scene}. "
        "Grounded everyday life, believable anatomy, colored shadows, cinematic 16:9, no text."
    )
    encoded = pipe.tokenizer(prompt, truncation=True, max_length=77, return_tensors=None)["input_ids"]
    prompt = pipe.tokenizer.decode(encoded, skip_special_tokens=True)
    generator = torch.Generator(device="cpu").manual_seed(2026090900 + i * 7919)
    print("render", i, "tokens", len(encoded), flush=True)
    im = pipe(
        prompt=prompt, negative_prompt=NEG, width=640, height=360,
        num_inference_steps=4, guidance_scale=2.2, generator=generator,
    ).images[0].convert("RGB")
    im = Image.blend(im, ImageOps.posterize(im, 7), 0.08)
    im = ImageEnhance.Color(im).enhance(1.10)
    im = ImageEnhance.Brightness(im).enhance(1.04)
    im = ImageEnhance.Contrast(im).enhance(1.02)
    im = im.resize((1920, 1080), Image.Resampling.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=75, threshold=4))
    im.save(OUT / f"shot_{i:03d}.jpg", quality=92, subsampling=0, optimize=True)

print("done", len(items), flush=True)
