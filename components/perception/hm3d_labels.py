"""HM3D/Matterport-oriented labels for GroundingDINO and semantic validation."""

from typing import Dict, List


OBJECT_CATEGORY_FINE_NAMES: Dict[str, List[str]] = {
    "Door": [
        "door", "door frame", "door knob", "doorpost", "door window",
        "door hinge", "sliding door", "garage door", "attic door",
        "garage door railing", "sliding glass door", "doorstep", "ceiling door",
        "door/window", "door stopper", "archway", "doorway", "door handle",
        "elevator", "elevator door",
    ],
    "Lamp": [
        "lamp", "ceiling lamp", "wall lamp", "light fixture", "table lamp",
        "bedside lamp", "chandelier", "ceiling light", "floor lamp", "light",
        "desk lamp", "lampshade", "sconce", "fluorescent light",
        "ceiling fan lamp", "lamp stand", "lighting fixture", "garage light",
    ],
    "Cabinet": [
        "cabinet", "kitchen cabinet", "drawer", "wardrobe", "nightstand",
        "bathroom cabinet", "kitchen lower cabinet", "cabinet door", "dresser",
        "chest of drawers", "sink cabinet", "chest", "display cabinet",
        "wash cabinet", "closet door", "cabinet /otherroom", "wall cabinet",
        "tv stand", "closet", "bedside cabinet", "cabinet drawer",
        "storage cabinet", "bath cabinet", "kitchen cabinet lower",
        "door cabinet", "refrigerator cabinet", "shoe case",
        "kitchen cabinet door", "closet rod", "locker",
        "kitchen cabinet drawer", "sideboard", "file cabinet", "bar cabinet",
    ],
    "Picture": [
        "picture", "painting", "wall hanging decoration", "photo", "frame",
        "art frame", "picture frame", "decorative plate", "sculpture", "statue",
        "poster", "artwork", "figure", "wall sign", "diploma", "flag",
        "figurine", "painting frame", "drawing", "canvas",
    ],
    "Window": [
        "window", "window frame", "window glass", "window shutter", "shutter",
        "window shutters", "exhibition window frame", "exhibition window",
        "ceiling window", "window /outside", "skylight",
    ],
    "Pillow": [
        "pillow", "cushion", "blanket", "throw blanket", "bed comforter",
        "blankets", "bed sheet", "stack of blankets", "duvet", "bedding",
        "round cushion", "stack of pillows",
    ],
    "Chair": [
        "chair", "armchair", "stool", "dining chair", "bench", "pouffe",
        "sofa chair", "bar chair", "desk chair", "seat", "office chair",
        "ottoman", "folding chair", "patio chair", "footrest", "rocking chair",
        "lounge chair", "dinner chair", "piano stool", "lounger", "footstool",
        "kitchen chair", "highchair", "computer chair",
    ],
    "Clothing": [
        "bag", "shoe", "clothes", "hanger", "hanging clothes", "hat",
        "clothes hanger", "laundry basket", "jacket", "backpack", "handbag",
        "luggage", "stack of clothes", "bathrobe", "clothes hanger rod", "cap",
        "bags", "robe", "clothes hamper", "coat hanger", "shirt", "coat",
        "coat rack", "briefcase", "cloth hanger", "slippers", "purse", "scarf",
    ],
    "Table": [
        "table", "desk", "coffee table", "side table", "bed table", "stand",
        "dining table", "bedside table", "computer desk", "flower stand",
        "bed stand", "lamp table", "end table", "kitchen table",
        "small table/stand", "display table", "dinner table", "dressing table",
        "office table",
    ],
    "Shelf": [
        "shelf", "rack", "shelving", "bookshelf", "bathroom shelf",
        "shelf with clutter", "kitchen shelf", "book rack", "book display",
        "clothes rack", "wall hanger", "spice rack", "closet shelf",
        "storage shelving", "wine rack", "high shelf", "shoe rack",
    ],
    "Curtain": [
        "curtain", "blinds", "window curtain", "curtain rod", "shower curtain",
        "window shade", "curtain rail", "shade", "curtain bar",
        "shower curtain rod", "shades", "shower curtain bar",
    ],
    "Plant": [
        "plant", "vase", "decorative plant", "flowerpot", "flower vase",
        "flower", "bouquet", "wreath", "dried flowers", "tree",
        "christmas tree", "bonsai tree",
    ],
    "Sink": [
        "sink", "tap", "faucet", "washbasin", "bath sink", "bath faucet",
        "hot water/cold water knob", "basin", "sink pipe", "bath tap",
        "vessel sink", "kitchen sink", "basin faucet",
    ],
    "Towel": [
        "towel", "towel bar", "paper towel", "bathroom towel", "bath towel",
        "stack of towels", "towel ring", "hand towel", "bath towels",
        "set of towels",
    ],
    "Stairs": [
        "stairs", "stairs railing", "handrail", "stair step", "railing",
        "step", "banister", "ladder", "staircase handrail", "balustrade",
        "rail", "stair", "staircase trim", "stair wall", "staircase",
    ],
    "Mirror": [
        "mirror", "mirror frame", "closet mirror wall", "mirror door",
        "mirror /otherroom", "shower mirror",
    ],
    "Rug": [
        "rug", "carpet", "mat", "doormat", "floor mat", "bath mat",
        "carpet roll", "shower mat", "bathroom rug",
    ],
    "Toilet": [
        "toilet", "toilet paper", "toilet brush", "toilet paper dispenser",
        "wall toilet paper", "bidet", "urinal", "toilet seat", "potty",
    ],
    "Shower": [
        "shower wall", "shower cabin", "showerhead", "shower dial",
        "shower knob", "shower tap", "shower hose/head", "shower door frame",
        "shower", "shower hose", "shower bar", "shower soap shelf",
        "shower ceiling", "shower rail", "shower handle", "shower shelf",
        "shower bench", "shower caddy", "shower door", "shower rod",
        "shower glass", "shower stall",
    ],
    "Bed": [
        "bed", "crib", "bedframe", "headboard", "cradle", "bed base",
        "bunk bed",
    ],
    "Screen": [
        "tv", "monitor", "led tv", "wall tv", "screen", "tv remote",
        "projector", "projector screen",
    ],
    "Appliance": [
        "microwave", "oven", "ventilation hood", "stove", "oven and stove",
        "toaster", "range hood", "cooker", "stovetop", "kitchen extractor",
        "extractor hood",
    ],
    "Sofa": [
        "couch", "sofa", "sofa seat", "l-shaped sofa", "beanbag",
        "circular sofa", "sofa set", "beanbag chair", "chaise longue",
    ],
    "Counter": [
        "kitchen counter", "countertop", "bar", "kitchen island",
        "bathroom counter", "kitchen top", "worktop", "washbasin counter",
        "counter", "counter desk",
    ],
    "Bathtub": [
        "bathtub", "bath wall", "bath", "shower tub", "bathtub platform",
        "bath tub", "jacuzzi", "spa bathtub",
    ],
    "Refrigerator": [
        "refrigerator", "fridge", "mini fridge", "freezer", "icebox",
    ],
    "Washer": [
        "washing machine", "laundry machine", "clothes dryer", "washer-dryer",
        "washing machine and dryer",
    ],
    "Fireplace": [
        "fireplace", "fireplace wall", "hearth", "mantle", "chimney",
        "fireplace shelf", "mantel", "fire pit", "firebox",
    ],
}


_CATEGORY_TO_CANONICAL = {
    "Clothing": "Clothes",
    "Screen": "TV",
}

_STRUCTURAL_LABEL_ALIASES = {
    "wall": "Wall",
    "floor": "Floor",
    "ceiling": "Ceiling",
    "column": "Column",
    "beam": "Beam",
}

HM3D_REWARD_EXCLUDED_LABELS = {
    "Wall",
    "Floor",
    "Ceiling",
    "Window",
    "Door",
    "Stairs",
    "Column",
    "Beam",
}


def _canonical_category(category: str) -> str:
    return _CATEGORY_TO_CANONICAL.get(category, category)


def _dedupe_preserve_order(labels: List[str]) -> List[str]:
    seen = set()
    result = []
    for label in labels:
        label = str(label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


HM3D_DINO_PROMPT_LABELS = _dedupe_preserve_order(
    [
        fine_name
        for fine_names in OBJECT_CATEGORY_FINE_NAMES.values()
        for fine_name in fine_names
    ]
)

HM3D_DINO_PROMPT = " . ".join(HM3D_DINO_PROMPT_LABELS) + " ."

HM3D_REWARD_DINO_PROMPT_LABELS = _dedupe_preserve_order(
    [
        fine_name
        for category, fine_names in OBJECT_CATEGORY_FINE_NAMES.items()
        if _canonical_category(category) not in HM3D_REWARD_EXCLUDED_LABELS
        for fine_name in fine_names
    ]
)

HM3D_REWARD_DINO_PROMPT = " . ".join(HM3D_REWARD_DINO_PROMPT_LABELS) + " ."

HM3D_CANONICAL_LABELS = {
    _canonical_category(category)
    for category in OBJECT_CATEGORY_FINE_NAMES
} | {
    "Wall",
    "Floor",
    "Ceiling",
    "Column",
    "Beam",
}

HM3D_LABEL_ALIASES = {
    fine_name: _canonical_category(category)
    for category, fine_names in OBJECT_CATEGORY_FINE_NAMES.items()
    for fine_name in fine_names
}
HM3D_LABEL_ALIASES.update(_STRUCTURAL_LABEL_ALIASES)

for _category in OBJECT_CATEGORY_FINE_NAMES:
    HM3D_LABEL_ALIASES.setdefault(_category.lower(), _canonical_category(_category))
for _label in HM3D_CANONICAL_LABELS:
    HM3D_LABEL_ALIASES.setdefault(_label, _label)
    HM3D_LABEL_ALIASES.setdefault(_label.lower(), _label)

HM3D_COMPATIBLE_LABEL_GROUPS = [
    {"Chair", "Sofa"},
    {"Table", "Counter"},
    {"Cabinet", "Shelf", "Refrigerator", "Washer"},
    {"Sink", "Bathtub", "Shower", "Toilet"},
    {"Lamp", "Appliance"},
    {"TV", "Picture"},
    {"Pillow", "Bed", "Sofa", "Chair"},
    {"Curtain", "Window"},
    {"Rug", "Floor"},
]
