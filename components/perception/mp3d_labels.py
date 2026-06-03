"""MP3D-specific labels for GroundingDINO and semantic validation."""

from components.perception.hm3d_labels import (
    HM3D_CANONICAL_LABELS,
    HM3D_COMPATIBLE_LABEL_GROUPS,
    HM3D_DINO_PROMPT_LABELS,
    HM3D_LABEL_ALIASES,
    HM3D_REWARD_DINO_PROMPT_LABELS,
    HM3D_REWARD_EXCLUDED_LABELS,
    _dedupe_preserve_order,
)


MP3D_EXTRA_LABEL_ALIASES = {
    "lighting": "Lamp",
    "light": "Lamp",
    "lights": "Lamp",
    "table lamp": "Lamp",
    "appliance": "Appliance",
    "appliances": "Appliance",
    "tv_monitor": "TV",
    "tv monitor": "TV",
    "chest_of_drawers": "Cabinet",
    "chest of drawers": "Cabinet",
    "shelving": "Shelf",
    "shelves": "Shelf",
    "seating": "Chair",
    "seat": "Chair",
    "bath cabinet": "Cabinet",
    "display case": "Cabinet",
    "windowframe": "Window",
    "doorframe": "Door",
    "shower wall": "Shower",
    "shower floor": "Shower",
    "book": "Book",
    "books": "Book",
    "bottle": "Bottle",
    "pot": "Pot",
    "shoes": "Clothes",
    "fruit bowl": "Bowl",
    "fruit#bowl": "Bowl",
    "clock": "Clock",
    "radio": "Appliance",
    "toiletry": "Toiletry",
    "decoration": "Picture",
    "decor": "Picture",
    "lower cabinet": "Cabinet",
    "upper cabinet": "Cabinet",
    "fan": "Appliance",
    "ceiling fan": "Appliance",
    "bunk": "Bed",
    "bunk bed": "Bed",
    "machine": "Washer",
}

MP3D_EXTRA_PROMPT_LABELS = [
    "lighting",
    "light",
    "table lamp",
    "appliance",
    "appliances",
    "tv monitor",
    "chest of drawers",
    "shelving",
    "shelves",
    "seating",
    "seat",
    "bath cabinet",
    "display case",
    "windowframe",
    "doorframe",
    "shower wall",
    "shower floor",
    "book",
    "books",
    "bottle",
    "pot",
    "shoes",
    "fruit bowl",
    "clock",
    "radio",
    "toiletry",
    "decoration",
    "decor",
    "lower cabinet",
    "upper cabinet",
]

MP3D_IGNORE_LABELS = {
    "object",
    "objects",
    "unknown",
    "delete",
    "remove",
    "partial",
    "patio",
    "void",
    "misc",
    "pipes",
    "vent",
    "kitchen",
    "otherroom",
    "stack",
    "rod",
    "wash",
}

MP3D_EXTRA_CANONICAL_LABELS = {
    "Book",
    "Bottle",
    "Bowl",
    "Clock",
    "Pot",
    "Toiletry",
}

MP3D_CANONICAL_LABELS = set(HM3D_CANONICAL_LABELS) | MP3D_EXTRA_CANONICAL_LABELS
MP3D_LABEL_ALIASES = dict(HM3D_LABEL_ALIASES)
MP3D_LABEL_ALIASES.update(MP3D_EXTRA_LABEL_ALIASES)

for _label in MP3D_CANONICAL_LABELS:
    MP3D_LABEL_ALIASES.setdefault(_label, _label)
    MP3D_LABEL_ALIASES.setdefault(_label.lower(), _label)

MP3D_COMPATIBLE_LABEL_GROUPS = [set(group) for group in HM3D_COMPATIBLE_LABEL_GROUPS]
MP3D_COMPATIBLE_LABEL_GROUPS.extend(
    [
        {"Chair", "Sofa"},
        {"Lamp", "Appliance"},
        {"Cabinet", "Shelf", "Refrigerator", "Washer", "Appliance"},
    ]
)

MP3D_REWARD_EXCLUDED_LABELS = set(HM3D_REWARD_EXCLUDED_LABELS) | {
    "Pillow",
    "Picture",
}

MP3D_DINO_PROMPT_LABELS = _dedupe_preserve_order(
    list(HM3D_DINO_PROMPT_LABELS) + MP3D_EXTRA_PROMPT_LABELS
)
MP3D_REWARD_DINO_PROMPT_LABELS = _dedupe_preserve_order(
    list(HM3D_REWARD_DINO_PROMPT_LABELS)
    + [
        label
        for label in MP3D_EXTRA_PROMPT_LABELS
        if MP3D_LABEL_ALIASES.get(label) not in MP3D_REWARD_EXCLUDED_LABELS
    ]
)

MP3D_DINO_PROMPT = " . ".join(MP3D_DINO_PROMPT_LABELS) + " ."
MP3D_REWARD_DINO_PROMPT = " . ".join(MP3D_REWARD_DINO_PROMPT_LABELS) + " ."
