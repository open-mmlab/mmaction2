import os

from detectron2.data.datasets.register_coco import register_coco_instances

categories_v1 = [
    {
        'id': 164,
        'name': 'cutting/chopping board'
    },
    {
        'id': 49,
        'name': 'tie'
    },
    {
        'id': 306,
        'name': 'crosswalk sign'
    },
    {
        'id': 145,
        'name': 'gun'
    },
    {
        'id': 14,
        'name': 'street lights'
    },
    {
        'id': 223,
        'name': 'bar soap'
    },
    {
        'id': 74,
        'name': 'wild bird'
    },
    {
        'id': 219,
        'name': 'ice cream'
    },
    {
        'id': 37,
        'name': 'stool'
    },
    {
        'id': 25,
        'name': 'storage box'
    },
    {
        'id': 153,
        'name': 'giraffe'
    },
    {
        'id': 52,
        'name': 'pen/pencil'
    },
    {
        'id': 61,
        'name': 'high heels'
    },
    {
        'id': 340,
        'name': 'mangosteen'
    },
    {
        'id': 22,
        'name': 'bracelet'
    },
    {
        'id': 155,
        'name': 'piano'
    },
    {
        'id': 162,
        'name': 'vent'
    },
    {
        'id': 75,
        'name': 'laptop'
    },
    {
        'id': 236,
        'name': 'toaster'
    },
    {
        'id': 231,
        'name': 'fire truck'
    },
    {
        'id': 42,
        'name': 'basket'
    },
    {
        'id': 150,
        'name': 'zebra'
    },
    {
        'id': 124,
        'name': 'head phone'
    },
    {
        'id': 90,
        'name': 'sheep'
    },
    {
        'id': 322,
        'name': 'steak'
    },
    {
        'id': 39,
        'name': 'couch'
    },
    {
        'id': 209,
        'name': 'toothbrush'
    },
    {
        'id': 59,
        'name': 'bicycle'
    },
    {
        'id': 336,
        'name': 'red cabbage'
    },
    {
        'id': 228,
        'name': 'golf ball'
    },
    {
        'id': 120,
        'name': 'tomato'
    },
    {
        'id': 132,
        'name': 'computer box'
    },
    {
        'id': 8,
        'name': 'cup'
    },
    {
        'id': 183,
        'name': 'basketball'
    },
    {
        'id': 298,
        'name': 'butterfly'
    },
    {
        'id': 250,
        'name': 'garlic'
    },
    {
        'id': 12,
        'name': 'desk'
    },
    {
        'id': 141,
        'name': 'microwave'
    },
    {
        'id': 171,
        'name': 'strawberry'
    },
    {
        'id': 200,
        'name': 'kettle'
    },
    {
        'id': 63,
        'name': 'van'
    },
    {
        'id': 300,
        'name': 'cheese'
    },
    {
        'id': 215,
        'name': 'marker'
    },
    {
        'id': 100,
        'name': 'blackboard/whiteboard'
    },
    {
        'id': 186,
        'name': 'printer'
    },
    {
        'id': 333,
        'name': 'bread/bun'
    },
    {
        'id': 243,
        'name': 'penguin'
    },
    {
        'id': 364,
        'name': 'iron'
    },
    {
        'id': 180,
        'name': 'ladder'
    },
    {
        'id': 34,
        'name': 'flag'
    },
    {
        'id': 78,
        'name': 'cell phone'
    },
    {
        'id': 97,
        'name': 'fan'
    },
    {
        'id': 224,
        'name': 'scale'
    },
    {
        'id': 151,
        'name': 'duck'
    },
    {
        'id': 319,
        'name': 'flute'
    },
    {
        'id': 156,
        'name': 'stop sign'
    },
    {
        'id': 290,
        'name': 'rickshaw'
    },
    {
        'id': 128,
        'name': 'sailboat'
    },
    {
        'id': 165,
        'name': 'tennis racket'
    },
    {
        'id': 241,
        'name': 'cigar'
    },
    {
        'id': 101,
        'name': 'balloon'
    },
    {
        'id': 308,
        'name': 'hair drier'
    },
    {
        'id': 167,
        'name': 'skating and skiing shoes'
    },
    {
        'id': 237,
        'name': 'helicopter'
    },
    {
        'id': 65,
        'name': 'sink'
    },
    {
        'id': 129,
        'name': 'tangerine'
    },
    {
        'id': 330,
        'name': 'crab'
    },
    {
        'id': 320,
        'name': 'measuring cup'
    },
    {
        'id': 260,
        'name': 'fishing rod'
    },
    {
        'id': 346,
        'name': 'saw'
    },
    {
        'id': 216,
        'name': 'ship'
    },
    {
        'id': 46,
        'name': 'coffee table'
    },
    {
        'id': 194,
        'name': 'facial mask'
    },
    {
        'id': 281,
        'name': 'stapler'
    },
    {
        'id': 118,
        'name': 'refrigerator'
    },
    {
        'id': 40,
        'name': 'belt'
    },
    {
        'id': 349,
        'name': 'starfish'
    },
    {
        'id': 87,
        'name': 'hanger'
    },
    {
        'id': 116,
        'name': 'baseball glove'
    },
    {
        'id': 261,
        'name': 'cherry'
    },
    {
        'id': 334,
        'name': 'baozi'
    },
    {
        'id': 267,
        'name': 'screwdriver'
    },
    {
        'id': 158,
        'name': 'converter'
    },
    {
        'id': 335,
        'name': 'lion'
    },
    {
        'id': 170,
        'name': 'baseball'
    },
    {
        'id': 111,
        'name': 'skis'
    },
    {
        'id': 136,
        'name': 'broccoli'
    },
    {
        'id': 342,
        'name': 'eraser'
    },
    {
        'id': 337,
        'name': 'polar bear'
    },
    {
        'id': 139,
        'name': 'shovel'
    },
    {
        'id': 193,
        'name': 'extension cord'
    },
    {
        'id': 284,
        'name': 'goldfish'
    },
    {
        'id': 174,
        'name': 'pepper'
    },
    {
        'id': 138,
        'name': 'stroller'
    },
    {
        'id': 328,
        'name': 'yak'
    },
    {
        'id': 83,
        'name': 'clock'
    },
    {
        'id': 235,
        'name': 'tricycle'
    },
    {
        'id': 248,
        'name': 'parking meter'
    },
    {
        'id': 274,
        'name': 'trophy'
    },
    {
        'id': 324,
        'name': 'binoculars'
    },
    {
        'id': 51,
        'name': 'traffic light'
    },
    {
        'id': 314,
        'name': 'donkey'
    },
    {
        'id': 45,
        'name': 'barrel/bucket'
    },
    {
        'id': 292,
        'name': 'pomegranate'
    },
    {
        'id': 13,
        'name': 'handbag'
    },
    {
        'id': 262,
        'name': 'tablet'
    },
    {
        'id': 68,
        'name': 'apple'
    },
    {
        'id': 226,
        'name': 'cabbage'
    },
    {
        'id': 23,
        'name': 'flower'
    },
    {
        'id': 58,
        'name': 'faucet'
    },
    {
        'id': 206,
        'name': 'tong'
    },
    {
        'id': 291,
        'name': 'trombone'
    },
    {
        'id': 160,
        'name': 'carrot'
    },
    {
        'id': 172,
        'name': 'bow tie'
    },
    {
        'id': 122,
        'name': 'tent'
    },
    {
        'id': 163,
        'name': 'cookies'
    },
    {
        'id': 115,
        'name': 'remote'
    },
    {
        'id': 175,
        'name': 'coffee machine'
    },
    {
        'id': 238,
        'name': 'green beans'
    },
    {
        'id': 233,
        'name': 'cello'
    },
    {
        'id': 28,
        'name': 'wine glass'
    },
    {
        'id': 295,
        'name': 'mushroom'
    },
    {
        'id': 344,
        'name': 'scallop'
    },
    {
        'id': 125,
        'name': 'lantern'
    },
    {
        'id': 123,
        'name': 'shampoo/shower gel'
    },
    {
        'id': 285,
        'name': 'meat balls'
    },
    {
        'id': 266,
        'name': 'key'
    },
    {
        'id': 296,
        'name': 'calculator'
    },
    {
        'id': 168,
        'name': 'scissors'
    },
    {
        'id': 103,
        'name': 'cymbal'
    },
    {
        'id': 6,
        'name': 'bottle'
    },
    {
        'id': 264,
        'name': 'nuts'
    },
    {
        'id': 234,
        'name': 'notepaper'
    },
    {
        'id': 211,
        'name': 'mango'
    },
    {
        'id': 287,
        'name': 'toothpaste'
    },
    {
        'id': 196,
        'name': 'chopsticks'
    },
    {
        'id': 140,
        'name': 'baseball bat'
    },
    {
        'id': 244,
        'name': 'hurdle'
    },
    {
        'id': 195,
        'name': 'tennis ball'
    },
    {
        'id': 144,
        'name': 'surveillance camera'
    },
    {
        'id': 271,
        'name': 'volleyball'
    },
    {
        'id': 94,
        'name': 'keyboard'
    },
    {
        'id': 339,
        'name': 'seal'
    },
    {
        'id': 11,
        'name': 'picture/frame'
    },
    {
        'id': 348,
        'name': 'okra'
    },
    {
        'id': 191,
        'name': 'sausage'
    },
    {
        'id': 166,
        'name': 'candy'
    },
    {
        'id': 62,
        'name': 'ring'
    },
    {
        'id': 311,
        'name': 'dolphin'
    },
    {
        'id': 273,
        'name': 'eggplant'
    },
    {
        'id': 84,
        'name': 'drum'
    },
    {
        'id': 143,
        'name': 'surfboard'
    },
    {
        'id': 288,
        'name': 'antelope'
    },
    {
        'id': 204,
        'name': 'clutch'
    },
    {
        'id': 207,
        'name': 'slide'
    },
    {
        'id': 43,
        'name': 'towel/napkin'
    },
    {
        'id': 352,
        'name': 'durian'
    },
    {
        'id': 276,
        'name': 'board eraser'
    },
    {
        'id': 315,
        'name': 'electric drill'
    },
    {
        'id': 312,
        'name': 'sushi'
    },
    {
        'id': 198,
        'name': 'pie'
    },
    {
        'id': 106,
        'name': 'pickup truck'
    },
    {
        'id': 176,
        'name': 'bathtub'
    },
    {
        'id': 26,
        'name': 'vase'
    },
    {
        'id': 133,
        'name': 'elephant'
    },
    {
        'id': 256,
        'name': 'sandwich'
    },
    {
        'id': 327,
        'name': 'noodles'
    },
    {
        'id': 10,
        'name': 'glasses'
    },
    {
        'id': 109,
        'name': 'airplane'
    },
    {
        'id': 95,
        'name': 'tripod'
    },
    {
        'id': 247,
        'name': 'CD'
    },
    {
        'id': 121,
        'name': 'machinery vehicle'
    },
    {
        'id': 365,
        'name': 'flashlight'
    },
    {
        'id': 53,
        'name': 'microphone'
    },
    {
        'id': 270,
        'name': 'pliers'
    },
    {
        'id': 362,
        'name': 'chainsaw'
    },
    {
        'id': 259,
        'name': 'bear'
    },
    {
        'id': 197,
        'name': 'electronic stove and gas stove'
    },
    {
        'id': 89,
        'name': 'pot/pan'
    },
    {
        'id': 220,
        'name': 'tape'
    },
    {
        'id': 338,
        'name': 'lighter'
    },
    {
        'id': 177,
        'name': 'snowboard'
    },
    {
        'id': 214,
        'name': 'violin'
    },
    {
        'id': 217,
        'name': 'chicken'
    },
    {
        'id': 2,
        'name': 'sneakers'
    },
    {
        'id': 161,
        'name': 'washing machine'
    },
    {
        'id': 131,
        'name': 'kite'
    },
    {
        'id': 354,
        'name': 'rabbit'
    },
    {
        'id': 86,
        'name': 'bus'
    },
    {
        'id': 275,
        'name': 'dates'
    },
    {
        'id': 282,
        'name': 'camel'
    },
    {
        'id': 88,
        'name': 'nightstand'
    },
    {
        'id': 179,
        'name': 'grapes'
    },
    {
        'id': 229,
        'name': 'pine apple'
    },
    {
        'id': 56,
        'name': 'necklace'
    },
    {
        'id': 18,
        'name': 'leather shoes'
    },
    {
        'id': 358,
        'name': 'hoverboard'
    },
    {
        'id': 345,
        'name': 'pencil case'
    },
    {
        'id': 359,
        'name': 'pasta'
    },
    {
        'id': 157,
        'name': 'radiator'
    },
    {
        'id': 201,
        'name': 'hamburger'
    },
    {
        'id': 268,
        'name': 'globe'
    },
    {
        'id': 332,
        'name': 'barbell'
    },
    {
        'id': 329,
        'name': 'mop'
    },
    {
        'id': 252,
        'name': 'horn'
    },
    {
        'id': 350,
        'name': 'eagle'
    },
    {
        'id': 169,
        'name': 'folder'
    },
    {
        'id': 137,
        'name': 'toilet'
    },
    {
        'id': 5,
        'name': 'lamp'
    },
    {
        'id': 27,
        'name': 'bench'
    },
    {
        'id': 249,
        'name': 'swan'
    },
    {
        'id': 76,
        'name': 'knife'
    },
    {
        'id': 341,
        'name': 'comb'
    },
    {
        'id': 64,
        'name': 'watch'
    },
    {
        'id': 105,
        'name': 'telephone'
    },
    {
        'id': 3,
        'name': 'chair'
    },
    {
        'id': 33,
        'name': 'boat'
    },
    {
        'id': 107,
        'name': 'orange'
    },
    {
        'id': 60,
        'name': 'bread'
    },
    {
        'id': 147,
        'name': 'cat'
    },
    {
        'id': 135,
        'name': 'gas stove'
    },
    {
        'id': 307,
        'name': 'papaya'
    },
    {
        'id': 227,
        'name': 'router/modem'
    },
    {
        'id': 357,
        'name': 'asparagus'
    },
    {
        'id': 73,
        'name': 'motorcycle'
    },
    {
        'id': 77,
        'name': 'traffic sign'
    },
    {
        'id': 67,
        'name': 'fish'
    },
    {
        'id': 326,
        'name': 'radish'
    },
    {
        'id': 213,
        'name': 'egg'
    },
    {
        'id': 203,
        'name': 'cucumber'
    },
    {
        'id': 17,
        'name': 'helmet'
    },
    {
        'id': 110,
        'name': 'luggage'
    },
    {
        'id': 80,
        'name': 'truck'
    },
    {
        'id': 199,
        'name': 'frisbee'
    },
    {
        'id': 232,
        'name': 'peach'
    },
    {
        'id': 1,
        'name': 'person'
    },
    {
        'id': 29,
        'name': 'boots'
    },
    {
        'id': 310,
        'name': 'chips'
    },
    {
        'id': 142,
        'name': 'skateboard'
    },
    {
        'id': 44,
        'name': 'slippers'
    },
    {
        'id': 4,
        'name': 'hat'
    },
    {
        'id': 178,
        'name': 'suitcase'
    },
    {
        'id': 24,
        'name': 'tv'
    },
    {
        'id': 119,
        'name': 'train'
    },
    {
        'id': 82,
        'name': 'power outlet'
    },
    {
        'id': 245,
        'name': 'swing'
    },
    {
        'id': 15,
        'name': 'book'
    },
    {
        'id': 294,
        'name': 'jellyfish'
    },
    {
        'id': 192,
        'name': 'fire extinguisher'
    },
    {
        'id': 212,
        'name': 'deer'
    },
    {
        'id': 181,
        'name': 'pear'
    },
    {
        'id': 347,
        'name': 'table tennis paddle'
    },
    {
        'id': 113,
        'name': 'trolley'
    },
    {
        'id': 91,
        'name': 'guitar'
    },
    {
        'id': 202,
        'name': 'golf club'
    },
    {
        'id': 221,
        'name': 'wheelchair'
    },
    {
        'id': 254,
        'name': 'saxophone'
    },
    {
        'id': 117,
        'name': 'paper towel'
    },
    {
        'id': 303,
        'name': 'race car'
    },
    {
        'id': 240,
        'name': 'carriage'
    },
    {
        'id': 246,
        'name': 'radio'
    },
    {
        'id': 318,
        'name': 'parrot'
    },
    {
        'id': 251,
        'name': 'french fries'
    },
    {
        'id': 98,
        'name': 'dog'
    },
    {
        'id': 112,
        'name': 'soccer'
    },
    {
        'id': 355,
        'name': 'french horn'
    },
    {
        'id': 79,
        'name': 'paddle'
    },
    {
        'id': 283,
        'name': 'lettuce'
    },
    {
        'id': 9,
        'name': 'car'
    },
    {
        'id': 258,
        'name': 'kiwi fruit'
    },
    {
        'id': 325,
        'name': 'llama'
    },
    {
        'id': 187,
        'name': 'billiards'
    },
    {
        'id': 210,
        'name': 'facial cleanser'
    },
    {
        'id': 81,
        'name': 'cow'
    },
    {
        'id': 331,
        'name': 'microscope'
    },
    {
        'id': 148,
        'name': 'lemon'
    },
    {
        'id': 302,
        'name': 'pomelo'
    },
    {
        'id': 85,
        'name': 'fork'
    },
    {
        'id': 154,
        'name': 'pumpkin'
    },
    {
        'id': 289,
        'name': 'shrimp'
    },
    {
        'id': 71,
        'name': 'teddy bear'
    },
    {
        'id': 184,
        'name': 'potato'
    },
    {
        'id': 102,
        'name': 'air conditioner'
    },
    {
        'id': 208,
        'name': 'hot dog'
    },
    {
        'id': 222,
        'name': 'plum'
    },
    {
        'id': 316,
        'name': 'spring rolls'
    },
    {
        'id': 230,
        'name': 'crane'
    },
    {
        'id': 149,
        'name': 'liquid soap'
    },
    {
        'id': 55,
        'name': 'canned'
    },
    {
        'id': 35,
        'name': 'speaker'
    },
    {
        'id': 108,
        'name': 'banana'
    },
    {
        'id': 297,
        'name': 'treadmill'
    },
    {
        'id': 99,
        'name': 'spoon'
    },
    {
        'id': 104,
        'name': 'mouse'
    },
    {
        'id': 182,
        'name': 'american football'
    },
    {
        'id': 299,
        'name': 'egg tart'
    },
    {
        'id': 127,
        'name': 'cleaning products'
    },
    {
        'id': 313,
        'name': 'urinal'
    },
    {
        'id': 286,
        'name': 'medal'
    },
    {
        'id': 239,
        'name': 'brush'
    },
    {
        'id': 96,
        'name': 'hockey'
    },
    {
        'id': 279,
        'name': 'dumbbell'
    },
    {
        'id': 32,
        'name': 'umbrella'
    },
    {
        'id': 272,
        'name': 'hammer'
    },
    {
        'id': 16,
        'name': 'plate'
    },
    {
        'id': 21,
        'name': 'potted plant'
    },
    {
        'id': 242,
        'name': 'earphone'
    },
    {
        'id': 70,
        'name': 'candle'
    },
    {
        'id': 185,
        'name': 'paint brush'
    },
    {
        'id': 48,
        'name': 'toy'
    },
    {
        'id': 130,
        'name': 'pizza'
    },
    {
        'id': 255,
        'name': 'trumpet'
    },
    {
        'id': 361,
        'name': 'hotair balloon'
    },
    {
        'id': 188,
        'name': 'fire hydrant'
    },
    {
        'id': 50,
        'name': 'bed'
    },
    {
        'id': 253,
        'name': 'avocado'
    },
    {
        'id': 293,
        'name': 'coconut'
    },
    {
        'id': 257,
        'name': 'cue'
    },
    {
        'id': 280,
        'name': 'hamimelon'
    },
    {
        'id': 66,
        'name': 'horse'
    },
    {
        'id': 173,
        'name': 'pigeon'
    },
    {
        'id': 190,
        'name': 'projector'
    },
    {
        'id': 69,
        'name': 'camera'
    },
    {
        'id': 30,
        'name': 'bowl'
    },
    {
        'id': 269,
        'name': 'broom'
    },
    {
        'id': 343,
        'name': 'pitaya'
    },
    {
        'id': 305,
        'name': 'tuba'
    },
    {
        'id': 309,
        'name': 'green onion'
    },
    {
        'id': 363,
        'name': 'lobster'
    },
    {
        'id': 225,
        'name': 'watermelon'
    },
    {
        'id': 47,
        'name': 'suv'
    },
    {
        'id': 31,
        'name': 'dining table'
    },
    {
        'id': 54,
        'name': 'sandals'
    },
    {
        'id': 351,
        'name': 'monkey'
    },
    {
        'id': 218,
        'name': 'onion'
    },
    {
        'id': 36,
        'name': 'trash bin/can'
    },
    {
        'id': 20,
        'name': 'glove'
    },
    {
        'id': 277,
        'name': 'rice'
    },
    {
        'id': 152,
        'name': 'sports car'
    },
    {
        'id': 360,
        'name': 'target'
    },
    {
        'id': 205,
        'name': 'blender'
    },
    {
        'id': 19,
        'name': 'pillow'
    },
    {
        'id': 72,
        'name': 'cake'
    },
    {
        'id': 93,
        'name': 'tea pot'
    },
    {
        'id': 353,
        'name': 'game board'
    },
    {
        'id': 38,
        'name': 'backpack'
    },
    {
        'id': 356,
        'name': 'ambulance'
    },
    {
        'id': 146,
        'name': 'life saver'
    },
    {
        'id': 189,
        'name': 'goose'
    },
    {
        'id': 278,
        'name': 'tape measure/ruler'
    },
    {
        'id': 92,
        'name': 'traffic cone'
    },
    {
        'id': 134,
        'name': 'toiletries'
    },
    {
        'id': 114,
        'name': 'oven'
    },
    {
        'id': 317,
        'name': 'tortoise/turtle'
    },
    {
        'id': 265,
        'name': 'corn'
    },
    {
        'id': 126,
        'name': 'donut'
    },
    {
        'id': 57,
        'name': 'mirror'
    },
    {
        'id': 7,
        'name': 'cabinet/shelf'
    },
    {
        'id': 263,
        'name': 'green vegetables'
    },
    {
        'id': 159,
        'name': 'tissue '
    },
    {
        'id': 321,
        'name': 'shark'
    },
    {
        'id': 301,
        'name': 'pig'
    },
    {
        'id': 41,
        'name': 'carpet'
    },
    {
        'id': 304,
        'name': 'rice cooker'
    },
    {
        'id': 323,
        'name': 'poker card'
    },
]


def _get_builtin_metadata(version):
    if version == 'v1':
        id_to_name = {x['id']: x['name'] for x in categories_v1}
    else:
        assert 0, version
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(365)}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        'thing_dataset_id_to_contiguous_id': thing_dataset_id_to_contiguous_id,
        'thing_classes': thing_classes
    }


_PREDEFINED_SPLITS_OBJECTS365 = {
    'objects365_train':
    ('objects365/train', 'objects365/annotations/objects365_train.json'),
    'objects365_val':
    ('objects365/val', 'objects365/annotations/objects365_val.json'),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata('v1'),
        os.path.join('datasets', json_file)
        if '://' not in json_file else json_file,
        os.path.join('datasets', image_root),
    )
