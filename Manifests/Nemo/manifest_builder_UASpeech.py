#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:44:52 2021

@author: u20020048
"""

import json 
import librosa
import os 
import soundfile as sf

vocab = {
"D3": "Three",
"D9": "Nine",
"D0": "Zero",
"D6": "Six",
"D7": "Seven",
"D8": "Eight",
"D4": "Four",
"D5": "Five",
"D1": "One",
"D2": "Two",
"LE": "Echo",
"LD": "Delta",
"LW": "Whiskey",
"LK": "Kilo",
"LS": "Sierra",
"LT": "Tango",
"LU": "Uniform",
"LX": "X-ray",
"LJ": "Juliet",
"LC": "Charlie",
"LQ": "Quebec",
"LP": "Papa",
"LZ": "Zulu",
"LR": "Romeo",
"LF": "Foxtrot",
"LI": "India",
"LL": "Lima",
"LV": "Victor",
"LY": "Yankee",
"LG": "Golf",
"LH": "Hotel",
"LN": "November",
"LB": "Bravo",
"LO": "Oscar",
"LA": "Alpha",
"LM": "Mike",
"C1": "Command",
"C2": "Backspace",
"C3": "Delete",
"C4": "Enter",
"C5": "Tab",
"C6": "Escape",
"C7": "Alt",
"C8": "Control",
"C9": "Shift",
"C10": "Line",
"C11": "Paragraph",
"C12": "Sentence",
"C13": "Paste",
"C14": "Cut",
"C15": "Copy",
"C16": "Upward",
"C17": "Downward",
"C18": "Left",
"C19": "Right",
"CW1": "the",
"CW2": "of",
"CW3": "and",
"CW4": "a",
"CW5": "to",
"CW6": "in",
"CW7": "is",
"CW8": "you",
"CW9": "that",
"CW10": "it",
"CW11": "he",
"CW12": "was",
"CW13": "for",
"CW14": "on",
"CW15": "are",
"CW16": "as",
"CW17": "with",
"CW18": "his",
"CW19": "they",
"CW20": "I",
"CW21": "at",
"CW22": "be",
"CW23": "this",
"CW24": "have",
"CW25": "from",
"CW26": "or",
"CW27": "had",
"CW28": "by",
"CW29": "word",
"CW30": "but",
"CW31": "not",
"CW32": "what",
"CW33": "all",
"CW34": "were",
"CW35": "we",
"CW36": "when",
"CW37": "your",
"CW38": "can",
"CW39": "said",
"CW40": "there",
"CW41": "use",
"CW42": "an",
"CW43": "each",
"CW44": "which",
"CW45": "she",
"CW46": "do",
"CW47": "how",
"CW48": "their",
"CW49": "if",
"CW50": "will",
"CW51": "up",
"CW52": "other",
"CW53": "about",
"CW54": "out",
"CW55": "many",
"CW56": "then",
"CW57": "them",
"CW58": "these",
"CW59": "so",
"CW60": "some",
"CW61": "her",
"CW62": "would",
"CW63": "make",
"CW64": "like",
"CW65": "him",
"CW66": "into",
"CW67": "time",
"CW68": "has",
"CW69": "look",
"CW70": "more",
"CW71": "write",
"CW72": "go",
"CW73": "see",
"CW74": "number",
"CW75": "no",
"CW76": "way",
"CW77": "could",
"CW78": "people",
"CW79": "my",
"CW80": "than",
"CW81": "first",
"CW82": "water",
"CW83": "been",
"CW84": "call",
"CW85": "who",
"CW86": "oil",
"CW87": "its",
"CW88": "now",
"CW89": "find",
"CW90": "long",
"CW91": "down",
"CW92": "day",
"CW93": "did",
"CW94": "get",
"CW95": "come",
"CW96": "made",
"CW97": "may",
"CW98": "part",
"CW99": "oh",
"CW100": "yes",
"B1_UW1": "naturalization",
"B1_UW2": "faithfulness",
"B1_UW3": "frugality",
"B1_UW4": "irresolute",
"B1_UW5": "psychological",
"B1_UW6": "supervision",
"B1_UW7": "able-bodied",
"B1_UW8": "bushels",
"B1_UW9": "cowboys",
"B1_UW10": "entertainer",
"B1_UW11": "gentlewoman",
"B1_UW12": "giggled",
"B1_UW13": "hypothesis",
"B1_UW14": "lethargy",
"B1_UW15": "merchandise",
"B1_UW16": "nevertheless",
"B1_UW17": "Pennsylvania",
"B1_UW18": "rejoinder",
"B1_UW19": "southeast",
"B1_UW20": "vouchsafe",
"B1_UW21": "achieve",
"B1_UW22": "adapt",
"B1_UW23": "autobiography",
"B1_UW24": "avenues",
"B1_UW25": "babyhood",
"B1_UW26": "behavior",
"B1_UW27": "bogies",
"B1_UW28": "boquets",
"B1_UW29": "casualties",
"B1_UW30": "cheshire",
"B1_UW31": "choking",
"B1_UW32": "circular",
"B1_UW33": "composure",
"B1_UW34": "epithet",
"B1_UW35": "equilibrium",
"B1_UW36": "exactitude",
"B1_UW37": "eyebrows",
"B1_UW38": "footmarks",
"B1_UW39": "fowler",
"B1_UW40": "ha-ha",
"B1_UW41": "inalienable",
"B1_UW42": "inquirers",
"B1_UW43": "jackdaws",
"B1_UW44": "journalism",
"B1_UW45": "legislature",
"B1_UW46": "moustache",
"B1_UW47": "overshadowed",
"B1_UW48": "perpetual",
"B1_UW49": "powwow",
"B1_UW50": "promulgate",
"B1_UW51": "python",
"B1_UW52": "adulation",
"B1_UW53": "advice",
"B1_UW54": "amethysts",
"B1_UW55": "annoyed",
"B1_UW56": "ashamed",
"B1_UW57": "Asia",
"B1_UW58": "azure",
"B1_UW59": "beguile",
"B1_UW60": "Birmingham",
"B1_UW61": "burrows",
"B1_UW62": "chatterbox",
"B1_UW63": "chauffeur",
"B1_UW64": "choice",
"B1_UW65": "chowder",
"B1_UW66": "cobblestones",
"B1_UW67": "Copenhagen",
"B1_UW68": "dissatisfaction",
"B1_UW69": "employment",
"B1_UW70": "futurity",
"B1_UW71": "gigantic",
"B1_UW72": "gouged",
"B1_UW73": "goulash",
"B1_UW74": "hallelujah",
"B1_UW75": "haranguing",
"B1_UW76": "hulk",
"B1_UW77": "immovable",
"B1_UW78": "interwoven",
"B1_UW79": "Joseph",
"B1_UW80": "massage",
"B1_UW81": "moisten",
"B1_UW82": "house",
"B1_UW83": "tree",
"B1_UW84": "window",
"B1_UW85": "telephone",
"B1_UW86": "cup",
"B1_UW87": "knife",
"B1_UW88": "spoon",
"B1_UW89": "girl",
"B1_UW90": "ball",
"B1_UW91": "wagon",
"B1_UW92": "shovel",
"B1_UW93": "monkey",
"B1_UW94": "banana",
"B1_UW95": "zipper",
"B1_UW96": "scissors",
"B1_UW97": "duck",
"B1_UW98": "quack",
"B1_UW99": "yellow",
"B1_UW100": "vacuum",
"B2_UW1": "mouth",
"B2_UW2": "needlepoint",
"B2_UW3": "Nuremberg",
"B2_UW4": "pennyworth",
"B2_UW5": "re-united",
"B2_UW6": "roof",
"B2_UW7": "schoolyard",
"B2_UW8": "sharpshooter",
"B2_UW9": "skyward",
"B2_UW10": "sugar",
"B2_UW11": "supreme",
"B2_UW12": "toothache",
"B2_UW13": "unusual",
"B2_UW14": "voyage",
"B2_UW15": "abbreviated",
"B2_UW16": "ablutions",
"B2_UW17": "absolve",
"B2_UW18": "absorb",
"B2_UW19": "adhesion",
"B2_UW20": "adjacent",
"B2_UW21": "advantageous",
"B2_UW22": "agricultural",
"B2_UW23": "allure",
"B2_UW24": "aloft",
"B2_UW25": "aloof",
"B2_UW26": "although",
"B2_UW27": "anxieties",
"B2_UW28": "anybody",
"B2_UW29": "anything",
"B2_UW30": "apothecary",
"B2_UW31": "appreciable",
"B2_UW32": "apprehend",
"B2_UW33": "approach",
"B2_UW34": "astounded",
"B2_UW35": "atrocious",
"B2_UW36": "authoritative",
"B2_UW37": "aversion",
"B2_UW38": "bachelor",
"B2_UW39": "bathe",
"B2_UW40": "baths",
"B2_UW41": "battlefield",
"B2_UW42": "battlements",
"B2_UW43": "battleship",
"B2_UW44": "beef",
"B2_UW45": "beleaguering",
"B2_UW46": "Bengal",
"B2_UW47": "bequeath",
"B2_UW48": "betook",
"B2_UW49": "betroth",
"B2_UW50": "blithe",
"B2_UW51": "bloodshed",
"B2_UW52": "bluebells",
"B2_UW53": "booth",
"B2_UW54": "bosom",
"B2_UW55": "both",
"B2_UW56": "bother",
"B2_UW57": "boulevard",
"B2_UW58": "boyhood",
"B2_UW59": "broil",
"B2_UW60": "brotherhood",
"B2_UW61": "buffoon",
"B2_UW62": "bulge",
"B2_UW63": "bulrush",
"B2_UW64": "butcher",
"B2_UW65": "butterflies",
"B2_UW66": "candlelight",
"B2_UW67": "Catholic",
"B2_UW68": "celebrity",
"B2_UW69": "chide",
"B2_UW70": "coherent",
"B2_UW71": "coil",
"B2_UW72": "convulsion",
"B2_UW73": "cowhide",
"B2_UW74": "deluge",
"B2_UW75": "digest",
"B2_UW76": "displeasure",
"B2_UW77": "dispossess",
"B2_UW78": "dowry",
"B2_UW79": "durable",
"B2_UW80": "Emilio",
"B2_UW81": "endowments",
"B2_UW82": "watch",
"B2_UW83": "plane",
"B2_UW84": "swimming",
"B2_UW85": "watches",
"B2_UW86": "lamp",
"B2_UW87": "car",
"B2_UW88": "blue",
"B2_UW89": "rabbit",
"B2_UW90": "carrot",
"B2_UW91": "orange",
"B2_UW92": "fishing",
"B2_UW93": "chair",
"B2_UW94": "feather",
"B2_UW95": "pencil",
"B2_UW96": "bathtub",
"B2_UW97": "bath",
"B2_UW98": "ring",
"B2_UW99": "finger",
"B2_UW100": "thumb",
"B3_UW1": "enthuse",
"B3_UW2": "exploit",
"B3_UW3": "Fayetteville",
"B3_UW4": "foil",
"B3_UW5": "good",
"B3_UW6": "greyhound",
"B3_UW7": "hanger",
"B3_UW8": "Heyward",
"B3_UW9": "hoist",
"B3_UW10": "Iroquois",
"B3_UW11": "jowls",
"B3_UW12": "joyful",
"B3_UW13": "juries",
"B3_UW14": "lawyer",
"B3_UW15": "Missouri",
"B3_UW16": "moonshine",
"B3_UW17": "Nathaniel",
"B3_UW18": "nodule",
"B3_UW19": "nowhere",
"B3_UW20": "output",
"B3_UW21": "ploughshare",
"B3_UW22": "protege",
"B3_UW23": "refugee",
"B3_UW24": "regime",
"B3_UW25": "rendezvous",
"B3_UW26": "resound",
"B3_UW27": "reward",
"B3_UW28": "righteous",
"B3_UW29": "Sawyer",
"B3_UW30": "scullion",
"B3_UW31": "shout",
"B3_UW32": "stringy",
"B3_UW33": "swoon",
"B3_UW34": "sympathize",
"B3_UW35": "thine",
"B3_UW36": "thou",
"B3_UW37": "toil",
"B3_UW38": "unyielding",
"B3_UW39": "wherewithal",
"B3_UW40": "without",
"B3_UW41": "Yale",
"B3_UW42": "Massachusetts",
"B3_UW43": "designate",
"B3_UW44": "forgetfulness",
"B3_UW45": "quadrupled",
"B3_UW46": "southeasterly",
"B3_UW47": "berserker",
"B3_UW48": "chambermaid",
"B3_UW49": "demolish",
"B3_UW50": "exaggerate",
"B3_UW51": "multiflora",
"B3_UW52": "observation",
"B3_UW53": "professionals",
"B3_UW54": "roly-poly",
"B3_UW55": "thirty-five",
"B3_UW56": "vouchsafe",
"B3_UW57": "washerwoman",
"B3_UW58": "afterthought",
"B3_UW59": "awhile",
"B3_UW60": "browbeat",
"B3_UW61": "bungalows",
"B3_UW62": "choking",
"B3_UW63": "clergyman",
"B3_UW64": "commercial",
"B3_UW65": "convenience",
"B3_UW66": "cushy",
"B3_UW67": "dodgers",
"B3_UW68": "episode",
"B3_UW69": "equilibrium",
"B3_UW70": "fathom",
"B3_UW71": "Gustave",
"B3_UW72": "homeopath",
"B3_UW73": "inexhaustible",
"B3_UW74": "Judith",
"B3_UW75": "Morgantown",
"B3_UW76": "moustache",
"B3_UW77": "nausea",
"B3_UW78": "newfound",
"B3_UW79": "powwow",
"B3_UW80": "praiseworthy",
"B3_UW81": "sandpipers",
"B3_UW82": "soothing",
"B3_UW83": "jumping",
"B3_UW84": "pajamas",
"B3_UW85": "flowers",
"B3_UW86": "brush",
"B3_UW87": "drum",
"B3_UW88": "frog",
"B3_UW89": "clown",
"B3_UW90": "green",
"B3_UW91": "balloons",
"B3_UW92": "crying",
"B3_UW93": "glasses",
"B3_UW94": "slide",
"B3_UW95": "stars",
"B3_UW96": "fire",
"B3_UW97": "watch",
"B3_UW98": "ahead",
"B3_UW99": "away",
"B3_UW100": "crayon"}


train_path = "./UASpeech-train-manifest.json"
test_path = "./UASpeech-test-manifest.json"
audio_dir = "../../Datasets/UASpeech/audio/"

TrainDir = ["M01", "M04", "M05", "M07", "M09", "M10", "M11", "F02", "F03"]
TestDir = ["M12", "M14", "M16", "F04", "F05"]

count = 0
with open(train_path, "w") as fTrain:
    with open(test_path, "w") as fTest:
        for subdir, dirs, files in os.walk(audio_dir):
            count += 1

            print(len(files))
            for file in files:

                audio_path = subdir + "/" + file
                
                # Skipping control speakers and non-audio files
                if ("control" in subdir or "html" in file):
                    continue
                speaker = file[0:3]
                word_code = file[4:-7]

                # Flagging the right set (train for B1 & B3, test for B2)
                isTrain = False if word_code[1] == "2" else True

                # Removing the Block code for non uncommon words (they're the same across the 3 blocks)
                if ("UW" not in word_code):
                    word_code = word_code[3:]
                
                # print(vocab[word_code])
                transcript = vocab[word_code]
                duration = librosa.core.get_duration(filename=audio_path)

                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }

                if isTrain:
                    json.dump(metadata, fTrain)
                    fTrain.write("\n")
                else:
                    json.dump(metadata, fTest)
                    fTest.write("\n")
