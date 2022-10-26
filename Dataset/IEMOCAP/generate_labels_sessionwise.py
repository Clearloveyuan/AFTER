import os
import numpy as np
import json
import random
from pathlib import Path

Path('labels_sess').mkdir(exist_ok=True)
with open('metalabel.json', 'r') as f:
    metalabel = json.load(f)
labeldict = {
    'ang': 'anger',
    'hap': 'happy',
    'exc': 'happy',
    'sad': 'sad',
    'neu': 'neutral'
#    'fru': 'frustration'
#    'dis': 'disgust',
#    'fea': 'fear',
#    'sur': 'surprise'
}

def in_session(speakerset, audioname):
    audio_gender = audioname[-8]
    audio_session = audioname[4]
    for speaker in speakerset:
        gender = speaker[0]
        session = speaker[1]
        if gender == audio_gender and session == audio_session:
            return True
    return False

all_speakers = []
test_data = [1085, 1023, 1151, 1031, 1241]
for i in range(5):
    count1 = 0
    count2 = 0
    sess = i + 1
    test_speakers = [f'M{sess}', f'F{sess}']
    labels = {
        'Train': {},
        'Val': {},
        'Test': {}
    }
    for audio in os.listdir('Audio_16k'):
        label_key = metalabel[audio]
        if label_key not in labeldict:
            continue
        label = labeldict[label_key]
        if in_session(test_speakers, audio) and count1 <= test_data[i]/2 :
            labels['Test'][audio] = label
            count1 += 1
        elif in_session(test_speakers, audio) and count1 > test_data[i]/2 :
            labels["Val"][audio] = label
            count1 += 1
        else:
            labels['Train'][audio] = label
            count2 += 1
    # print(f"session: {i}, and the test data are {count1}, the train data are {count2}")
            
    with open(f'labels_sess/label_{i+1}.json', 'w') as f:
        json.dump(labels, f, indent=4)
