#! python

# Based on a processing script created by Aleksei Wan on 13.11.2019
# Adapted for use with this Project by Aleksei

# Imports
import os
import sys
from argparse import ArgumentParser
import json


# Takes in raw AWS emotions data struct, returns list sorted by confidence (highest to lowest)
def read_aws_emotions(raw_emotions: list) -> list:
    ret_list = [(em['Type'], em['Confidence']) for em in raw_emotions]
    ret_list.sort(key=lambda t: t[1], reverse=True)
    return ret_list


def first_element_list_from_tuple_list(in_list: list) -> list:
    return [t[0] for t in in_list]


if __name__ == "__main__":
    parser = ArgumentParser(description='Process results from AWS to determine correct classification')
    parser.add_argument('--input_file', type=str, default='cached_aws_info.json', help='File containing results')
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    ans_dict = {}
    for e in os.listdir('../data/test/'):
        if not e.startswith('.'):
            name = e.upper()
            if name == 'FEARFUL':
                name = 'FEAR'
            elif name == 'DISGUST':
                name = 'DISGUSTED'
            for f in os.listdir(os.path.join('../data/test/', e)):
                ans_dict[os.path.splitext(f)[0]] = name

    total = 0
    direct_match_counter = 0
    second_match_counter = 0
    emotion_not_covered = 0
    surprise_counter, nc, af = 0, 0, 0
    for e in contents:
        actual_emotion = ans_dict[e[0]]
        if actual_emotion == 'SURPRISED':
            surprise_counter += 1
            continue
        aws_emotions = first_element_list_from_tuple_list(read_aws_emotions(e[1]['FaceDetails'][0]['Emotions']))
        # Check if top emotion matches
        print('Actual:', actual_emotion, '\nAWS:', aws_emotions)
        if actual_emotion == aws_emotions[0]:
            direct_match_counter += 1
        # elif actual_emotion == aws_emotions[1] or actual_emotion == aws_emotions[2]:
        #     second_match_counter += 1
        elif actual_emotion == 'NEUTRAL' and aws_emotions[0] == 'CALM':
            nc += 1
            second_match_counter += 1
        elif (actual_emotion == 'ANGRY' and aws_emotions[0] == 'FEAR') or (actual_emotion == 'FEAR' and aws_emotions[0] == 'ANGRY'):
            af += 1
            second_match_counter += 1
        else:
            if actual_emotion not in aws_emotions:
                emotion_not_covered += 1
        total += 1

    total_for_comp = float(total - emotion_not_covered)
    def make_printable(x): return round(100 * float(x), 3)
    print('Surprised:', surprise_counter, 'NeutralCalm:', nc, 'AngryFearful:', af)
    print('Number without direct match:', emotion_not_covered)
    print('First Preference Accuracy', make_printable(direct_match_counter / total_for_comp), '%')
    print('Adjusted Match Accuracy', make_printable((direct_match_counter + second_match_counter) / total_for_comp), '%')
