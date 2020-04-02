#! python

# Created by Aleksei Wan on 28.10.2019, used with author's consent, all rights reserved

# Imports
import os
from sys import exit
import re
from argparse import ArgumentParser
import boto3
import json

# regex for image file matching
IMG_FILE = re.compile('(.*)\.jp[e]?g$')


# Note: This depends on your AWS credentials and configs being set up
def detect_labels_local_file(photo, full_responses: bool):
    """
    :param photo: photo file
    :param full_responses: return full responses or just the emotion section
    :return: list of labels
    """
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])

    return response if full_responses else response['FaceDetails'][0]['Emotions']


def detect_photos_in_dir(directory: str, full_responses: bool) -> list:
    """
    :param directory: directory to evaluate
    :param full_responses: return full responses or just the emotion section
    :return: list of pairs of photos and tags
    """
    labels = list()
    files = os.listdir(directory)
    num_files = str(len(files))

    print('Using', directory, 'as images directory...')
    print(num_files, 'file(s) found.')

    tracking = 0
    for photo_name in files:
        tracking += 1
        print('(' + str(tracking) + '/' + num_files + ')', end=' ')
        img_file = re.match(IMG_FILE, photo_name)
        if img_file:
            print('Evaluating image file', img_file[1])
            photo = os.path.join(directory, photo_name)
            labels.append((img_file[1], detect_labels_local_file(photo, full_responses)))
        else:
            print('Ignoring non-image file', photo_name)

    return labels


if __name__ == "__main__":
    parser = ArgumentParser(description='Send images to AWS for processing')
    parser.add_argument('--input_dir', type=str, default='images', help='Dir containing images for classification')
    parser.add_argument('--output_file', type=str, help='File to cache results', required=False)
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_dir)

    get_full_responses = args.output_file if args.output_file is not None else False

    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        exit(-1)

    photo_labels = detect_photos_in_dir(path_to_check, get_full_responses)

    if get_full_responses:
        with open(args.output_file, 'w+') as f:
            json.dump(photo_labels, f)
    else:
        print(photo_labels)
