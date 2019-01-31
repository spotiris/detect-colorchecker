#!/usr/bin/env python3
import argparse
import os
import sys

import keras
import numpy as np

from keras_applications.mobilenet_v2 import preprocess_input
# from snark.imaging import cv_image

def load_model(src_dir):
    from keras.models import model_from_json
    with open(os.path.join(src_dir, "model.json"), "r") as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(os.path.join(src_dir, "weights.h5"))
    return model


# def cv_cat_gen(stream=sys.stdin.buffer):
#   for pair in cv_image.iterator(stream):
#       yield pair.header, pair.data


def filesystem_gen(stream=sys.stdin, output_shape=None):
    from PIL import Image
    for line in stream:
        line = line.strip()
        image = Image.open(line)
        if output_shape is not None:
            image = image.resize(output_shape[:2])
        image = np.array(image)
        yield line, image


def main(args):
    print("Loading model:", args.model_dir, file=sys.stderr)
    model = load_model(src_dir=args.model_dir)
    batch = []
    batch_headers = []
    # for header, image in cv_cat_gen():
    for header, image in filesystem_gen(output_shape=args.input_shape):
        print("<<", header, file=sys.stderr)
        if len(batch) == args.batch_size:
            images = preprocess_input(np.array(batch, dtype=np.float32))
            predicted = model.predict(images)[..., 1]
            for head, pred in zip(batch_headers, predicted):
                print("{:s},{:.2f}".format(str(head), pred), file=sys.stdout)
            batch = []
            batch_headers = []
        else:
            batch.append(image)
            batch_headers.append(header)
        sys.stdout.flush()

def get_args():
    '''Get args from the command line args'''
    parser = argparse.ArgumentParser(
        description="Takes image paths from STDIN and outputs to STDOUT the confidence that they contain a Macbeth checkerboard.")
    parser.add_argument("model_dir", type=str, help="The directory that houses model.json and weights.h5")
    parser.add_argument("--batch-size", type=int, help="Batch size to use on GPU, higher gives slightly more performance", default=10)
    parser.add_argument("--input-shape", type=str, help="Model input shape in CSV format (default '224,224,3').", default="224,224,3")
    args = parser.parse_args()
    args.input_shape = tuple([int(i) for i in args.input_shape.split(",")])
    return args

if __name__ == "__main__":
    main(get_args())
