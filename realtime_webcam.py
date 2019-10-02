"""
Using Webcam for realtime inference.

"""
import os
import numpy as np
from imageio import imwrite
from PIL import Image
import tensorflow as tf
from logger import get_logger
import cv2


# NOTE: TF warnings are too noisy without this
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(40)


def main(m_path, resize_path, out_dir):

    imported = tf.saved_model.load(m_path)

    cv2.namedWindow("original")
    cv2.namedWindow("cartoon")

    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        
        scale_percent = 20 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_ = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        f = imported.signatures["serving_default"]
        resized = np.array(resized_)
        resized = np.expand_dims(resized_, 0).astype(np.float32) / 127.5 - 1

        out = f(tf.constant(resized))['output_1']
        out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
        img = Image.fromarray(out, 'RGB')

        cv2.imshow("original", resized_)
        cv2.imshow("cartoon", np.asarray(img))

        rval, frame = vc.read()



        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("original")
    cv2.destroyWindow("cartoon")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str,
                        default=os.path.join("exported_models", "light_shinkai_SavedModel"))
    parser.add_argument("--resize_path", type=str,
                        default=os.path.join("input_images", "me.jpg"))
    parser.add_argument("--out_dir", type=str, default='out')
    args = parser.parse_args()
    main(args.m_path, args.resize_path, args.out_dir)
