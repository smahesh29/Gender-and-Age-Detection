from utils.detect import display_frame
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args=parser.parse_args()
    video = cv2.VideoCapture(args.image if args.image else 0)
    padding : int = 20
    display_frame(video, padding)
    cv2.destroyAllWindows() 