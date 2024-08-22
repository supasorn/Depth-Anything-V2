import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

parser = argparse.ArgumentParser(description='Depth Anything V2')

parser.add_argument('--path', type=str, default='/data/supasorn/img3dviewer/images')
parser.add_argument('--img-path', type=str)
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--outdir', type=str, default='./vis_depth')

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

args = parser.parse_args()

def estimate(depth_anything, infile, outfile):
    print("reading", infile)

    try:
        image = cv2.imread(infile)
        h, w = image.shape[:2]
    except:
        return


    depth = depth_anything.infer_image(image, args.input_size)
    # depth = depth_anything.infer_image(image, 518)
    print(depth.shape)

    depth_encoded = (depth - depth.min()) / (depth.max() - depth.min()) * (2**24 - 1)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    print(depth.shape, depth_encoded.shape)
    depth = depth.astype(np.uint8)
    print(depth.dtype, depth_encoded.dtype)

    def encodeValueToRGB(depth_encoded):
        r = (depth_encoded / 256 / 256) % 256
        g = (depth_encoded / 256) % 256
        b = depth_encoded % 256
        return np.stack([b, g, r], axis=-1).astype(np.uint8)

    depth_encoded = encodeValueToRGB(depth_encoded)
    # resize depth_encoded so that the smaller dimension has size args.input_size, while keeping the aspect ratio
    if h < w:
        new_h = args.input_size
        new_w = int(w * args.input_size / h)
    else:
        new_w = args.input_size
        new_h = int(h * args.input_size / w)
    depth_encoded = cv2.resize(depth_encoded, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(depth_encoded.shape)

    cv2.imwrite(outfile, depth_encoded)
    print("writing ", outfile)

def process_files(depth_anything):
    
    pipe_path = "/tmp/my_pipe"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)

    os.makedirs(os.path.join(args.path, "depth"), exist_ok=True)

    while True:
        with open(pipe_path, "r") as pipe:
            message = pipe.readline()[:-1]  # Read a line and remove the newline character
            print("Received:", message)
            for filename in os.listdir(args.path):
                kind = ""
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    kind = "image"
                if filename.endswith('.mp4'):
                    kind = "video"

                if kind == "":
                    continue
                if kind == "image" and os.path.exists(os.path.join(args.path, "depth", filename[:-4] + '.png')):
                    continue

                filename = os.path.join(args.path, filename)

                
                if kind == "image":
                    filename_sans_type = os.path.basename(filename).split('.')[0]
                    outfile = os.path.join(args.path, "depth", filename_sans_type + '.png')
                    estimate(depth_anything, filename, outfile)
                elif kind == "video":
                    pass
                    

def replacev1(depth_anything):
    dir = "../img3dviewer/images"
    for filename in os.listdir(dir):
        # if file is type json, skip
        if filename.endswith('.json') or filename.endswith('.mp4'):
            continue
        # skip directory
        if os.path.isdir(os.path.join(dir, filename)):
            continue

        filename_sans_type = filename.split('.')[0]
        outfile = os.path.join(args.path, "depth", filename_sans_type + '.png')
        estimate(depth_anything, os.path.join(dir, filename), outfile)
        # break
    exit()



if __name__ == '__main__':
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    # replacev1(depth_anything)
    process_files(depth_anything)
    # estimate(depth_anything, "../img3dviewer/images/00000-2114839625.png", "tmp.png")
