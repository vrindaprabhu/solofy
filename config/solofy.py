# GENERIC
GPU_DEVICE = "cpu"
INIMAGE = "./input/inimage.png"
DETIMAGE = "./tmp/detection.png"
MASKIMAGE = "./tmp/mask.png"
SALIENTIMAGE = "./tmp/salient.png"
BGIMAGE = "./tmp/onlybg.png"
OUTIMAGE = "./output/final.png"
RESIZE_TO = (512, 512)
CUDA = False
LOGFILENAME = "solofy.log"

# DEEPFILLv2
DEEPFILL_MODEL_PATH = "./models/deepfillv2_WGAN.pth"
GPU_ID = -1
INIT_TYPE = "xavier"
INIT_GAIN = 0.02
PAD_TYPE = "zero"
IN_CHANNELS = 4
OUT_CHANNELS = 3
LATENT_CHANNELS = 48
ACTIVATION = "elu"
NORM = "in"
RESIZE_SHAPE = 512
NUM_WORKERS = 0

# U2NET
UNET_MODEL_PATH = "./models/u2netp.pth"

# YOLOv3
YOLO_MODEL_PATH = "./models/yolov3.weights"
OBJ_THRESH = 0.5
NMS_THRESH = 0.4

# MASKS
EXTRA_PIXEL = 30
EXTRA_FG_PIXEL = 10
