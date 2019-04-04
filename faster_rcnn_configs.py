

# SYSTEM CONFIGS
MODE = 0    # gpu: 0    cpu: 1
GPU_ID = 0

# DATA CONFIGS
IMAGE_BATCH_SIZE = 1
IMAGE_SHAPE = None
DATASET_PATH = None
ANNO_PATH = None
LABEL_PATH = None
NUM_CLS = 80     # Exclude background
CLS_NAMES = ['BG']
# MEAN_COLOR = [103.94383649, 113.94377407, 119.86271446]     # BGR
MEAN_COLOR = [119.86271446, 113.94377407, 103.94383649]        # RGB

# TRAIN CONFIGS
BACKBONE = 'resnet50'
FEATURE_STRIDE = 32

L2_WEIGHT = 0.0005

MODEL_NAME = BACKBONE + '_Faster_RCNN'
PRE_TRAIN_MODEL_PATH = './logs/pretrain_model/pretrain_resnet50.ckpt'
SAVE_MODEL_ITER = 100
SAVE_MODEL_MAXIMUM_ITERS = 10000
MAXIMUM_ITERS = 200000

SUMMARY_PATH = './logs'
REFRESH_LOGS_ITERS = 10

ADD_GT_BOX_TO_TRAIN = True

LEARNING_RATE_BOUNDARIES = [100, 1000, 10000, 20000]
LEARNING_RATE_SCHEDULAR = [0.1, 0.01, 0.001, 0.0005, 0.0001]

# TEST CONFIGS
TEST_SCORE_THRESHOLD = 0.7

# ANCHOR CONFIGS
ANCHOR_BASE_SIZE = 16
ANCHOR_SCALE = [4, 8, 16, 32]
ANCHOR_RATE = [0.5, 1.0, 2.0]
ANCHOR_NUM = len(ANCHOR_SCALE) * len(ANCHOR_RATE)
ROI_SCALE_FACTORS = [5., 5., 10., 10.]

# RPN_CONFIGS
RPN_KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
RPN_FOREGROUND_FRACTION = 0.5

RPN_TOP_K_NMS_TRAIN = 12000
RPN_PROPOSAL_MAX_TRAIN = 2000
RPN_PROPOSAL_MAX_TEST = 300

RPN_MINIBATCH_SIZE = 256
RPN_NMS_IOU_THRESHOLD = 0.7

RPN_BN_DECACY = 0.995
RPN_BN_EPS = 0.0001
RPN_WEIGHTS_L2_PENALITY_FACTOR = 0.0005

RPN_CLASSIFICATION_LOSS_WEIGHTS = 1.0
RPN_LOCATION_LOSS_WEIGHTS = 1.0


# FASTER_RCNN_CONFIGS
FASTER_RCNN_ROI_SIZE = 14
FASTER_RCNN_POOL_KERNEL_SIZE = 2

FASTER_RCNN_NMS_IOU_THRESHOLD = 0.2
FASTER_RCNN_NMS_MAX_BOX_PER_CLASS = 100

FASTER_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FASTER_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0
FASTER_RCNN_MINIBATCH_SIZE = 128
FASTER_RCNN_POSITIVE_RATE = 0.75

FASTER_RCNN_CLASSIFICATION_LOSS_WEIGHTS = 1.0
FASTER_RCNN_LOCATION_LOSS_WEIGHTS = 1.0
