import platform
import os.path

system = platform.system()

if system == 'Windows':
    BASE_DIRECTORY = 'D:\\deeplearning'
    DATASET_BASE_DIRECTORY = os.path.join(BASE_DIRECTORY, 'dataset')
elif system == 'Darwin':
    BASE_DIRECTORY = '/tmp'
    DATASET_BASE_DIRECTORY = '/Users/eric/Documents/GitHub/dataset'
else:  # Linux
    BASE_DIRECTORY = '/tmp'

SUMMARIES_DIRECTORY = '/tmp/retrain_logs'
MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, 'model/prestige')
BOTTLENECK_DIRECTORY = os.path.join(BASE_DIRECTORY, 'bottleneck')

DATASET_FLICKR_DIRECTORY = os.path.join(DATASET_BASE_DIRECTORY, 'FLICKR')
DATASET_CUHKPQ_DIRECTORY = os.path.join(DATASET_BASE_DIRECTORY, 'CUHKPQ')
DATASET_AVA_DIRECTORY = os.path.join(DATASET_BASE_DIRECTORY, 'AVA')
