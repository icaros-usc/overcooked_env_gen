import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

# GAN paths
# data path
GAN_DIR = os.path.join(_current_dir, "GAN_training")
GAN_DATA_DIR = os.path.join(GAN_DIR, "data")
GAN_TRAINING_DIR = os.path.join(GAN_DATA_DIR, "training")
GAN_LOSS_DIR = os.path.join(GAN_DATA_DIR, "loss")

if not os.path.exists(GAN_LOSS_DIR):
    os.mkdir(GAN_LOSS_DIR)

# plot pic path
ERR_LOG_PIC = os.path.join(GAN_LOSS_DIR, "err.png")

# G_param file path
G_PARAM_FILE = os.path.join(GAN_DATA_DIR, "G_param.json")

# LSI paths
# data path
LSI_DIR = os.path.join(_current_dir, "LSI")
LSI_DATA_DIR = os.path.join(LSI_DIR, "data")
LSI_LOG_DIR = os.path.join(LSI_DATA_DIR, "log")
LSI_IMAGE_DIR = os.path.join(LSI_DATA_DIR, "images")

# config LSI search
LSI_CONFIG_DIR = os.path.join(LSI_DATA_DIR, "config")
LSI_CONFIG_EXP_DIR = os.path.join(LSI_CONFIG_DIR, "experiment")
LSI_CONFIG_MAP_DIR = os.path.join(LSI_CONFIG_DIR, "elite_map")
LSI_CONFIG_ALGO_DIR = os.path.join(LSI_CONFIG_DIR, "algorithms")
LSI_CONFIG_AGENT_DIR = os.path.join(LSI_CONFIG_DIR, "agents")

# human study
LSI_HUMAN_STUDY_DIR = os.path.join(LSI_DATA_DIR, "human_study")
LSI_HUMAN_STUDY_CONFIG_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "config")
LSI_HUMAN_STUDY_LOG_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "result")
LSI_HUMAN_STUDY_AGENT_DIR = os.path.join(LSI_HUMAN_STUDY_CONFIG_DIR, "agents")

LSI_HUMAN_STUDY_RESULT_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "result")
