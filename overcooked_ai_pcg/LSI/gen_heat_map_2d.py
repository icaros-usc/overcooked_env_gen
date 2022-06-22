import matplotlib

matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2
import csv
import glob
import toml
import argparse
import seaborn as sns
import pandas as pd
from itertools import product, combinations
from overcooked_ai_pcg import LSI_IMAGE_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR

# handled by command line argument parser
FEATURE1_LABEL = None  # label of the first feature to plot
FEATURE2_LABEL = None  # label of the second feature to plot
IMAGE_TITLE = None  # title of the image, aka name of the algorithm
STEP_SIZE = None  # step size of the animation to generate
LOG_FILE_NAME = None  # filepath to the elite map log file
NUM_FEATURES = None  # total number of features (bc) used in the experiment
ROW_INDEX = None  # index of feature 1 to plot
COL_INDEX = None  # index of feature 2 to plot
ELITE_MAP_NAME = None  # type of feature map used

# max and min value of fitness
FITNESS_MIN = 0
FITNESS_MAX = 400000


def read_in_lsi_config(exp_config_file):
    experiment_config = toml.load(exp_config_file)
    algorithm_config = toml.load(
        os.path.join(
            LSI_CONFIG_ALGO_DIR,
            experiment_config["experiment_config"]["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(
            LSI_CONFIG_MAP_DIR,
            experiment_config["experiment_config"]["elite_map_config"]))
    # agent_config = toml.load(
    # os.path.join(LSI_CONFIG_AGENT_DIR, experiment_config["experiment_config"]["agent_config"][1]))
    return experiment_config, algorithm_config, elite_map_config, None


def createRecordList(mapData, mapDims):
    recordList = []
    trackRmIndexPairs = {}

    # create custom indexPairs if needed:
    if ELITE_MAP_NAME == "workloads_diff.tml":
        lows = np.array([-6, -2, -2])
        highs = np.array([6, 2, 2]) + 1
        indexPairs = [
            (x, y)
            for x, y in product(range(lows[ROW_INDEX], highs[ROW_INDEX]),
                                range(lows[COL_INDEX], highs[COL_INDEX]))
        ]
    else:
        indexPairs = [(x, y) for x, y in product(range(mapDims[ROW_INDEX]),
                                                 range(mapDims[COL_INDEX]))]
    for i, cellData in enumerate(mapData):
        # splite data from csv file
        splitedData = cellData.split(":")
        # re-range the bc if needed:
        indexes = np.array([int(splitedData[i]) for i in range(NUM_FEATURES)])
        if ELITE_MAP_NAME == "workloads_diff.tml":
            indexes[0] -= 6
            indexes[1:] -= 2

        cellRow = indexes[ROW_INDEX]
        cellCol = indexes[COL_INDEX]
        nonFeatureIdx = NUM_FEATURES
        indID = int(splitedData[nonFeatureIdx])
        fitness = float(splitedData[nonFeatureIdx + 1])

        # re-range the fitness
        # if fitness == 1:
        #     fitness = 0
        # elif fitness >= 200000 and fitness < 400000:
        #     fitness -= 200000
        # elif fitness >= 400000:
        #     fitness -= 390000

        f1 = float(splitedData[nonFeatureIdx + 2 + ROW_INDEX])
        f2 = float(splitedData[nonFeatureIdx + 2 + COL_INDEX])

        data = [cellRow, cellCol, indID, fitness, f1, f2]
        if (cellRow, cellCol) in indexPairs:
            indexPairs.remove((cellRow, cellCol))
            trackRmIndexPairs[str(cellRow) + '_' +
                              str(cellCol)] = len(recordList)
            recordList.append(data)
        else:
            track_idx = trackRmIndexPairs[str(cellRow) + '_' + str(cellCol)]
            if data[3] > recordList[track_idx][3]:
                recordList[track_idx] = data

    # Put in the blank cells
    for x, y in indexPairs:
        recordList.append([x, y, 0, np.nan, 0, 0])

    return recordList


def createRecordMap(dataLabels, recordList):
    dataDict = {label: [] for label in dataLabels}
    for recordDatum in recordList:
        for i in range(len(dataLabels)):
            dataDict[dataLabels[i]].append(recordDatum[i])
    return dataDict


def createImage(rowData, filename):
    mapDims = tuple(map(int, rowData[0].split('x')))
    mapData = rowData[1:]

    dataLabels = [
        'CellRow',
        'CellCol',
        # 'CellSize',
        'IndividualId',
        # 'WinCount',
        'Fitness',
        'Feature1',
        'Feature2',
    ]

    recordList = createRecordList(mapData, mapDims)
    dataDict = createRecordMap(dataLabels, recordList)

    recordFrame = pd.DataFrame(dataDict)

    # Write the map for the cell fitness
    fitnessMap = recordFrame.pivot(index=dataLabels[1],
                                   columns=dataLabels[0],
                                   values='Fitness')
    fitnessMap.sort_index(level=1, ascending=False, inplace=True)
    sns.color_palette("flare", as_cmap=True)
    with sns.axes_style("white"):
        numTicks = 5  #11
        numTicksX = mapDims[ROW_INDEX] // numTicks + 1
        numTicksY = mapDims[COL_INDEX] // numTicks + 1
        plt.figure(figsize=(3, 3))
        g = sns.heatmap(
            fitnessMap,
            annot=False,
            cmap="flare_r",
            fmt=".0f",
            xticklabels=numTicksX,
            yticklabels=numTicksY,
            vmin=FITNESS_MIN,
            vmax=FITNESS_MAX,
        )
        fig = g.get_figure()
        # plt.axis('off')
        g.set(title=IMAGE_TITLE, xlabel=FEATURE1_LABEL, ylabel=FEATURE2_LABEL)
        plt.tight_layout()
        fig.savefig(filename)
    plt.close('all')


def createImages(stepSize, rows, filenameTemplate):
    for endInterval in range(stepSize, len(rows), stepSize):
        print('Generating : {}'.format(endInterval))
        filename = filenameTemplate.format(endInterval)
        createImage(rows[endInterval], filename)


def createMovie(folderPath, filename):
    globStr = os.path.join(folderPath, '*.png')
    imageFiles = sorted(glob.glob(globStr))

    # Grab the dimensions of the image
    img = cv2.imread(imageFiles[0])
    imageDims = img.shape[:2][::-1]

    # Create a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameRate = 30
    video = cv2.VideoWriter(os.path.join(folderPath, filename), fourcc,
                            frameRate, imageDims)

    for imgFilename in imageFiles:
        img = cv2.imread(imgFilename)
        video.write(img)

    video.release()


def generateAll(logPath):
    with open(logPath, 'r') as csvfile:
        # Read all the data from the csv file
        allRows = list(csv.reader(csvfile, delimiter=','))

        # generate the movie
        template = os.path.join(tmpImageFolder, 'grid_{:05d}.png')
        createImages(STEP_SIZE, allRows[1:], template)
        movieFilename = 'fitness_' + str(ROW_INDEX) + '_' + str(
            COL_INDEX) + '.avi'
        createMovie(tmpImageFolder, movieFilename)

        # Create the final image we need
        imageFilename = 'fitnessMap_' + str(ROW_INDEX) + '_' + str(
            COL_INDEX) + '.png'
        createImage(allRows[-1], os.path.join(tmpImageFolder, imageFilename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_file',
                        help='path to the experiment log file',
                        required=True)
    parser.add_argument('-f1',
                        '--feature1_idx',
                        help='index of the first feature to plot',
                        required=False,
                        default=0)
    parser.add_argument('-f2',
                        '--feature2_idx',
                        help='index of the second feature to plot',
                        required=False,
                        default=1)
    parser.add_argument('-s',
                        '--step_size',
                        help='step size of the animation to generate',
                        required=False,
                        default=100)
    # parser.add_argument('-l',
    #                     '--log_file',
    #                     help='filepath to the elite map log file',
    #                     required=False,
    #                     default=os.path.join((parser.parse_args().log_file), "elite_map.csv"))
    opt = parser.parse_args()

    # read in the name of the algorithm and features to plot
    experiment_config, algorithm_config, elite_map_config, _ = read_in_lsi_config(
        os.path.join(opt.log_file, "config.tml"))
    features = elite_map_config['Map']['Features']
    ELITE_MAP_NAME = experiment_config["experiment_config"]["elite_map_config"]
    print(ELITE_MAP_NAME)

    # read in parameters
    NUM_FEATURES = len(features)
    LOG_FILE_NAME = os.path.join(opt.log_file, "elite_map.csv")
    # Clear out the previous images
    tmpImageFolder = os.path.join(opt.log_file, "images/")
    if not os.path.exists(tmpImageFolder):
        os.mkdir(tmpImageFolder)
    # for curFile in glob.glob(tmpImageFolder + '/*'):
    #     os.remove(curFile)

    for ROW_INDEX, COL_INDEX in combinations(range(NUM_FEATURES), 2):
        STEP_SIZE = int(opt.step_size)
        IMAGE_TITLE = algorithm_config['name']
        FEATURE1_LABEL = features[ROW_INDEX]['name']
        FEATURE2_LABEL = features[COL_INDEX]['name']

        generateAll(LOG_FILE_NAME)
