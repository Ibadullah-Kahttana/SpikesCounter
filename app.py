import csv

from flask import Flask, render_template, request
import pandas as pd  # for Loading dataset
import numpy as np  # Used for array
import cv2  # images , inputS
import os  # Os operations , counting etc
#import re
#from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
#from torch import device
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------------------

def runModel():

    DIR_INPUT = 'D:/Project/Spikes Counter/Dataset'
    # DIR_TRAIN = f'{DIR_INPUT}/train'
    DIR_TEST = f'{DIR_INPUT}/test'
    # DIR_ASSETS = 'C:/Project/Spikes Counter/assets'
    DIR_WEIGHTS = 'D:/Project/Spikes Counter/weights'
    WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn.pth'

    # ----------------------------------------------------------------------------------------

    test_df = pd.read_csv('D:/Project/Spikes Counter/assets/sample_submission.csv')
    test_df.shape

    # ----------------------------------------------------------------------------------------

    class WheatTestDataset(Dataset):

        def __init__(self, dataframe, image_dir, transforms=None):
            super().__init__()

            self.image_ids = dataframe['image_id'].unique()
            self.df = dataframe
            self.image_dir = image_dir
            self.transforms = transforms

        def __getitem__(self, index: int):
            image_id = self.image_ids[index]
            records = self.df[self.df['image_id'] == image_id]

            image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']

            return image, image_id

        def __len__(self) -> int:
            return self.image_ids.shape[0]

    # ----------------------------------------------------------------------------------------

    # Albumentations
    def get_test_transform():
        return A.Compose([
            # A.Resize(512, 512),
            ToTensorV2(p=1.0)
        ])

    # ----------------------------------------------------------------------------------------

    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # -----------------------------------------------------------------------------------------

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
    model.eval()

    x = model.to(device)


    # -----------------------------------------------------------------------------------------

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )

    # ---------------------------------------------------------------------------------------

    def format_prediction_string(boxes, scores):
        pred_strings = []
        for j in zip(scores, boxes):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

        return " ".join(pred_strings)

    # ------------------------------------------------------------------------------------

    detection_threshold = 0.5
    results = []
    for x in os.listdir('static\\Bboximg'):
        os.remove('static\\Bboximg\\' + x)
    name = 0

    for images, image_ids in test_data_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            # --------------------------------------------------------------------------------

            sample = images[name].permute(1, 2, 0).cpu().numpy()
            boxes = outputs[name]['boxes'].data.cpu().numpy()
            scores = outputs[name]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)

            # ---------------------------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ittr = 0

            for box in boxes:
                ittr += 1
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 0), 2)

            ax.set_axis_off()
            ax.imshow(sample)

            plt.savefig("static\\Bboximg\\box" + str(name) + ".png")
            name = name +1

            #-------------------------------------------------------------------------------

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores),
                'Spikes_Count': ittr
            }

            results.append(result)

    # ---------------------------------------------------------------------------------------------

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString', 'Spikes_Count'])
    print(test_df)

    # --------------------------------------------------------------------------------------------

    test_df.to_csv('D:/Project/Spikes Counter/outputCSV/Predicted_data.csv', index=False)

    # --------------------------------------------------------------------------------------------


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    headers = ['image_id', 'PredictionString']
    randomString = "1.0 0 0 50 50"
    UPLOAD_FOLDER = './static/data'
    imageNames = []
    if request.method == 'POST':

        # Delete all previous images from data folder
        dir = 'D:/Project/Spikes Counter/static/data'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        # Saving new images into data folder
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        files = request.files.getlist("file")
        for file in files:
            imageNames.append(file.filename.split(".")[0])
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

        # Making csv file
        f = open('D:/Project/Spikes Counter/assets/sample_submission.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        for name in imageNames:
            row = [name, randomString]
            writer.writerow(row)
        f.close()

        # Model code
        runModel()

        f2 = open('D:/Project/Spikes Counter/outputCSV/Predicted_data.csv', 'r', newline='')
        reader = csv.reader(f2)
        i = 1
        spikesCount = 0
        for row in reader:
            if i != 1:
                spikesCount += int(row[2])
            i += 1

        yieldValue =  (spikesCount * 2.17)
        normalizedyield = round(yieldValue,3)

        return render_template('results.html', spikes=spikesCount, yieldVal=normalizedyield)

    else:
        return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    imageNames = []
    if request.method == 'POST':
        f2 = open('D:/Project/Spikes Counter/outputCSV/Predicted_data.csv', 'r', newline='')
        reader = csv.reader(f2)
        i = 1
        spikesCount = 0
        for row in reader:
            if i != 1:
                spikesCount += int(row[2])
            i += 1

        yieldValue = (spikesCount * 2.17)
        normalizedyield = round(yieldValue,3)

        return render_template('results.html', spikes=spikesCount, yieldVal=normalizedyield)


@app.route('/boxView', methods=['POST', 'GET'])
def boxView():
    headers = ['image_id', 'PredictionString']
    randomString = "1.0 0 0 50 50"
    UPLOAD_FOLDER = './static/data'
    imageNames = []
    if request.method == 'POST':
        dir = 'D:/Project/Spikes Counter/static/data'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        # Saving new images into data folder
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        files = request.files.getlist("file")
        for file in files:
            imageNames.append(file.filename.split(".")[0])
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

        # Making csv file
        f = open('D:/Project/Spikes Counter/assets/sample_submission.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        for name in imageNames:
            row = [name, randomString]
            writer.writerow(row)
        f.close()

        # Model code
        runModel()
        images=[]
        for x in os.listdir('static\\Bboximg'):
            images.append('Bboximg/' + x)
        return render_template('showImage.html', images=images)

@app.route('/home', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


# pip install flask
# pip install pandas
# pip install opencv-python
# pip install Pillow
# pip install -U albumentations
# pip3 install torchvision
# pip install matplotlib
