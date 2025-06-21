import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import csv
import sys
from mask_circle import MaskCircleFinder
from mask_circle import MaskCircleFinder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------- #
#                     Load the Model                #
# ------------------------------------------------- #

def load_model(device):
    # Load weights of the model
    print("Loading the model .......")
    weights = torch.load('vit.pth', map_location=device)

    # ViT 
    model = models.vit_b_16(pretrained=True)
    num_ftrs = model.heads[-1].in_features
    model.heads[-1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(weights['state_dict'])
    
    return model

# ------------------------------------------------- #
#                 Image Transformation              #
# ------------------------------------------------- #

def get_transform():
    img_size = 224
    transform = transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
    return transform

# -------------------------------------------------- #
#                 Iris Extraction Function           #
# -------------------------------------------------- #

def extract_irisxyr(img, device):
    # initialize the SIF loss model
    mask_circle_finder = MaskCircleFinder(device=device)
    # find the pupil and iris mask
    _, _, tar_iris_xyr = mask_circle_finder.segment_and_circApprox(img)
    return tar_iris_xyr 

# ------------------------------------------------- #
#      Image Processing and Score Calculation       #
# ------------------------------------------------- #

def process_image(csvPath, transform, device):

    imagesScores=[]
    
    print("Processing images from", csvPath)
    with open(csvPath, 'r') as f:
        csvFile = csv.reader(f)
        for row in csvFile:
            idx = row[0]
            imgPath = row[1]
            #print(f"Processing image {idx}: {imgPath}")
            try:
                image = Image.open(imgPath)
                iris_xyr = extract_irisxyr(image, device=device)
                # crop the image to the iris region
                iris_x = int(iris_xyr[0])
                iris_y = int(iris_xyr[1])
                iris_r = int(iris_xyr[2])
                image = image.crop((iris_x - iris_r - 16, iris_y - iris_r - 16, iris_x + iris_r + 16, iris_y + iris_r + 16))
                
                # Resize the image to the required size
                tranformImage = transform(image)
                image.close()
                tranformImage = tranformImage.repeat(3, 1, 1) 
                tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
                tranformImage = tranformImage.to(device)

                # Get model prediction score for the image, move it to cpu and convert it to numpy
                output = model(tranformImage)
                PAScore = torch.sigmoid(output).detach().cpu().numpy()[:, 1]
                imagesScores.append([idx,imgPath,PAScore[0]])
            except Exception as e:
                print(f"Can't process the image {imgPath}: {e}")
                imagesScores.append([idx,imgPath,None])
    return imagesScores

# ------------------------------------------------- #
#                     Main Function                 #
# ------------------------------------------------- #

if __name__ == '__main__':
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(device=device)
    model = model.to(device)
    model.eval() 
    
    
    # Calculate the score for each image in the input CSV file
    imagesScores = process_image(csvPath=input_csv, transform=get_transform(), device=device)

    # Save the results to the output CSV file
    header = ['index','filename','score']
    with open(output_csv,'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(imagesScores)
        
    print(f"Results saved to {output_csv}")
  
        
