from drawContours import cellCount_DrawContours
from get_data_ids import get_ids_in_list
from tqdm import tqdm

data_path = "../test/images/"
prediction_path = "../test/gr_predictionsinter/"
images = get_ids_in_list(data_path)

'''img_id='0076_.png'
origImgFilename = prediction_path  + img_id
procImgFilename = prediction_path + img_id[0:-4] + "masksubs.png"
savedImgFilename= prediction_path +  img_id[0:-4]+ "segmented.png"
cellCount, ratio = cellCount_DrawContours(origImgFilename, procImgFilename, savedImgFilename)
print(cellCount)'''

for img_id in tqdm(images, total=len(images)):
	#print(img_id[0:-4])
	origImgFilename = prediction_path  + img_id
	procImgFilename = prediction_path + img_id[0:-4] + "masksubs.png"
	savedImgFilename= prediction_path +  img_id[0:-4]+ "segmented.png"
	cellCount, ratio = cellCount_DrawContours(origImgFilename, procImgFilename, savedImgFilename)