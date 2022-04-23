import os
import h5py
import numpy as np
np.random.seed(309)

loadPath = "../data/LA/2018LA_Seg_Training Set/"
savePath = "../data/LA/processed_h5/"

def getRangImageDepth(image):
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[2]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

if __name__ == "__main__":
    with open(loadPath + '../train.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    for i in range(len(image_list)):
        image_name = image_list[i]
        h5f = h5py.File(loadPath + "/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        startpostion, endpostion = getRangImageDepth(label)
        rdm_start = startpostion + (endpostion - startpostion) / 5
        rdm_end = endpostion - (endpostion - startpostion) / 5
        lbl_idx = np.random.randint(rdm_start, rdm_end, size=1)
        print(rdm_start, rdm_end, lbl_idx)

        new_lbl = np.zeros(label.shape)
        if i < 16:
            new_lbl[..., lbl_idx] = label[..., lbl_idx]

        # save
        if not os.path.exists(savePath + "/" + image_name):
            os.makedirs(savePath + "/" + image_name)
        save_file = h5py.File(savePath + "/" + image_name + "/mri_norm2.h5", 'w')
        save_file.create_dataset('image', data=image)
        save_file.create_dataset('label_full', data=label)
        save_file.create_dataset('label', data=new_lbl)
        save_file.close()

