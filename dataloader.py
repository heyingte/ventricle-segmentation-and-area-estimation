from torch.utils.data import Dataset
from helpers import center_crop, get_SAX_SERIES
import pydicom, cv2, re
import os, fnmatch, sys
import numpy as np
from torchvision import transforms

SAX_SERIES = get_SAX_SERIES()

def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        # match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path) #windows
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path) #linux
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__


def read_contour(contour, data_path):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(data_path, contour.case, filename)
    f = pydicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    return img, mask


def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = list(map(Contour, contours))
    
    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks

class SunnybrookDataset(Dataset):
    def __init__(self,img_path,contour_path,contour_type,crop_size):
        self.img_path = img_path
        self.contour_path = contour_path
        self.contour_type = contour_type
        self.crop_size = crop_size

        ctrs = map_all_contours(contour_path, contour_type, shuffle=True)
        self.img, self.mask = export_all_contours(ctrs,img_path,crop_size=crop_size)


    
    def __getitem__(self,index):
        img, mask = self.img[index], self.mask[index]

        img = np.transpose(img,(2,0,1))
        
        mask = np.transpose(mask,(2,0,1))
        mask = mask.squeeze(0)

        return img,mask

    def __len__(self):

        return len(self.img)








