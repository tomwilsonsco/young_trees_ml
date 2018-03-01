import numpy as np
import os
import abc
#import cPickle as pickle
import dill
import skimage.feature as ft
import skimage as sk
#from skimage.morphology import watershed, disk
#from skimage.filters.rank import entropy
from skimage.filters import rank
from skimage import morphology
from scipy import ndimage as ndi

WORK_DIR = "/home/ubuntu/Documents/Earth_Observation/Exploration/"

class FeatureFunction(object):
    def __init__(self):
        pass    
    params = {}
    name = ''        
    description = ''

    @abc.abstractmethod
    def transform(self, src):
        """
        takes an image of shape (2, height, width)
        returns an array of shape (2, n) 
        """
        raise NotImplementedError, "{} transform() needs to be implemented".format(name)
    
    def save(self, file_path):
        filename = self.name.strip() + '_'.join([it[0]+'='+str(it[1]) for it in self.params.items()]) + '.pkl'
        with open(os.path.join(file_path, filename), 'w') as fp:
            dill.dump(self, fp)


class LBP(FeatureFunction):
    def __init__(self, method, radius):
        self.params = {'method': method, 'radius': radius}
        self.description = 'Local Binary Patterns with parameters: method={}; radius={}'.format(self.params['method'],
                                                                                                 self.params['radius'])
    name = 'Local Binary Pattern'
    def transform(self, src):
        method = self.params['method']        
        radius = self.params['radius']
        n_points = int(radius * 8)
        n_bins = n_points + 2
        lbp0 = ft.local_binary_pattern(src[0], n_points, radius, method).reshape(-1)
        lbp0 = lbp0[~np.isnan(lbp0)]
        lbp1 = ft.local_binary_pattern(src[1], n_points, radius, method).reshape(-1)
        lbp1 = lbp1[~np.isnan(lbp1)]
        lbp0_hist = np.histogram(lbp0.reshape(-1), bins=n_bins, normed=True, range = (0, n_bins))[0]
        lbp1_hist = np.histogram(lbp1.reshape(-1), bins=n_bins, normed=True, range = (0, n_bins))[0]
        return [lbp0_hist, lbp1_hist]

class Distribution(FeatureFunction):
    def __init__(self, nbins=20):
        self.params = {'nbins': 20}
        self.description = 'Normalised Distribution of image pixels with {} bins'.format(self.params['nbins'])        
    name = 'Pixel Distribution'
    
    def transform(self, src):
        src0 = src[0].reshape(-1)
        src0 = src0[~np.isnan(src0)]
        src1 = src[1].reshape(-1)
        src1 = src1[~np.isnan(src1)]
        hist0 = np.histogram(src0.reshape(-1), bins=self.params['nbins'], normed=True, range=(-25, -5))[0]
        hist1 = np.histogram(src1.reshape(-1), bins=self.params['nbins'], normed=True, range=(-25, -5))[0]
        return [hist0, hist1]

class WatershedSegmentation(FeatureFunction):
    def __init__(self, disk_radius, max_range):
        self.params['disk_radius'] = disk_radius
        self.params['max_range'] = max_range
        self.name = 'WatershedSegmentation'
        self.description = 'Watershed Segmentation with radius {}'.format(disk_radius)

    def transform(self, src):
        denoised = np.array([rank.median(np.power(10, (src[0])/10.), morphology.disk(self.params['disk_radius'])),
                             rank.median(np.power(10, (src[1])/10.), morphology.disk(self.params['disk_radius']))])
        # find continuous region (low gradient -
        # where less than 10 for this image) --> markers
        # disk(5) is used here to get a more smooth image
        markers = np.array([rank.gradient(denoised[0], disk(5)) < 10, rank.gradient(denoised[1], sk.morphology.disk(5)) < 10])
        markers = ndi.label(markers)[0]
        # local gradient (disk(2) is used to keep edges thin)
        gradient = np.array([rank.gradient(denoised[0], morphology.disk(self.params['disk_radius'])), 
                             rank.gradient(denoised[1], morphology.disk(self.params['disk_radius']))])
        # process the watershed
        labels = np.array(morphology.watershed(gradient, markers), dtype='float')
        labels[np.isnan(src)] = np.NaN
        hist0 = np.histogram(labels[0].reshape(-1), bins=[1,2,3,4,5], normed=True, range=(0, self.params['max_range']))[0]
        hist1 = np.histogram(labels[1].reshape(-1), bins=[1,2,3,4,5], normed=True, range=(0, self.params['max_range']))[0]

        unique_labels = np.array(sorted(np.unique(labels)))
        unique_labels = unique_labels[~np.isnan(unique_labels)]        
        label_means = {label: np.array([0, 0]) for label in range(int(7))}
        label_counts = {label:0 for label in range(int(7))}
        label_idx = 0
        for label in unique_labels:
            label_means[label_idx] = np.nanmean(src[labels==label].reshape(2,-1), axis=1)
            label_counts[label_idx] = np.sum(~np.isnan(src[labels==label]))
            label_idx += 1
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])[::-1]
        label_vals0 = np.array([label_means[label[0]][0] for label in sorted_labels])
        label_vals1 = np.array([label_means[label[0]][1] for label in sorted_labels])
        return [label_vals0, label_vals1]
        #return [np.hstack([label_vals0, hist0]), np.hstack([label_vals1, hist1])]
        #return [hist0, hist1]

class Entropy(FeatureFunction):
    def __init__(self, disk_radius, nbins):
        self.params['disk_radius'] = disk_radius
        self.params['nbins'] = nbins
        self.name = 'Entropy' 
        self.description = 'Local Entropy with disk radius {} and nbins {}'.format(self.params['disk_radius'], nbins)

    def transform(self, src):
        ent = np.array([rank.entropy(np.power(10, (src[0])/10.), morphology.disk(self.params['disk_radius'])),
                        rank.entropy(np.power(10, (src[1])/10.), morphology.disk(self.params['disk_radius']))])
        hist0 = np.histogram(ent[0].reshape(-1), bins=[1,2,3,4,5], normed=True, range=(0, 5))[0]
        hist1 = np.histogram(ent[1].reshape(-1), bins=[1,2,3,4,5], normed=True, range=(0, 5))[0]
        return [hist0, hist1]

if __name__=="__main__":
    lbp_params = {'method': 'uniform', 'radius': 2}
    lbp = LBP(**lbp_params)
    lbp.save(os.path.join(WORK_DIR ,'features'))

    watershed_params = {'disk_radius': 2, 'max_range': 7}
    watershed = WatershedSegmentation(**watershed_params)
    watershed.save(os.path.join(WORK_DIR ,'features/'))

    dist = Distribution(nbins=20)
    dist.save(os.path.join(WORK_DIR ,'features/'))

    ent = Entropy(nbins=20, disk_radius=2)
    ent.save(os.path.join(WORK_DIR ,'features/'))
    