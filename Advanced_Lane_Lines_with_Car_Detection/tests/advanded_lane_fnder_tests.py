import unittest
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../classes/'))
from Thresholding import Thresholding
from Transformation import Transformation
from LaneLines import LaneLines
from Lines import Lines
#from Advanced_Lane_Line import Advanced_Lane_Line

class Test_advanded_lane_fnder_tests(unittest.TestCase):
    output_folder = "output_images/{}/{}/{}"
    mode = "tests"

    def get_output_file(self, prefix, filename):
        
        file = filename.split("\\")[-1]
        output_filename = "{}_{}".format(prefix, file)
        output_folder = self.get_output_folder()
        output_file = "{}/{}".format (output_folder, output_filename)
        
        #make sure the output floder are present
        if not (os.path.exists(output_folder)):
            os.makedirs(output_folder)
        
        return output_file

    def get_output_folder(self):
        return "output_images/{}/{}/{}".format(self.mode, self.obj, self.action)
       
    def test_birdseye_transformation(self): 
        file_missing = None
        s = "birdseye"
        n = "transformation"
        test_output_folder = self.get_test_folder(n, s)
        if not os.path.exists(test_output_folder):
               os.mkdir(test_output_folder)
        image = plt.imread("test_images/straight_lines1.jpg", 1)
        processor = Transformation(True, True, image)
       
        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
            image = plt.imread(fname, 1)
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            new_image = processor.transform_perspective(image)

            f = fname.split('\\')[-1]
            file = "{}{}{}".format(test_output_folder, "{}_".format(s), f)
        
            if  os.path.exists(file):
                if os.path.isfile(file):
                    os.remove(file)
               
            plt.imsave(file, new_image, cmap = "gray")
                 

        images = glob.glob(test_output_folder)
        for idx, fname in enumerate(images):
        
            file = "{}{}{}".format(test_output_folder, "{}_".format(s), fname)
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True


        if file_missing:
            self.fail("Images not created")
        else:
            pass

    def test_birdseye_inverse_transformation(self): 
        file_missing = None
        s = "birdseye_inverse"
        n = "transformation"
        test_output_folder = self.get_test_folder(n, s)
        if not os.path.exists(test_output_folder):
               os.mkdir(test_output_folder)
        
        image = plt.imread("test_images/straight_lines1.jpg", 1)
        processor = Transformation(True, True, image)
       
        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
            image = plt.imread(fname, 1)
            
            new_image = processor.get_birds_eye_view(image)
            new_image = processor.get_birds_eye_inverse_view(image)

            f = fname.split('\\')[-1]
            file = "{}{}{}".format(test_output_folder, "{}_".format(s), f)
        
            if  os.path.exists(file):
                if os.path.isfile(file):
                    os.remove(file)
               
            plt.imsave(file, new_image, cmap = "gray")
                 

        images = glob.glob(test_output_folder)
        for idx, fname in enumerate(images):
        
            file = "{}{}{}".format(test_output_folder, "{}_".format(s), fname)
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True


        if file_missing:
            self.fail("Images not created")
        else:
            pass

    def test_camera_calibration(self): 
        file_missing = True
        self.obj = "camera"
        self.action = "undistort"
        
        image = plt.imread("test_images/straight_lines1.jpg", 1)
        processor = Transformation(True, True, image)
        
        images = glob.glob('camera_cal/calibration*.jpg')

        for idx, fname in enumerate(images):
            image = plt.imread(fname)
            undist_image = processor.undistort_image(image)             
            file_name = self.get_output_file("undistorted", fname)
            plt.imsave(file_name, undist_image, cmap = "gray")

        
        
        images = glob.glob('test_images/test*.jpg')

        for idx, fname in enumerate(images):
            image = plt.imread(fname)
            undist_image = processor.undistort_image(image)             
            file_name = self.get_output_file("undistorted", fname)
            plt.imsave(file_name, undist_image, cmap = "gray")

        images = glob.glob('camera_cal/calibration*.jpg')
        for idx, fname in enumerate(images):
        
            file = self.get_output_file("undistorted", fname)
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True

        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
        
            file = self.get_output_file("undistorted", fname)
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True

        if file_missing:
            self.fail("Images not created")
        else:
            pass

    def test_image_thresholding(self): 
        file_missing = True
        self.obj = "threshold"
        self.action = "binary"
        
        images = glob.glob('test_images/test*.jpg')

        for idx, fname in enumerate(images):
            image = plt.imread(fname)
            edge_pos, edge_neg, yellow_edge_pos,  yellow_edge_neg,  white_tight,  white_loose, yellow = Thresholding.split_channels(image, True)
            file_name = self.get_output_file("edge_pos", fname)
            plt.imsave(file_name, edge_pos, cmap = "gray")
            #self.plot_image(file_name, edge_pos)

            file_name = self.get_output_file("edge_neg", fname)
            plt.imsave(file_name, edge_neg, cmap = "gray")
            #self.plot_image(file_name, edge_pos)

            file_name = self.get_output_file("yellow_edge_pos", fname)
            plt.imsave(file_name, yellow_edge_pos, cmap = "gray")
            #self.plot_image(file_name, edge_pos)
            
            file_name = self.get_output_file("yellow_edge_neg", fname)
            plt.imsave(file_name, yellow_edge_neg, cmap = "gray")
            #self.plot_image(file_name, edge_pos)

            file_name = self.get_output_file("white_tight", fname)
            plt.imsave(file_name, white_tight, cmap = "gray")
            #self.plot_image(file_name, edge_pos)

            file_name = self.get_output_file("white_loose", fname)
            plt.imsave(file_name, white_loose, cmap = "gray")
            #self.plot_image(file_name, edge_pos)
            
            file_name = self.get_output_file("yellow", fname)
            plt.imsave(file_name, yellow, cmap = "gray")
            #self.plot_image(file_name, edge_pos)

        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
        
            file = self.get_output_file("yellow", fname)
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True

        if file_missing:
            self.fail("Images not created")
        else:
            pass

    def test_lane_line_finder(self):
        line = {'left':Lines(n_iter=7), 'right':Lines(n_iter=7)}
        fail = {'left':2, 'right':2}

        file_missing = True
        self.obj = "lane_lines"
        self.action = "detect"

        image = plt.imread("test_images/straight_lines1.jpg", 1)
        processor = Transformation(True, True, image)
        
        lane_lines = LaneLines(processor)
        images = glob.glob('test_images/test*.jpg')
        #images = glob.glob('output_images/debug/standard/original_output*.jpg')
        for idx, fname in enumerate(images):
            image = plt.imread(fname)
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            line = {'left':Lines(n_iter=7), 'right':Lines(n_iter=7)}
            fail = {'left':2, 'right':2}
            final, line, fail = lane_lines.get_image_with_lanes(image, line, fail, 15, True, True )
            file_name = self.get_output_file("lane_line_test", fname)
            plt.imsave(file_name, final, cmap = "gray")
        
        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
            file = self.get_output_file("lane_line_test", fname)
            
            if  os.path.exists(file) or idx == 0:
                file_missing = False
            else:
                file_missing = True
    
    
        if file_missing:
            self.fail("Images not created")
        else:
            pass

    def plot_image(self, file_name, image) :
        plt.figure(figsize=(10,10)) #
        plt.imshow(image)
        plt.savefig(file_name)

if __name__ == '__main__':
    unittest.main()
