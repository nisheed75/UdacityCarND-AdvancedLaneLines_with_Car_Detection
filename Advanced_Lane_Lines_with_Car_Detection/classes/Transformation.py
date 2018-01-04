import numpy as np
import cv2
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../classes/'))
from Thresholding import Thresholding
from LaneLines import LaneLines


class Transformation(object):
    """This class is used to get binary images using various thresholding methods"""
    
    def __init__(self, cal, plot, image):
        self.calibrate_camera(9, 6, cal)
        #image = plt.imread(image)
        self.get_transformation_matrix(image, plot=plot)

    def transform(self, image, matrix):
        """
        the wrapper method for perspective warping 
        :param image: 
        :param matrix: 
        :return: 

        """

        return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def get_birds_eye_view(self, image, leftShift=0):
        """
        perspective to bird-eye projection
        :param image: 
        :param leftShift: 
        :return: 

        """
        if leftShift == 0:
            return self.transform(image, self.M)
        else:
            src = self.src.copy()
            src[:, 0] = src[:, 0] + leftShift
            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + leftShift
            m_inv = cv2.getPerspectiveTransform(src, dst)
            return self.transform(image, m_inv)

    def get_birds_eye_inverse_view(self, image, leftShift=0):

        """
        the inverse of bird-eye.
        
        :param img: 
        :param leftShift: 
        :return: 

        """
        if leftShift == 0:
            return self.transform(image, self.M_inv)
        else:
            src = self.src.copy()
            src[:, 0] = src[:, 0] + leftShift
            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + leftShift
            m_inv = cv2.getPerspectiveTransform(dst, src)
            return self.transform(image, m_inv)

    def save_to_file(self, mtx, dist, cal_file_name):
         dist_pickle = {}
         dist_pickle["mtx"] = mtx
         dist_pickle["dist"] = dist
         pickle.dump( dist_pickle, open( cal_file_name, "wb" ) )
    
    def calibrate_camera (self, nx=9, ny=6, re_cal = False):
        """
        Returns the camera matrix, distortion coefficients only. The rotation and translation vectors 
        are calculated but not returned. The first time this is run it it will calcutate the values 
        and save it to a file. The next time it is run with re_cal = false it will look for the file
        and return the values from the previous calibation.
        
        To re_calibrate even if a file exists use re_cal = True
        """
        #The file where the calibration data us persisted       
        cal_file_name = "data/wide_dist_pickle.p"
        
        if (os.path.exists(cal_file_name) and re_cal == False):
            # file exists
            with open(cal_file_name, mode='rb') as f:
                calibration_data = pickle.load(f)
        
                self.mtx, self.dist = calibration_data["mtx"], calibration_data["dist"]
        
        else:
            # Criteria for termination of the iterative process of corner refinement.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.1)
        
            #Prepare known cordinates for a chess board with 9x6 object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((ny*nx,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d points in real world space
            imgpoints = [] # 2d points in image plane.
        
            # Make a list of calibration images
            images = glob.glob('camera_cal/*.jpg')
        
            # Step through the list and search for chessboard corners
            for idx, fname in enumerate(images):
                img = mpimg.imread(fname, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
                # If found, add object points, image points
                if (ret == True):
                    objpoints.append(objp)
                    #Once we find the corners, we can increase their accuracy using this code below.
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
            img_size = (img.shape[1], img.shape[0])
            # Do camera calibration given object points and image points
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
            # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
            self.save_to_file(self.mtx, self.dist, cal_file_name)
        
    def undistort_image(self, img): 
        """
        Use this method to to undistored images 
        """
        #img_size = (img.shape[1], img.shape[0])
        
        # undistort the image
        
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_transformation_matrix(self, image, plot=False):
        """ Given an image calculate the tranformation matrix and inverse 
        
        Parameters:
            image -- image with lane lines that are preferably straight
        """
        h,w = image.shape[:2]  
        self.src = np.float32([[220,700],[595,450],[685,450],[1060,700]])
        self.dst = np.float32([[w/4,h],[w/4,-100],[3*w/4,-100],[3*w/4,h]]) 
        self.compute_transformation_matrix(image, plot)

    def compute_transformation_matrix(self, image, plot=False):
        """
        
        Parameters: 
            image: 
        
        """
        h,w = image.shape[:2]
                
        # Compute the perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        
        # Compute the inverse perspective transformation matrix
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = self.get_birds_eye_view(image)
        if(plot):
            pts = np.int32(self.src)
            pts = pts.reshape((-1,1,2))
            annotated_img = np.copy(image)
            annotated_img = cv2.polylines(annotated_img,[pts],True,(255,0,0), thickness=5)
            warped = cv2.line(warped,(int(w/4),h),(int(w/4),-100),(255,0,0), thickness=5)
            warped = cv2.line(warped,(int(3*w/4),h),(int(3*w/4),-100),(255,0,0), thickness=5)

            s = "birdseye"
            test_output_folder = "output_images/tests/transformation/{}/get_trans/".format(s)  
            file = "{}{}{}".format(test_output_folder, "{}_{}_".format("debug_trans", "undistored"), "original_image.jpg")
            plt.imsave(file, annotated_img, cmap = "gray")

            s = "birdseye"
            test_output_folder = "output_images/tests/transformation/{}/get_trans/".format(s)  
            file = "{}{}{}".format(test_output_folder, "{}_{}_".format("debug_trans", "undistored"), "warp_image.jpg")
            plt.imsave(file, warped, cmap = "gray")       
        

    def transform_perspective(self, image):
        x = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return x

    def inverse_transform_perspective(self, image):
        x = cv2.warpPerspective(image, self.M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return x

    def hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
                
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
        line_img = np.zeros(image.shape, dtype=np.uint8)
        lines = Draw.draw_lines(line_img, lines, thickness=5)
        
        test_output_folder = "output_images/tests/transformation/birdseye/get_trans/" 
        file = "{}{}".format(test_output_folder, "hough_lines.jpg")
        plt.imsave(file, line_img, cmap = "gray")

        return lines, line_img

    def get_blank_image(self, image):
        """
            Return a blank image with the same dimensions as
            the input image
        """
        return np.zeros(image.shape, dtype=np.uint8)
  
    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines function, An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.
        
        `initial_img` should be the image before any processing.
        
        The result image is computed as follows:
        
        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def get_roi_points(self, image):
        lane = LaneLines()
        bottom_left , bottom_right = lane.get_lane_lines_base(image)
        
               
        #bottom_left = (image.shape[1] * 0.15, image.shape[0])
        #bottom_right = (image.shape[1] * 0.93, image.shape[0])
        
        top_left = (image.shape[1] * 0.30, image.shape[0]/2)
        top_right = (image.shape[1] * 0.70, image.shape[0]/2)
        return bottom_left, bottom_right, top_left, top_right

    def select_region_of_interest(self, image, plot=False):
        """
            Given an image, this function returns an image
            with only the part which is likely to contain lane lines
            not masked out
        """
        bottom_left, bottom_right, top_left, top_right = self.get_roi_points(image)
        
        white = np.zeros_like(image)
        points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(white, points , 255)
        
        masked_image = cv2.bitwise_and(image, white)
        
        if(plot):
            masked_image = cv2.bitwise_or(masked_image, cv2.polylines(masked_image, points, False, 255))
    
        return masked_image

    

