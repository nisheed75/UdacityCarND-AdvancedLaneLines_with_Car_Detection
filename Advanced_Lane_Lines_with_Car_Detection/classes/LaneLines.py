import numpy as np
import cv2
import os.path
import sys 
import os
from scipy.signal import find_peaks_cwt
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'classes/'))
from Thresholding import Thresholding
from Lines import Lines

#from Transformation import Transformation
class LaneLines(object):
    """description of class"""
    def __init__(self, transformation, n_iter=8) :
        self.line = {}
        self.fail =  {}
        self.transformation = transformation
        self.counter = 0

    def window_analysis(self, X1, Y1, X2, Y2) :
        found         = {'left':None, 'right':None}
        good_pixels_x = {'left':None, 'right':None}
        good_pixels_y = {'left':None, 'right':None}
        
        for side in ['left','right'] :
            # define region of interest
            roi={}
            for channel in self.binary :
                nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
                roi[channel] = ((nonzero_x>X1[side]) & (nonzero_x<X2[side]) & (nonzero_y>Y1) & (nonzero_y<=Y2))

            found[side], good_pixels_x[side], good_pixels_y[side] = self.get_good_pixels(roi) 

        if found['left'] or found['right'] :
            self.margin['left']  = 50 if (found['left'])  else 150
            self.margin['right'] = 50 if (found['right']) else 150
            self.x_current['left']  = np.mean(good_pixels_x['left'])  if found['left'] \
                                    else np.mean(good_pixels_x['right']) -min(600, self.lane_gap)
            self.x_current['right'] = np.mean(good_pixels_x['right']) if found['right'] \
                                    else np.mean(good_pixels_x['left'])  +min(600, self.lane_gap)
            for side in ['left','right'] : self.x_current[side] = np.int(self.x_current[side])
            if found['left'] or found['right'] :
                self.lane_gap = (self.lane_gap + self.x_last_found['right'] - self.x_last_found['left'])/2
        else :
            self.margin['left']=150 
            self.margin['right']=150
            
        return found, good_pixels_x, good_pixels_y

    def find_lines_in_windows(self, image, nb_windows=15, visualize=True, debug=False):
         # get channels and warp them
        
        self.binary = Thresholding.split_channels(image)
        self.binary = {k: self.transformation.transform_perspective(v) for k, v in self.binary.items()}

        # group A consists of all line edges and white color 
        group_A = np.dstack((self.binary['edge_pos'], self.binary['edge_neg'], self.binary['white_loose']))
        # group B consists of yellow edges and yellow color
        group_B = np.dstack((self.binary['yellow_edge_pos'], self.binary['yellow_edge_neg'], self.binary['yellow']))
        
        if visualize :
            out_img_A = np.copy(group_A)*255
            out_img_B = np.copy(group_B)*255
            out_img_C = np.zeros_like(out_img_A)
        

        
        #Set the number of windows and and the width & height for each window
        height, width = group_A.shape[:2]
        num_windows = nb_windows
        num_rows = height
        self.dims = (width,height)

        window_height = np.int(height / num_windows)
        window_width = 50
        
        midpoint = np.int(width/2)
        # window with +/- margin
        self.margin = {'left' : np.int(0.5*midpoint), 
                       'right': np.int(0.5*midpoint)}
        self.min_pixels = 100
        self.lane_gap = 600

        # center of current left and right windows
        self.x_current = {'left' : np.int(0.5*midpoint), 
                          'right': np.int(1.5*midpoint)}
        # center of left and right windows last found
        self.x_last_found = {'left' : np.int(0.5*midpoint), 
                             'right': np.int(1.5*midpoint)}
        # center of previous left and right windows
        x_prev = {'left' : None, 
                  'right': None}
        
        momentum    = {'left' :0, 
                       'right':0}
        last_update = {'left' :0, 
                       'right':0}
        self.found  = {'left' :False, 
                       'right':False}
        
        self.nonzero_x, self.nonzero_y = self.get_nonzero_pixels() 
         # good pixels
        self.good_pixels_x = {'left' : [], 'right' : []} 
        self.good_pixels_y = {'left' : [], 'right' : []} 
        
        # Step through the windows one by one
        for window in range(num_windows):
                
            # final window refinement with updated centers and margins
            Y1 = height - (window+1)*window_height
            Y2 = height - window*window_height
            X1 = {side : self.x_current[side]-self.margin[side] for side in ['left','right']} 
            X2 = {side : self.x_current[side]+self.margin[side] for side in ['left','right']} 
            if debug :
                print("-----",window, X1, X2, Y1, Y2)
               
            found, good_pixels_x, good_pixels_y = self.window_analysis(X1,Y1,X2,Y2)
            if not self.check_lanes(min_lane_gap=350, img_range=(-50,width+50)) : 
                break
                 
            for i,side in enumerate(['left','right']) :
                # Add good pixels to list
                if found[side] :
                    self.good_pixels_x[side].append(good_pixels_x[side])
                    self.good_pixels_y[side].append(good_pixels_y[side])
                    self.x_last_found[side] = self.x_current[side]
                
                # Draw the windows on the visualization image
                if visualize :
                    cv2.rectangle(out_img_A,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2)  
                    cv2.rectangle(out_img_B,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2) 
                    # Draw good pixels 
                    out_img_C[good_pixels_y[side], good_pixels_x[side],i] = 255 
        
        for side in ['left','right'] :
            if self.good_pixels_x[side] :
                self.found[side] = True
                self.good_pixels_x[side] = np.concatenate(self.good_pixels_x[side])
                self.good_pixels_y[side] = np.concatenate(self.good_pixels_y[side])
            else :
                self.good_pixels_x[side] = None
                self.good_pixels_y[side] = None
        if visualize :
            return out_img_A.astype(np.uint8), out_img_B.astype(np.uint8), out_img_C.astype(np.uint8)
   
    def get_image_with_lanes(self, image,  line, fail, nb_windows=15, visualize=True, debug=False):
        global counter
        self.counter +=1
        self.line = line
        self.fail = fail
        self.img = image
        ym_per_pix = 30/720 # meters per pixel in y dimension

        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        height, width = image.shape[:2]
        img_dim = (width, height)

        img = self.transformation.undistort_image(image)
        
        
        imgA = np.zeros_like(img)
        imgB = np.zeros_like(img)
        imgC = np.zeros_like(img)
        main_img = np.zeros_like(img).astype(np.uint8) #blank image like img

        imgA,imgB,imgC = self.find_lines_in_windows(image, nb_windows, visualize, debug) # find the lines using a window search 
        
        fit = {'left':None, 'right':None}    
        sides = ['left','right']
    
        for side in sides :
            if not self.found[side] :
                self.fail[side]+=1
                self.line[side].detected=False
            else :
                pixels_x, pixels_y = self.good_pixels_x, self.good_pixels_y
                self.line[side].add_line(pixels_x[side], pixels_y[side])
                self.line[side].detected=True
            
        if self.line['left'].check_diverging_curves(self.line['right']) or self.line['left'].fit_ratio(self.line['right'])>10 \
                or (not 400*xm_per_pix<self.line['left'].base_gap(self.line['right'])<750*xm_per_pix) :
        
            for side in sides :
                if self.line[side].delta_xfitted() > 1000 or self.line[side].res > 55: 
                    self.fail[side] += 1
                else :
                    self.line[side].update()
        else :
            for side in sides :
                if self.line[side].res > 55  :
                    self.fail[side] +=1
                elif self.line[side].detected : 
                    self.fail[side]=0
                    self.line[side].update()
            
        for side in sides :  
            fit[side] = self.line[side].avg_fit
            pts = np.array(np.vstack((self.line[side].avg_xfitted, self.line[side].yfitted)).T, dtype=np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgC,[pts],False,(255,255,0), thickness=5)
            
            pts = np.array(np.vstack((self.line[side].current_xfitted, self.line[side].yfitted)).T, dtype=np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgC,[pts],False,(0,255,255), thickness=2)
            self.line[side].calc_R(self.line[side].avg_fit)
            self.line[side].calc_base_dist(self.line[side].avg_fit)

        R_avg = (self.line['left'].radius_of_curvature + self.line['right'].radius_of_curvature)/2
        base_gap = (self.line['left'].base_gap(self.line['right']))
        center_pos = (self.line['left'].line_base_pos + self.line['right'].line_base_pos)/2

        main_img = self.plot_lane(fit) #This is where we plot the lane lines on the image
        filename = "file{}.jpg".format(self.counter)
        if debug:         
            img_A = cv2.resize(imgA,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            hA,wA = img_A.shape[:2]
            img_B = cv2.resize(imgB,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            hB,wB = img_B.shape[:2]
            img_C = cv2.resize(imgC,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            text_A = np.zeros((int(hA/4), wA,3))
            h_text = text_A.shape[0]    
            text_B = np.zeros((h_text, wA,3))
            text_C = np.zeros((h_text, wA,3))
        else:   
            text_A = np.zeros((int(height/4), width,3))
            h_text = text_A.shape[0] 
            text_B = np.zeros((h_text, width,3))
            text_C = np.zeros((h_text, width,3))
        
        text_A = text_A.astype(np.uint8)
        text_B = text_B.astype(np.uint8)
        text_C = text_C.astype(np.uint8)

        for i in range(1,3) :
            text_A[:,:,i] = 255
            text_B[:,:,i] = 255
            text_C[:,:,i] = 255
        
        
        font = cv2.FM_8POINT
        cv2.putText(text_A,'Threshold',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(text_B,'Threshold (yellow)',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(text_C,'Best fit',(10,h_text-20), font,1,(0,0,255),3,cv2.LINE_AA)
        
        if debug: 
            img_combined_right = np.vstack((text_A, img_A, text_B, img_B, text_C, img_C))
            main_text = np.zeros((3*h_text+3*hA-height,width,3)).astype(np.uint8)
        else:
            main_text = np.zeros((int(0.8*h_text),width,3)).astype(np.uint8)

        h_main_text, w_main_text = main_text.shape[:2]
        cv2.putText(main_text,'Radius of curvature : {:5.2f} m'.format(abs(R_avg)),
                    (10,35), font, 1,(255,255,255),3,cv2.LINE_AA)
        shift = "left" if center_pos>0 else "right"
        cv2.putText(main_text,'Vehicle is {:6.2f} m {:5} of center'.format(abs(center_pos), shift),
                    (10,80), font, 1,(255,255,255),3,cv2.LINE_AA)
        if self.line['left'].avg_fit[0]>0.0001 and  self.line['right'].avg_fit[0]>0.0001 :
            cv2.putText(main_text,'Right curve ahead',
                    (10,135), font, 1,(100,100,25),3,cv2.LINE_AA)
        elif self.line['left'].avg_fit[0]<-0.0001 and  self.line['right'].avg_fit[0]<-0.0001 :
            cv2.putText(main_text,'Left curve ahead',
                    (10,135), font, 1,(100,100,25),3,cv2.LINE_AA)
        img_combined_left = np.vstack((main_img, main_text))
        
        
        if debug:
            final = np.hstack((img_combined_left, img_combined_right))
            self.plot_image("img_A", filename, img_A)
            self.plot_image("img_B", filename, img_B)
            self.plot_image("img_C", filename, img_C)
        else: 
            final = img_combined_left
        if visualize:
            self.plot_image("main_img", filename, main_img)
            self.plot_image("final", filename, final)
 
        return final, self.line, self.fail
        
    def get_good_pixels(self, roi, min_pix=100, max_pix=10000, window_search=True, debug=False) :
        nb_pixels, x_mean, x_stdev = {},{},{}
        for channel in self.binary :
            nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
            # region of interest
            roi_ = roi[channel]
            count = np.sum(roi_)
            if count<min_pix or (count>max_pix and window_search) :
                continue
#             
            nb_pixels[channel] = count
            x_mean[channel]    = np.mean(nonzero_x[roi_])
            x_stdev[channel]   = np.std(nonzero_x[roi_])
            if debug :
                print(channel, count, x_mean[channel], x_stdev[channel])
        
        if window_search :
            selected_channels = [c for c in x_stdev.keys() if x_stdev[c]<35]
        else :
            selected_channels = [c for c in x_stdev.keys()]
        # some consistency checks to select channels
        if 'edge_pos' in selected_channels :
            if ('edge_neg' not in x_stdev.keys()) :
                selected_channels.remove('edge_pos')
            elif (nb_pixels['edge_neg'] < nb_pixels['edge_pos']/3) or \
                    (abs(x_mean['edge_neg']-x_mean['edge_pos'])>50) :
                selected_channels.remove('edge_pos')
                if 'edge_neg' in selected_channels : selected_channels.remove('edge_neg')
            elif window_search and (x_mean['edge_pos'] > x_mean['edge_neg']) :
                selected_channels.remove('edge_pos')
                if 'edge_neg' in selected_channels : selected_channels.remove('edge_neg')
        if 'edge_neg' in selected_channels :
            if ('edge_pos' not in x_stdev.keys()) :
                selected_channels.remove('edge_neg')
            elif (nb_pixels['edge_pos'] < nb_pixels['edge_neg']/3) or \
                    (abs(x_mean['edge_neg']-x_mean['edge_pos'])>50) :
                if 'edge_pos' in selected_channels : selected_channels.remove('edge_pos')
                selected_channels.remove('edge_neg')
            elif window_search and (x_mean['edge_pos'] > x_mean['edge_neg']) :
                selected_channels.remove('edge_neg')
                if 'edge_pos' in selected_channels : selected_channels.remove('edge_pos')
        if 'yellow_edge_pos' in selected_channels :
            if ('yellow_edge_neg' not in x_stdev.keys()) :
                selected_channels.remove('yellow_edge_pos')
            elif (nb_pixels['yellow_edge_neg']< nb_pixels['yellow_edge_pos']/3) or \
                    (abs(x_mean['yellow_edge_neg']-x_mean['yellow_edge_pos'])>50) :
                if 'yellow_edge_neg' in selected_channels : selected_channels.remove('yellow_edge_neg')
                selected_channels.remove('yellow_edge_pos')
            elif ('yellow' in selected_channels) and (abs(x_mean['yellow']-x_mean['yellow_edge_pos'])>20):
                selected_channels.remove('yellow_edge_pos')
        if 'yellow_edge_neg' in selected_channels :
            if ('yellow_edge_pos' not in x_stdev.keys()) :
                selected_channels.remove('yellow_edge_neg')
            elif (nb_pixels['yellow_edge_pos']< nb_pixels['yellow_edge_neg']/3) or \
                    (abs(x_mean['yellow_edge_pos']-x_mean['yellow_edge_neg'])>50) : 
                selected_channels.remove('yellow_edge_neg')
                if 'yellow_edge_pos' in selected_channels : selected_channels.remove('yellow_edge_pos')
            elif ('yellow' in selected_channels) and (abs(x_mean['yellow']-x_mean['yellow_edge_neg'])>20):
                selected_channels.remove('yellow_edge_neg')
        if ('white_tight' in selected_channels) :
            if 'white_loose' in selected_channels : selected_channels.remove('white_loose')
            if nb_pixels['white_tight']>8000 or (window_search and nb_pixels['white_tight']>2000) :
                selected_channels.remove('white_tight')
        if 'white_loose' in selected_channels and (nb_pixels['white_loose']<100 or nb_pixels['white_loose']>5000):  
            selected_channels.remove('white_loose')
        if window_search and 'white_loose' in selected_channels and nb_pixels['white_loose']>500 : 
            selected_channels.remove('white_loose')
        if len(selected_channels)==1 and 'yellow' in selected_channels and nb_pixels['yellow']<300 :
            selected_channels.remove('yellow')
        
        if debug :
            print("selected " , selected_channels)
        
        # combine the selected channels
        comb_nonzero_x, comb_nonzero_y = [],[]
        for channel in selected_channels :
            nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
            roi_ = roi[channel]
            comb_nonzero_x.append(nonzero_x[roi_])
            comb_nonzero_y.append(nonzero_y[roi_])
        if comb_nonzero_x :
            return (True, np.concatenate(comb_nonzero_x), np.concatenate(comb_nonzero_y))
        else :
            return (False,None,None)

    def get_nonzero_pixels(self) :
        nonzero_x, nonzero_y = {},{}
        for channel in self.binary :
            nonzero = self.binary[channel].nonzero()
            nonzero_x[channel], nonzero_y[channel] = (np.array(nonzero[1]), np.array(nonzero[0]))
        return nonzero_x, nonzero_y

    def check_lanes(self, img_range, min_lane_gap=350, max_lane_gap=750) :
        #checks that the gap is with a range and the lines detected are in the frame 
        if self.lane_gap < min_lane_gap : return False 
        if self.lane_gap > max_lane_gap : return False
        elif self.x_current['left']<img_range[0] or self.x_current['right']>img_range[1] : return False
        else : return True
   
    def plot_lane(self, fit, poly_order=2) :
        
        width,height = self.dims
        y = np.linspace(0, height-1, height)
        x_fit = {'left':None, 'right':None}
        for side in ['left','right'] :
            if fit[side] is not None :
                x_fit[side] = fit[side][poly_order]
                for i in range(poly_order) :
                    x_fit[side] += fit[side][i]*y**(poly_order-i)
                x_fit[side] = x_fit[side]
                    
        if (x_fit['left'] is not None) and (x_fit['right'] is not None) :
            lane_img = np.zeros((height,width+300,3))
            pts_x = np.hstack((x_fit['left'],x_fit['right'][::-1]))
            pts_y = np.hstack((y,y[::-1]))
            pts = np.vstack((pts_x, pts_y)).T

            cv2.fillPoly(lane_img, np.int32([pts]), (0,255, 0))

            # unwarp image
            
            lane_img = cv2.warpPerspective(lane_img, self.transformation.M_inv, (self.img.shape[1], self.img.shape[0]))
            
            out_img = cv2.addWeighted(self.img, 1, np.uint8(lane_img), 0.3, 0)
            return out_img
        else :
            return self.img
    
    def plot_image(self, prefix, output_filename, img):
         out_file = self.get_output_file(prefix, output_filename) 
         mpimg.imsave(out_file, img, cmap = "gray")

    def get_output_file(self, prefix, filename):
        mode = "debug"
        obj= "pipeline"
        action = "lanelines"
        file = filename.split("\\")[-1]
        output_filename = "{}_{}".format(prefix, file)
        output_folder ="output_images/{}/{}/{}".format(mode, obj, action)
        output_file = "{}/{}".format (output_folder, output_filename)
        
        #make sure the output floder are present
        if not (os.path.exists(output_folder)):
            os.makedirs(output_folder)
        
        return output_file
