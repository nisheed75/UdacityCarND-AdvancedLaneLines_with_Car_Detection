import numpy as np
import sys 

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
img_dim = (1280, 720)
class Lines(object):
    """description of class"""
   
    def __init__(self, n_iter=8) :
        # number of iterations to use for avraging/smoothing
        self.n_iter = n_iter
        # was the line detected in the last iteration?
        self.detected = False
        # counter
        self.counter = 0
        
        
        # x values of the current and last n fits of the line
        self.current_xfitted = None 
        self.recent_xfitted = [] 
        self.avg_xfitted = None
        self.yfitted = np.linspace(0, img_dim[1]-1, img_dim[1])
        
        # polynomial coefficients of the current and last n iterations
        self.current_fit = None
        self.recent_fits = []
        self.avg_fit = None
        self.res = None # residuals from fit
        
        self.current_pixels = None
        self.previous_pixels = []
        self.radius_of_curvature = None
        self.line_base_pos = None
        # x values for detected line pixels
        self.pixels_x = None
        # y values for detected line pixels
        self.pixels_y = None
        
    def add_line(self,x_pixels, y_pixels) :
        self.detected = True
        self.counter += 1
        self.pixels_x = x_pixels
        self.pixels_y = y_pixels
        self.curve_fit()
        self.calc_R(self.current_fit)
        self.calc_base_dist(self.current_fit)
        
    def curve_fit(self, poly_order=2):
        fit, self.res,_,_,_ = np.polyfit(self.pixels_y, self.pixels_x, poly_order, full=True)
        self.res = self.res/len(self.pixels_x)**1.2
        self.poly_order=poly_order
        self.current_fit = fit
        
        h = img_dim[1]
        y = self.yfitted
        x_fit = fit[poly_order]
        for i in range(poly_order) :
            x_fit += fit[i]*y**(poly_order-i)
        self.current_xfitted = x_fit
        if self.avg_xfitted is None :
            self.update()
            
    def calc_R(self, fit) :
        y=img_dim[1]
        self.radius_of_curvature = ((ym_per_pix**2 + xm_per_pix**2*(2*fit[0]*y + fit[1])**2)**1.5)/(2
                                    *xm_per_pix*ym_per_pix*fit[0])
    
    def calc_base_dist(self, fit) :
        y = img_dim[1]
        dist = -img_dim[0]/2
        for i in range(self.poly_order+1) :
            dist += fit[i]*y**(self.poly_order-i)
        self.line_base_pos = dist*xm_per_pix
            
    def update(self) :
        if len(self.recent_fits) >= self.n_iter : 
            self.recent_xfitted.pop(0)
            self.recent_fits.pop(0)
        
        self.recent_xfitted.append(self.current_xfitted)
        self.avg_xfitted = np.average(np.array(self.recent_xfitted), axis=0)
        self.recent_fits.append(self.current_fit)
        self.avg_fit = np.average(np.array(self.recent_fits), axis=0)
        self.calc_R(self.avg_fit)
        self.calc_base_dist(self.avg_fit)
        self.detected = False
        
    def radius_ratio(self, other_line) :
        delta_r = abs(self.radius_of_curvature-other_line.radius_of_curvature)
        min_r = min(abs(self.radius_of_curvature),abs(other_line.radius_of_curvature))
        return delta_r/min_r
    
    def check_diverging_curves(self, other_line):
        R1 = self.radius_of_curvature
        R2 = other_line.radius_of_curvature
        if max(abs(R1), abs(R2)) > 1500 :
            return False
        else :
            return (R1*R2<0) 
    
    def fit_ratio(self, other_line):
        fit1 = np.array(self.current_fit)
        fit2 = np.array(other_line.current_fit)
        delta_fit = fit1-fit2
        min_fit = np.minimum(np.absolute(fit1), np.absolute(fit2)) 
        return np.linalg.norm(delta_fit[:2]/min_fit[:2])*2000/(
            abs(self.radius_of_curvature) + abs(other_line.radius_of_curvature))
    
    def base_gap(self, other_line):
        return abs(self.line_base_pos-other_line.line_base_pos)
    
    def delta_xfitted(self):
            return np.linalg.norm(self.current_xfitted - self.avg_xfitted)