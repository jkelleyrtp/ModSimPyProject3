import numpy as np
import imageio
from emojipy import Emoji #😂😂😂😂😂😂😂


#from modsim_cdf import image2cdf, emoji2image

def image2cdf(image, image_width = 1):
    
    aerodynamics = {'coefficient_of_drag' : .3, 
                    'coefficient_of_lift' : 1,
                    'wing_area'           : 1,
                    'frontal_area'        : .2,
                    'properties'          : 1}
    return aerodynamics
    
def emoji2cdf(emoji, emoji_width = 1):
    '''
    Converts the inputted emoji to numpy array and feeds it into image2cdf.
    The emoji_width parameter helps scale the emoji properly for the cfd solver
    
    '''
    emoji_img_item = Emoji.to_image(emoji)    
    start = a.find("src=")
    link = a[start+5: -3]
    im = imageio.imread(link)
    
    aerodynamics = {'coefficient_of_drag' : .3, 
                    'coefficient_of_lift' : 1,
                    'wing_area'           : 1,
                    'frontal_area'        : .2,
                    'properties'          : 1}
    return aerodynamics
    
    