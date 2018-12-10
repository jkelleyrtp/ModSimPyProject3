import numpy as np
import imageio
from emojipy import Emoji #😂😂😂😂😂😂😂


#from modsim_cdf import image2cdf, emoji2image

def image2cdf:
    
    pass


def emoji2cdf(emoji, emoji_width = 1):
    '''
    Converts the inputted emoji to numpy array and feeds it into image2cdf.
    The emoji_width parameter helps scale the emoji properly for the cfd solver
    
    '''
    emoji_img_item = Emoji.to_image(emoji)    
    start = a.find("src=")
    link = a[start+5: -3]
    im = imageio.imread(link)
    
    