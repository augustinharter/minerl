import torch as T
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import pickle
import minerl
    
def vis_batch(batch, path, pic_id, text = [], rows=[], descr=[], save=True, font_size=11):
    #print(batch.shape)

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1,1,1,1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0,0,1,1,1,1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    #print(padded.shape)
    reshaped = T.cat([T.cat([channels for channels in sample], dim=1) for sample in padded], dim=0)
    #print(reshaped.shape)
    if np.max(reshaped.numpy())>1.0:
        reshaped = reshaped/256
    os.makedirs(path, exist_ok=True)
    if text or rows or descr:
        if rows:
            row_width = font_size//2*max([len(item) for item in rows])
        else:
            row_width = 0

        if descr:
            descr_wid = font_size//2*max([len(item) for item in descr])
        else:
            descr_wid = 0
        if text:
            text_height= font_size*max([len(item.split('\n')) for item in text])
        else:
            text_height=0

        if len(reshaped.shape) == 2:
            reshaped = F.pad(reshaped, (row_width,descr_wid,text_height,0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255), mode="L")
        elif len(reshaped.shape) == 3:
            reshaped = F.pad(reshaped, (0,0,row_width,descr_wid,text_height,0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255))
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
        draw = ImageDraw.Draw(img)
        
        for i, words in enumerate(text):
            x, y = row_width+i*(reshaped.shape[1]-row_width-descr_wid)//len(text), 0
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)

        for j, words in enumerate(rows):
            x,y = 3, 10+text_height+j*(reshaped.shape[0]-text_height)//len(rows)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)
        
        for j, words in enumerate(descr):
            x,y =  5+reshaped.shape[1]-descr_wid, text_height+j*(reshaped.shape[0]-text_height)//len(descr)
            #print(x,y)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)

        if save:
            img.save(f'{path}/'+pic_id+'.png')
        else:
            return img
    else:
        if save:
            plt.imsave(f'{path}/'+pic_id+'.png', reshaped.numpy(), dpi=1000)
        else:
            return reshaped