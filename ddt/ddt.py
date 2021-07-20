from typing import Sequence, List
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from math import cos, sin, atan
from copy import deepcopy

import numpy as np
import cv2


class DdtImage:
    def __init__(self,image:np.ndarray, label_color:dict) -> None:
        #저장된 값이 BGR이라 가정
        self.label_color = label_color
        self.image = image
        self.fontscale = np.max(self.image.shape[:2]*0.02)
        self.thick = int(np.max([*list(self.image.shape[:2]*0.002),1]))
        fontpath = Path(__file__).parent/'NanumGothicBold.ttf'
        self.font =ImageFont.truetype(str(fontpath), int(self.fontscale))

    @staticmethod
    def return_order_changed_image(image):
        '''
        RGB(A) to BGR(A), BGR(A) to RGB(A)
        '''
        img = deepcopy(image)
        if len(img.shape) == 3 and img.shape[2] in {3, 4}:
            img[:, :, :3] = img[:, :, 2::-1]
        return img

    def getColor(self,label, order='BGR')->np.ndarray:
        #저장된 값이 BGR이라 가정
        color = self.label_color[label]
        channel = self.image.shape[2]
        if order=='BGR':
            return color if channel==3 else color+(255,)
        elif order=='RGB':
            return color[::-1] if channel==3 else color[::-1]+(255,)
        else:
            raise AssertionError(f'order 인수를 바르게 입력하세요. 현재 입력={order}')

    def drawBbox(self,label, bbox:Sequence, lineStyle='solid', fill=True, tag=False):
        assert lineStyle in ['solid','dot','no'],'"lineStyle" 인수는 "solid"와 "dot","no" 중 하나여야 합니다.'
        color = self.getColor(label)
        self._rectangle((bbox[0], bbox[1]), (bbox[2], bbox[3]), color, self.thick, linestyle=lineStyle)
        if fill:
            self.fill(lambda :self._rectangle((bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1, linestyle=lineStyle))
        if tag:
            self.drawLabel(label,bbox)
        return self

    def _rectangle(self, topleft, bottomright, color, thick, linestyle='solid'):
        if linestyle=='solid' or thick==-1: #solid or fill_yes
            cv2.rectangle(self.image, (topleft[0], topleft[1]), (bottomright[0], bottomright[1]), color, thick)
        elif linestyle=='dot': #dot
            points = [(topleft[0],topleft[1]),(topleft[0],bottomright[1]),(bottomright[0],bottomright[1]),(bottomright[0],topleft[1])]
            points = [np.array(point) for point in (points + [points[0]])]
            self._dotlines(self.image, points,color)
        elif linestyle=='no' :
            pass
        else:
            Exception('Check out the outline.')
        
    def _get_polar(self, vector):
        return np.array([self._get_length(vector),self._get_angle(vector)])

    @staticmethod
    def _get_length(vector):
        return np.sqrt(np.sum(vector**2))

    @staticmethod
    def _get_angle(vector):
        angle = np.sign(vector[1])*np.pi/2 if vector[0]==0 else atan(vector[1]/vector[0])
        return angle+np.pi if vector[0]<0 else angle

    @staticmethod
    def _get_vector(polar_vector):
        length, angle = polar_vector
        return length*np.array([cos(angle),sin(angle)])

    def _get_dot_points(self,polygons:List[np.ndarray], dot_length, rest_dot_length:float):
        polygons = polygons[:]
        vector = polygons[1]-polygons[0]
        r,angle = self._get_polar(vector)
        rest_dot_vector = self._get_vector((rest_dot_length,angle))
        expect_dot_end = polygons[0]+rest_dot_vector
        dot_r, dot_angle = self._get_polar(expect_dot_end-polygons[0])
        if dot_r<=r:
            polygons.insert(1,expect_dot_end)
            if np.all(expect_dot_end==polygons[-1]):
                return [(polygons[0],expect_dot_end)]
            else:
                return [(polygons[0],expect_dot_end)]+self._get_dot_points(polygons[1:],dot_length, dot_length)
        else:
            rest = dot_length-(rest_dot_length-self._get_length(polygons[1]-expect_dot_end))
            if len(polygons[1:])>1:
                next_polygons = self._get_dot_points(polygons[1:],dot_length, rest)
                return [(polygons[0],)+next_polygons[0]]+next_polygons[1:]
            else:
                return [(polygons[0],)]

    def _dotlines(self,polygons, color):
        dot_length = self.thick*5
        dot_points = self._get_dot_points(polygons,dot_length,dot_length)
        dot_points = [np.array([point.astype(int) for point in dots]) for dots in dot_points[::2]]
        cv2.polylines(self.image, dot_points, False, color, self.thick)

    def drawLabel(self, label, bbox):
        color_rgb = self.getColor(label,order='RGB')
        textColor = tuple(np.array([255,255,255]) - np.array(color_rgb))
        #draw tag
        tag_background = Image.new('RGB', (int(self.fontscale*100),int(self.fontscale*1.5)),color=color_rgb)
        draw = ImageDraw.Draw(tag_background)
        draw.text((0, 0), label, font = self.font, fill = textColor,anchor='lt')
        w, h = draw.textsize(label, font=self.font)
        tag_background = tag_background.crop((0,0,w,int(h*1.1)))
        text_y = max(bbox[1] - h,0) #label 위치 조정
        img = Image.fromarray(self.return_order_changed_image(self.image))
        img.paste(tag_background,(bbox[0],text_y))
        self.image = self.return_order_changed_image(np.array(img))
    
    def drawLegend(self):
        img = Image.fromarray(self.return_order_changed_image(self.image))
        draw = ImageDraw.Draw(img)
        _,h = draw.textsize('q', font=self.font)
        text_h = 2
        for label, color in self.label_color.items():
            draw.text((self.img.shape[1]-2, text_h), label, font = self.font, fill=color[::-1] , 
                        stroke_width=np.max([int(self.thick*0.3),1]), stroke_fill='white', anchor='rt')
            text_h+=h
        self.image = self.return_order_changed_image(np.array(img))

    def drawSeg(self, label, polygons, lineStyle, fill=True):
        color = self.getColor(label, 'BGR')
        polygons = [int(i) for i in polygons + polygons[:2]]
        polygons = [ np.array((polygons[idx],polygons[idx+1])) for idx in range(len(polygons))[::2] ]
        if lineStyle=='solid':
            cv2.polylines(self.image,[np.array(polygons)],True,color,self.thick)
        elif lineStyle=='dot':
            self._dotlines(polygons,color)
        elif lineStyle=='no':
            pass
        else:
            Exception('Check out the outline.')

        if fill:
            self.fill(lambda :cv2.fillPoly(self.image,pts=[np.array(polygons[:-1])],color=color))
        return self

    def fill(self, func):
        fill_img = np.copy(self.image)
        fill_img = func()
        self.image = np.mean([self.image,self.image,fill_img],axis=0).astype(np.uint8)
