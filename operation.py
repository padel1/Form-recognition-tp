import pygame
import numpy as np
from math import sin, cos, radians

pygame.init()


class Opertation():
    def __init__(self, screen):
        self.screen = screen




    def draw_histogram(self, image: pygame.Surface):

        width, height = image.get_size()

        hist = [0] * 256

        for y in range(height):
            for x in range(width):
                r, g, b, a = image.get_at((x, y))
                hist[r] += 1
        max_count = max(hist)
        hist = [int(h * 250 / max_count) for h in hist]
        bar_width = (self.screen.get_width() / len(hist))-1
        xx = 0
        yy = 0
        for i, h in enumerate(hist):
            x = i * bar_width + self.screen.get_width()//2 - bar_width*len(hist)//2
            y = self.screen.get_height() - h - 100

            if(i == 0):
                xx = x
                yy = y+h+10

            pygame.draw.rect(self.screen, (i, i, i),pygame.Rect(x, y, bar_width, h))

        pygame.draw.rect(self.screen, (200,200, 200),pygame.Rect(xx, yy, 256*bar_width, 2))

    def brightness(self, image, inc):

        width, height = image.get_size()
        for x in range(width):
            for y in range(height):
                r, g, b, a = image.get_at((x, y))
                new_r = r + inc
                new_g = g + inc
                new_b = b + inc
                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                image.set_at((x, y), (new_r, new_g, new_b, a))
        return image

    def convolution(self, img, new_image, masque):

        for x in range(img.get_width()):
            for y in range(img.get_height()):
                new_r, new_g, new_b = 0, 0, 0
                for i in range(len(masque)):
                    for j in range(len(masque)):
                        try:
                            old_r, old_g, old_b, _ = img.get_at(
                                (x+i-(len(masque)//2), y+j-(len(masque)//2)))
                        except:
                            old_r, old_g, old_b = 0, 0, 0
                        new_r += old_r * masque[i][j]
                        new_g += old_g * masque[i][j]
                        new_b += old_b * masque[i][j]

                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                new_image.set_at((x, y), (new_r, new_g, new_b))
        return new_image

    def redimensionner_image(self, image_orig, scale_level):
        largeur_orig, hauteur_orig = image_orig.get_size()
        largeur_nouv,hauteur_nouv = pygame.math.Vector2(image_orig.get_size())*(scale_level)
        image_nouv = pygame.Surface((largeur_nouv,hauteur_nouv))
        
        for x in range(int(largeur_nouv)):
            for y in range(int(hauteur_nouv)):
                x_orig = x * largeur_orig // int(largeur_nouv)
                y_orig = y * hauteur_orig // int(hauteur_nouv)
                couleur = image_orig.get_at((x_orig, y_orig))
                image_nouv.set_at((x, y), couleur)
        # image_nouv = pygame.transform.smoothscale(image_orig, (largeur_nouv, hauteur_nouv))

        return image_nouv
    
    def rotate_image(self,image, angle):
        width, height = image.get_size()

        center_x = width / 2
        center_y = height / 2

        theta = radians(angle)
        rotation_matrix = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])

        corners = np.array([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])
        rotated_corners = np.dot(corners - np.array([center_x, center_y]), rotation_matrix) + np.array([center_x, center_y])
        min_x, min_y = np.min(rotated_corners, axis=0)
        max_x, max_y = np.max(rotated_corners, axis=0)

        rotated_width = int(np.ceil(max_x - min_x))
        rotated_height = int(np.ceil(max_y - min_y))
        rotated_image = pygame.Surface((rotated_width, rotated_height))
        rotated_image.fill((22, 27, 37 ))

        for y in range(height):
            for x in range(width):
                new_x, new_y = np.dot(rotation_matrix, np.array([x - center_x, y - center_y])) + np.array([center_x, center_y])

                color = image.get_at((x, y))
                rotated_image.set_at((int(new_x - min_x), int(new_y - min_y)), color)

        return rotated_image

    def split_vertical(self,image):

        list_of_numbers =[]
        c = []
        found = False
        already_seen_dark  = False
        surface_array = []

        for x in  range(image.get_width()):
            for y in  range(image.get_height()):
                    color = image.get_at((x , y))[0]
                    c.append(list(image.get_at((x , y)))[:-1])
                    if color == 0:
                        found = True
                        already_seen_dark = True


            if found:
                surface_array.append(c)
            else:
                if already_seen_dark:
                    list_of_numbers.append(np.array(surface_array))
                    already_seen_dark = False
                    surface_array = []
                
            found = False
            c = []    
        return list_of_numbers

    def split_horizontal(self,image):

        list_of_numbers =[]
        c = []
        found = False
        already_seen_dark  = False
        surface_array = []

        for x in  range(image.get_height()):
            for y in  range(image.get_width()):
                    color = image.get_at((y , x))[0]
                    c.append(list(image.get_at((y , x)))[:-1])
                    if color == 0:
                        found = True
                        already_seen_dark = True


            if found:
                surface_array.append(c)
            else:
                if already_seen_dark:
                    list_of_numbers.append(np.array(surface_array))
                    already_seen_dark = False
                    surface_array = []
                
            found = False
            c = []    
        return list_of_numbers



    def dilatation(self,img,masque):
        result_img = pygame.Surface(img.get_size())
        for x in range(img.get_width()):
            for y in range(img.get_height()):

                overlap = False
                for i in range(len(masque)):
                    for j in range(len(masque)):
                        try:
                            if (img.get_at((x+i-(len(masque)//2), y+j-(len(masque)//2)))[0] == 255 and masque[i][j]==1  ):
                                overlap = True
                                break
                        except:
                            pass
                    if(overlap):
                        break    
                
                if overlap:
                    result_img.set_at((x, y), (255,255, 255))
                else: 
                    result_img.set_at((x, y),img.get_at((x, y)))
        return result_img

    def erosion(self,img,masque):
        result_img = pygame.Surface(img.get_size())

        for x in range(img.get_width()):
            for y in range(img.get_height()):

                overlap = True
                for i in range(len(masque)):
                    for j in range(len(masque)):
                        try:
                            if (img.get_at((x+i-(len(masque)//2), y+j-(len(masque)//2)))[0] == 0 and masque[i][j]==1  ):
                                overlap = False
                                break
                        except:
                            pass
                    if(overlap):
                        break    
                
                if not overlap:
                    result_img.set_at((x, y), (0,0, 0))
                else: 
                    result_img.set_at((x, y),img.get_at((x, y)))

        return result_img
    def overture(self,img,masque):
        return self.dilatation(self.erosion(img,masque),masque)
    def fermeture(self,img,masque):
        return self.erosion(self.dilatation(img,masque),masque)
    def counteur(self,img,masque):
        return pygame.surfarray.make_surface(np.array(pygame.surfarray.array3d(self.dilatation(img,masque))) - np.array(pygame.surfarray.array3d(img)))


