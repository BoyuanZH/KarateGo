# Boyuan Zhang.
# module designed for UI for karateGo.

###################################################
# imports
###################################################
import pygame
import os

NUM_BUTTONS = 6

class Button(object):
    def __init__(self, x, y, surface, h, type = None):
        self.isLocked = False
        self.type = type
        self.isSelected = False
        self.isHover = False
        self.x, self.y = x, y
        self.image_up = None
        self.image_down = None
        self.rect = None
        self.surface = surface
        self.height = h
        #self.height = surface.get_height() // (NUM_BUTTONS + 2)
        self.recNum = 0
    # input filename, can be 'button_up.jpg'
    def __repr__(self):
        return ''
    def setIcon(self, filename_up, filename_down):
        # set the button up icon
        original = pygame.image.load(os.path.join('images', filename_up))
        h_to_w = original.get_height() / original.get_width()
        self.width = int(self.height/h_to_w)
        self.image_up = pygame.transform.scale(original, (self.width, self.height))
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        # set the button down icon
        original = pygame.image.load(os.path.join('images', filename_down))
        self.image_down = pygame.transform.scale(original, (self.width, self.height))

    def show(self):
        if (not self.isSelected and not self.isHover):
            self.surface.blit(self.image_up, (self.x, self.y))
        else:
            self.surface.blit(self.image_down, (self.x, self.y))
    def get_selected(self):
        self.isSelected = True
        self.show()
    def get_unselected(self):
        self.isSelected = False
        self.show()
    def get_hover(self):
        self.isHover = True
        self.show()
    def get_unhover(self):
        self.isHover = False
        self.show()
    def set_recTime(self, recTime):
        self.recTime = recTime
    
    def getLocked(self):
        self.isLocked = True

    def getUnlocked(self):
        self.isLocked = False

# only use the board class when the image is in the right size
class Board(object):
    def __init__(self, filename, x, y, surface):
        self.im = pygame.image.load(os.path.join('images', filename))
        self.x, self.y = x, y
        self.surface = surface
        self.rect = pygame.Rect(x, y, self.im.get_width(), self.im.get_height())


    def show(self):
        self.surface.blit(self.im, (self.x, self.y))

    def get_midPos_abs(self):
        return self.rect.center

    def get_midPos_adjusted(self):
        x, y = self.rect.center
        return (x -15, y + 20)

    
    def get_timerPos(self):
        return self.rect.midbottom
    
    def print(self, text, timer = False):
        self.show()
        if timer:
            # if is to print on the timer
            self.surface.blit(text, self.get_midPos_adjusted())
        else:
            # if is to print on the infoboard.
            (x, y) = self.get_midPos_abs()
            (x, y) = (x - 130, y - 40)
            self.surface.blit(text, (x, y))



     
