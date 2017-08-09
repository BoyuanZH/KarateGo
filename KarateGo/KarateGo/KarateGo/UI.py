# Boyuan Zhang.
# module designed for UI for karateGo.

###################################################
# imports
###################################################
import pygame
import os

NUM_BUTTONS = 7

class Button(object):
    def __init__(self, x, y, surface):
        self.isSelected = False
        self.isHover = False
        self.x, self.y = x, y
        self.image_up = None
        self.image_down = None
        self.rect = None
        self.surface = surface
        self.height = surface.get_height() // (NUM_BUTTONS + 2)
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


     
