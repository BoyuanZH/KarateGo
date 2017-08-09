# Boyuan Zhang
# Citations:
# 1. draw_body function src from: https://github.com/Kinect/PyKinect2/blob/master/examples/PyKinectBodyGame.py


###############################################
# Imports
###############################################
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
from sklearn.externals import joblib
from sklearn import cluster

import os
import ctypes
import _ctypes
import pygame
import sys
import UI
import KNNRecog
import random

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# color for drawing the skeleton
SKELETON_COLOR = (40, 40, 40)
CIRCLE_COLOR = pygame.color.THECOLORS['orange']
CIRCLE_R = 30
BACKGROUND_COLOR = (232, 232, 232)

BACKGROUND_COLOR_INTRO = (181, 181, 181)


class KarateGo(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        self.lock = False

        # modes: INTRO, REC, GAME, INFO_REC, INFO_INTRO
        self.mode = 'INTRO'

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()

        
        #self.h_to_w = float(self._infoObject.current_h) / self._infoObject.current_w
        #screen_size = (self._infoObject.current_w >> 1, int((self._infoObject.current_w >> 1)*self.h_to_w))

        self.h_to_w = float(self._kinect.color_frame_desc.Height) / self._kinect.color_frame_desc.Width
        screen_size = (self._infoObject.current_w >> 1, int((self._infoObject.current_w >> 1)*self.h_to_w))
        self._screen = pygame.display.set_mode(screen_size, 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("KarateGo! 15-112 Term Project")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        self.ready = False


        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        # another buffer surface for drawing the game world (not showing the color frame)
        self._game_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        # init a small count down timer for recording movement.
        self.rec_time = 60

        # text to print on the infoboard.
        self.info = '''Welcome!'''
        
        def initIntro(self):
            self.intro_logo = pygame.image.load(os.path.join('images','intro_logo.jpg'))
            self.intro_logo = pygame.transform.scale(self.intro_logo, (self.intro_logo.get_width()//1, self.intro_logo.get_height()//1))
            button_size = self.intro_logo.get_height() * 3 // 8

            self.introButton_REC = UI.Button(1050,100, self._game_surface, button_size, type = 'REC')
            filename_up, filename_down = 'intro_learn_down.jpg', 'intro_learn_up.jpg'
            self.introButton_REC.setIcon(filename_up, filename_down)

            self.introButton_GAME = UI.Button(1250,350, self._game_surface, button_size, type = 'GAME')
            filename_up, filename_down = 'intro_game_down.jpg', 'intro_game_up.jpg'
            self.introButton_GAME.setIcon(filename_up, filename_down)

            self.introButton_INFO = UI.Button(1050,600, self._game_surface, button_size, type = 'INFO_INTRO')
            filename_up, filename_down = 'intro_help_down.jpg', 'intro_help_up.jpg'
            self.introButton_INFO.setIcon(filename_up, filename_down)

        def initRec(self):
            button_size = 150
            # add a button named record
            # list of button on a row (the rec buttons)
            rec_button_h = 20 # this keeps same
            rec_button_w = 550
            rec_button_i = 180

            # rec buttons list:
            # rec front kick
            self.recButton_LRK = UI.Button(rec_button_w + 0 * rec_button_i, rec_button_h, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_LRK.jpg', 'button_down_LRK.jpg'
            self.recButton_LRK.setIcon(filename_up, filename_down)
            self.recButton_LRK.set_recTime(self.rec_time)

            # rec round kick
            self.recButton_RRK = UI.Button(rec_button_w + 1 * rec_button_i, rec_button_h, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_RRK.jpg', 'button_down_RRK.jpg'
            self.recButton_RRK.setIcon(filename_up, filename_down)
            self.recButton_RRK.set_recTime(self.rec_time)

            # rec middle punch
            self.recButton_LMP = UI.Button(rec_button_w + 2 * rec_button_i, rec_button_h, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_LMP.jpg', 'button_down_LMP.jpg'
            self.recButton_LMP.setIcon(filename_up, filename_down)
            self.recButton_LMP.set_recTime(self.rec_time)

            # rec high block
            self.recButton_RMP = UI.Button(rec_button_w + 3 * rec_button_i, rec_button_h, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_RMP.jpg', 'button_down_RMP.jpg'
            self.recButton_RMP.setIcon(filename_up, filename_down)
            self.recButton_RMP.set_recTime(self.rec_time)

            # func buttons list:
            func_button_h = 20
            func_button_w = 300 # this keeps same
            func_button_i = 150

            # save button to save database into a pkl file.
            #self.saveButton = UI.Button(func_button_w, func_button_h + 0 *func_button_i, self._game_surface)
            #filename_up, filename_down = 'button_up_SAVE.jpg', 'button_down_SAVE.jpg'
            #self.saveButton.setIcon(filename_up, filename_down)

            self.saveButton = UI.Button(func_button_w, func_button_h + 0 *func_button_i, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_SAVE.jpg', 'button_down_SAVE.jpg'
            self.saveButton.setIcon(filename_up, filename_down)

            # train button to train the hmm model by using the stored train data set
            self.trainButton = UI.Button(func_button_w, func_button_h + 1 *func_button_i, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_TRAIN.jpg', 'button_down_TRAIN.jpg'
            self.trainButton.setIcon(filename_up, filename_down)

            # test button to test user's input gesture
            self.testButton = UI.Button(func_button_w, func_button_h + 2 *func_button_i, self._game_surface, button_size)
            filename_up, filename_down = 'button_up_TEST.jpg', 'button_down_TEST.jpg'
            self.testButton.setIcon(filename_up, filename_down)
            self.testButton.set_recTime(self.rec_time)

            # info button for help instructions
            self.infoButton = UI.Button(func_button_w, func_button_h + 3 *func_button_i, self._game_surface, button_size, type = 'INFO_REC')
            filename_up, filename_down = 'button_up_INFO.jpg', 'button_down_INFO.jpg'
            self.infoButton.setIcon(filename_up, filename_down)
            
            # home button to go beck to intro screen # using the logo as icon
            self.homeButton = UI.Button(func_button_w, func_button_h + 4 *func_button_i, self._game_surface, button_size, type = 'INTRO')
            filename_up, filename_down = 'button_up_LOGO.jpg', 'button_down_LOGO.jpg'
            self.homeButton.setIcon(filename_up, filename_down)

            # create info boards
            self.infoBoard_REC = UI.Board('rec_infoBoard.jpg', rec_button_w + 4 * rec_button_i + 10, rec_button_h - 10, self._game_surface)
            (self.rec_timer_x, self.rec_timer_y) = self.infoBoard_REC.get_timerPos()
            self.timerBoard_REC = UI.Board('rec_timer.jpg', self.rec_timer_x, self.rec_timer_y, self._game_surface)
            self.endTimerBoard_REC = UI.Board('rec_timer_end.jpg', self.rec_timer_x, self.rec_timer_y, self._game_surface)
            self.show_rec_timer = False
            self.show_rec_timer_end = False
            # init a dict which contain ndarray to store training data set
            # this database dict has keys: 'MP', 'RK'
            self.database = dict()
            # this snapshot is for temporary storing of the whole 60 frames in one recording.
            self.snapshot = []
            # output is text to store the recognized gesture
            self.output = ''

        def initInfo_REC(self):
            self.info_learn = UI.Board('info_learn.jpg', 250, 50, self._game_surface)

            # home button to go beck to intro screen # using the logo as icon
            self.backButton_REC = UI.Button(1500, 50, self._game_surface, 150, type = 'REC')
            filename_up, filename_down = 'button_up_LOGO.jpg', 'button_down_LOGO.jpg'
            self.backButton_REC.setIcon(filename_up, filename_down)
        
        def initInfo_INTRO(self):
            self.info_intro = UI.Board('info_intro.jpg', 250, 50, self._game_surface)

            # home button to go beck to intro screen # using the logo as icon
            self.backButton_INTRO = UI.Button(1500, 80, self._game_surface, 200, type = 'INTRO')
            filename_up, filename_down = 'button_up_LOGO.jpg', 'button_down_LOGO.jpg'
            self.backButton_INTRO.setIcon(filename_up, filename_down)

        def initGame(self):
            self.numRound = 10
            self.count_GAME = 0
            self.output_GAME = 'Welcome!'

            self.startButton = UI.Button(300, 30, self._game_surface, 200)
            filename_up, filename_down = 'button_up_TEST.jpg', 'button_down_TEST.jpg'
            self.startButton.setIcon(filename_up, filename_down)
            self.startButton.set_recTime(self.rec_time)

            # home button to go beck to intro screen # using the logo as icon
            self.backButton_GAME = UI.Button(300, 230, self._game_surface, 200, type = 'INTRO')
            filename_up, filename_down = 'button_up_LOGO.jpg', 'button_down_LOGO.jpg'
            self.backButton_GAME.setIcon(filename_up, filename_down)

            self.infoBoard_GAME = UI.Board('rec_infoBoard.jpg', 1200, 30, self._game_surface)

            (self.game_timer_x, self.game_timer_y) = self.infoBoard_GAME.get_timerPos()
            self.timerBoard_GAME = UI.Board('rec_timer.jpg', self.game_timer_x, self.game_timer_y, self._game_surface)
            self.show_game_timer = False

            XTrain, numRec, numObs = KNNRecog.loadData('database.pkl')
            self.models = KNNRecog.train('database.pkl', 'model.pkl')
            self.numTypes = len(XTrain) # How many techniques can be recognized
            self.types = list(XTrain.keys())
            # types = ['RRK', 'LRK',...]
            self.total_time = self.numRound * self.rec_time

        initIntro(self)
        initRec(self)
        initInfo_REC(self)
        initInfo_INTRO(self)
        initGame(self)

        self.buttonList = [self.recButton_LMP, self.recButton_LRK, self.recButton_RMP, self.recButton_RRK,
                    self.backButton_REC, self.homeButton, self.saveButton, self.startButton, self.trainButton,
                    self.testButton, self.introButton_REC, self.backButton_GAME, self.backButton_INTRO, 
                    self.infoButton]

    def initGameData(self):
        self.startButton.get_unselected()
        self.show_game_timer = False
        self.snapshot = []
        self.numRound = 10
        self.count_GAME = 0
        self.output_GAME = 'Welcome!'
        XTrain, numRec, numObs = KNNRecog.loadData('database.pkl')
        self.models = KNNRecog.train('database.pkl', 'model.pkl')
        self.numTypes = len(XTrain) # How many techniques can be recognized
        self.types = list(XTrain.keys())
        # types = ['RRK', 'LRK',...]
        self.total_time = self.numRound * self.rec_time

    # helper to draw circle at hand position. 
    # input: the pos of hand, circle color
    def draw_joint_circle(self, joints, jointPoints, color, r, jointType):
        jointState = joints[jointType].TrackingState

        if jointState == PyKinectV2.TrackingState_NotTracked:
            return
        # we have tracked the hand! Yeah! Let's draw the circle!
        (x, y) = (jointPoints[jointType].x, jointPoints[jointType].y)
        try:
            x, y = int(x), int(y)
            pygame.draw.circle(self._frame_surface, color, (x, y), r)
            pygame.draw.circle(self._game_surface, color, (x, y), r)
        except: # need to catch it due to possible invalid positions (with inf)
            print('Fail to Draw circle!')

    # surface is the surface to draw the bone on
    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1, surface):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(surface, color, start, end, 30)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    # surface is the surface to draw the body on
    def draw_body(self, joints, jointPoints, color, surface):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft, surface);
    
        # Right Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_ThumbRight, surface);   
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight, surface);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_ThumbLeft, surface); 
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft, surface);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_FootRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight, surface);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_FootLeft, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft, surface);

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def drawUI_INTRO(self):
        #intro_surface = pygame.transform.scale(intro_surface, (w, h))
        #print('screen -->',(w, h), 'image -->', (intro_surface.get_width(), intro_surface.get_height()))
        self._game_surface.fill(BACKGROUND_COLOR_INTRO)
        self._game_surface.blit(self.intro_logo, (200, 200))

        self.introButton_REC.show()
        self.introButton_GAME.show()
        self.introButton_INFO.show()

    def drawIndicator(self, filename, joints, jointPoints, jointType, surface):
        jointState = joints[jointType].TrackingState
        if jointState == PyKinectV2.TrackingState_NotTracked:
            return
        # we have tracked the thumb (joint)! Yeah! Let's draw the indicator!
        (x, y) = (jointPoints[jointType].x, jointPoints[jointType].y)
        try:
            x, y = int(x), int(y)
            indicator = pygame.image.load(os.path.join('images', filename))
            indicator = pygame.transform.scale(indicator, (indicator.get_width()//2, indicator.get_height()//2))
            surface.blit(indicator, (x,  y))
        except: # need to catch it due to possible invalid positions (with inf)
            print('Fail to Draw indicator !')

    def drawUI_INFO_REC(self):
        self._game_surface.fill(BACKGROUND_COLOR)
        self.info_learn.show()
        self.backButton_REC.show()

    def drawUI_INFO_INTRO(self):
        self._game_surface.fill(BACKGROUND_COLOR)
        self.info_intro.show()
        self.backButton_INTRO.show()

    def drawUI_REC(self):
        # ui including buttons, instructions, etc. should draw on the self._game_surface
        # buttons or other widgets are created here and can be used other place.
        self._game_surface.fill(BACKGROUND_COLOR)
        self.recButton_RRK.show()
        self.recButton_LMP.show()
        self.recButton_LRK.show()
        self.recButton_RMP.show()
        self.homeButton.show()
        self.infoButton.show()
        self.saveButton.show()
        self.testButton.show()
        self.trainButton.show()
        self.infoBoard_REC.show()
        if self.show_rec_timer:
            self.timerBoard_REC.show()
        if self.show_rec_timer_end:
            self.endTimerBoard_REC.show()
        font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
        rec_info_text = font.render(self.info, True, (223, 223, 223))
        self.infoBoard_REC.print(rec_info_text)

    def drawUI_GAME(self):
        self._game_surface.fill(BACKGROUND_COLOR)
        self.startButton.show()
        self.backButton_GAME.show()

        self.infoBoard_GAME.show()

        if self.show_game_timer:
            self.timerBoard_GAME.show()

        font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
        game_info_text = font.render(self.output_GAME, True, (223, 223, 223))
        self.infoBoard_GAME.print(game_info_text)

    # input: button to make the record(Button()), movement type (str), e.g. 'RK', 'MP'.
    # input: recoding time; (x, y): the position of the controlling hand.
    def do_recording(self, body, joints, button, type, x, y):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover and not button.isSelected:
            button.get_selected()
        elif button.isSelected and button.recTime > 0:
            self.getOtherButtonLocked(button)
            self.show_rec_timer = True
            print('You can record the movement for 1s')
            # store the 1/60 frames result into self.database[type]
            # each snapshot (1/60 total frames) has 5*(x, y, z) of AnkleRigtht, AnkleLeft, wristight, wristleft, spinebase
            (x0, y0, z0) = (joints[PyKinectV2.JointType_SpineBase].Position.x,
                                joints[PyKinectV2.JointType_SpineBase].Position.y,
                                joints[PyKinectV2.JointType_SpineBase].Position.z)
            h = joints[PyKinectV2.JointType_Head].Position.y - joints[PyKinectV2.JointType_SpineBase].Position.y
            snapshot = [(joints[PyKinectV2.JointType_AnkleRight].Position.x - x0) / h,
                                (joints[PyKinectV2.JointType_AnkleRight].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_AnkleRight].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.x- x0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.x - x0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.x - x0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.z - z0)/h]
            # ['AR_x','AR_y','AR_z','AL_x', 'AL_y', 'AL_z', 'WR_x', 'WR_y', 'WR_z', 'WL_x', 'WL_y', 'WL_z', 'SB_x', 'SB_y', 'SB_z']
            self.snapshot += snapshot
            snapshot = None
            # blit info on infoBoard
            self.info = 'Recording a %s ...' % type
            #font = pygame.font.SysFont('comicsansms', 30, bold=True, italic=False)
            #rec_info_text = font.render(text, True, (223, 223, 223))
            #self.infoBoard_REC.print(rec_info_text)

            # blit count down on timerBoard
            text = '%d' % ((button.recTime + 20) //20)
            font = pygame.font.SysFont('comicsansms', 60, bold=True, italic=True)
            rec_timer_text = font.render(text, True, (221, 111, 114))
            self.timerBoard_REC.print(rec_timer_text, timer = True)
            
            # self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.recTime -= 1
            # at the end of the recording event, save snapshot in the database as one instance
            if (button.recTime == 0):
                # blit info on inforBoard
                self.info = 'Recording Finished!'
                #text = 'Recording a %s is FINISHED! /n Do another recording or save it!' % type
                #font = pygame.font.SysFont('comicsansms', 20, bold=True, italic=False)
                #rec_info_text = font.render(text, True, (223, 223, 223))
                #self.infoBoard_REC.print(rec_info_text)

                # stop showing the timer board
                self.show_rec_timer = False
                # record the times of this rec_button triggered in button.recNum
                button.recNum += 1
                # store the self.snapshot in database, then reset self.snapshot for other recording event
                if type not in self.database:
                    self.database[type] = [self.snapshot]
                else:
                    self.database[type] = self.database[type] + [self.snapshot]
                self.snapshot = []
        else:
            # reset button's recTime for countdowning next rec event
            button.recTime = self.rec_time
            button.get_unselected()
            button.get_unhover()
            self.getAllButtonUnlocked()

    def do_gaming(self, body, joints, button, x, y, data_filename, model_filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover and not button.isSelected:
            button.get_selected()
        elif button.isSelected and self.total_time > 0:
            # if button is selected
                # record 60 frames each round, total is self.total_time frames
                if self.total_time % self.rec_time == 0:
                    # randomly pick one trained technique
                    index = random.randint(0, self.numTypes - 1)
                    type = self.types[index]
                    code = KNNRecog.code_book_dict[type]
                    self.name = KNNRecog.code_book[code]

                    # print the name on the infoboard_game
                    self.output_GAME = self.name
                    #self.recTime = self.rec_time
                    self.show_rec_timer = True

                (x0, y0, z0) = (joints[PyKinectV2.JointType_SpineBase].Position.x,
                                    joints[PyKinectV2.JointType_SpineBase].Position.y,
                                    joints[PyKinectV2.JointType_SpineBase].Position.z)
                h = joints[PyKinectV2.JointType_Head].Position.y - joints[PyKinectV2.JointType_SpineBase].Position.y
                snapshot = [(joints[PyKinectV2.JointType_AnkleRight].Position.x - x0) / h,
                                    (joints[PyKinectV2.JointType_AnkleRight].Position.y - y0)/h,
                                    (joints[PyKinectV2.JointType_AnkleRight].Position.z - z0)/h,
                                    (joints[PyKinectV2.JointType_AnkleLeft].Position.x- x0)/h,
                                    (joints[PyKinectV2.JointType_AnkleLeft].Position.y - y0)/h,
                                    (joints[PyKinectV2.JointType_AnkleLeft].Position.z - z0)/h,
                                    (joints[PyKinectV2.JointType_WristRight].Position.x - x0)/h,
                                    (joints[PyKinectV2.JointType_WristRight].Position.y - y0)/h,
                                    (joints[PyKinectV2.JointType_WristRight].Position.z - z0)/h,
                                    (joints[PyKinectV2.JointType_WristLeft].Position.x - x0)/h,
                                    (joints[PyKinectV2.JointType_WristLeft].Position.y - y0)/h,
                                    (joints[PyKinectV2.JointType_WristLeft].Position.z - z0)/h]
                self.snapshot += snapshot
                snapshot = None

                # blit count down on timerBoard
                text = '%d' % ((self.total_time % self.rec_time + 20) //20)
                font = pygame.font.SysFont('comicsansms', 60, bold=True, italic=True)
                game_timer_text = font.render(text, True, (221, 111, 114))
                self.timerBoard_GAME.print(game_timer_text, timer = True)

                #self.recTime -= 1
                self.total_time -= 1

                    
                if self.total_time % self.rec_time == 0:
                    self.show_game_timer = False
                    XTest = np.array(self.snapshot)
                    self.snapshot = []
                    output = KNNRecog.predict_real_time(XTest, self.models)
                    print(self.total_time)
                    if output == self.name:
                        self.output_GAME = 'Correct!'
                        self.count_GAME += 1
                    else:
                        self.output_GAME = 'You can do Better!'
                    # pygame.time.wait(300)

                if self.total_time == 0:
                    # after rounds is complete
                    self.output_GAME = 'Score: %d / %d ' % (self.count_GAME, self.numRound)
        else:
            self.total_time = self.numRound * self.rec_time
            button.get_unselected()
            button.get_unhover()





    # check if the save button is triggered and do saving
    def do_saving(self, body, joints, button, x, y, filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover and not button.isSelected:
            button.get_selected()
        elif button.isSelected:
            print('Saving database in database.pkl', self.database)
            # save the self.database dictionary in a pickle file named 'database.pkl'
            #self.database['recNum'] = 'recNum_MP = %d, recNum_RK = %d' % (self.recButton_LMP.recNum, self.recButton_RRK.recNum)
            joblib.dump(self.database, filename)
            # blit a info text onto the game surface
            self.info = 'Save Successfully!'
            self.ready = True
            button.get_unselected()
        else:
            button.get_unhover()
    def getOtherButtonLocked(self, currButton):
        for button in self.buttonList:
            if (button is not currButton):
                button.getLocked()

    def getAllButtonUnlocked(self):
        for button in self.buttonList:
            button.getUnlocked()

    def do_choosing_mode(self, body, joints, joint_points, button, x, y, indicator = True):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
            print('hovered!')
            if indicator:
                self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover:
            button.get_selected()
            print('selected!')
            if indicator:
                self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)
        elif button.isSelected:
            self.database = dict()
            if button is self.backButton_GAME or button is self.introButton_GAME:
                self.initGameData()
            self.mode = button.type
            print('Selected Mode:  %s' % self.mode)
            button.get_unselected()
        else:
            button.get_unhover()
            if indicator:
                self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)

    def do_training(self, body, joints, button, x, y, data_filename, model_filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover and not button.isSelected:
            button.get_selected()
        elif button.isSelected:
            KNNRecog.train(data_filename, model_filename)
            self.info = 'Model is Ready!'
            #font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
            #rec_timer_text = font.render(text, True, (200, 0, 0))
            #self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.get_unselected()
        else:
            button.get_unhover()

    def do_testing(self, body, joints, button, x, y, model_filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover and not button.isSelected:
            button.get_selected()
        elif button.isSelected and button.recTime > 0:
            self.show_rec_timer = True
            # store the 1/60 frames result into self.database[type]
            # each snapshot (1/60 total frames) has 5*(x, y, z) of AnkleRigtht, AnkleLeft, wristight, wristleft, spinebase
            (x0, y0, z0) = (joints[PyKinectV2.JointType_SpineBase].Position.x,
                                joints[PyKinectV2.JointType_SpineBase].Position.y,
                                joints[PyKinectV2.JointType_SpineBase].Position.z)
            h = joints[PyKinectV2.JointType_Head].Position.y - joints[PyKinectV2.JointType_SpineBase].Position.y
            #h = 0.1
            snapshot = [(joints[PyKinectV2.JointType_AnkleRight].Position.x - x0) / h,
                                (joints[PyKinectV2.JointType_AnkleRight].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_AnkleRight].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.x- x0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_AnkleLeft].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.x - x0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_WristRight].Position.z - z0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.x - x0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.y - y0)/h,
                                (joints[PyKinectV2.JointType_WristLeft].Position.z - z0)/h]
            # ['AR_x','AR_y','AR_z','AL_x', 'AL_y', 'AL_z', 'WR_x', 'WR_y', 'WR_z', 'WL_x', 'WL_y', 'WL_z', 'SB_x', 'SB_y', 'SB_z']
            self.snapshot += snapshot
            snapshot = None
            # blit a text on the game surface
            self.info = 'Testing...'

            # blit count down on timerBoard
            text = '%d' % ((button.recTime + 20) //20)
            font = pygame.font.SysFont('comicsansms', 60, bold=True, italic=False)
            rec_timer_text = font.render(text, True, (0, 0, 0))
            self.timerBoard_REC.print(rec_timer_text, timer = True)

            button.recTime -= 1
            # at the end of the recording event, save snapshot in the database as one instance
            if (button.recTime == 0):
                self.show_rec_timer = False
                XTest = np.array(self.snapshot)
                output = KNNRecog.predict(XTest, model_filename)
                self.info = output
                self.snapshot = []

        else:
            # reset button's recTime for countdowning next rec event
            button.recTime = self.rec_time
            #self.output = ''
            button.get_unselected()
            button.get_unhover()

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            if self.mode == 'INTRO':
                self.drawUI_INTRO()
            elif self.mode == 'REC':
                self.drawUI_REC()
            elif self.mode == 'INFO_REC':
                self.drawUI_INFO_REC()
            elif self.mode == 'INFO_INTRO':
                self.drawUI_INFO_INTRO()
            elif self.mode == 'GAME':
                self.drawUI_GAME()

            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()
            
            # --- draw skeletons to _frame_surface
            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)

                    if self.mode == 'INTRO':
                        self.drawUI_INTRO()
                        # get the left thumb position and draw the indicator at that pos
                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)
                        
                        #for button in [self.introButton_REC, self.introButton_GAME, self.introButton_INFO]:
                        self.do_choosing_mode(body, joints, joint_points, self.introButton_REC, x, y)
                        self.do_choosing_mode(body, joints, joint_points, self.introButton_GAME, x, y)
                        self.do_choosing_mode(body, joints, joint_points, self.introButton_INFO, x, y)

                    elif self.mode == 'REC':   
                        # --- Game logic should go here
                        self.drawUI_REC()
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._frame_surface)
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._game_surface)

                        # check if rec button is triggered, do recording, or do nothing.
                        #x, y = joint_points[PyKinectV2.JointType_HandLeft].x, joint_points[PyKinectV2.JointType_HandLeft].y
                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        if not self.recButton_RRK.isLocked:
                            self.do_recording(body, joints, self.recButton_RRK, 'RRK', x, y)
                        if not self.recButton_LMP.isLocked:
                            self.do_recording(body, joints, self.recButton_LMP, 'LMP', x, y)
                        if not self.recButton_LRK.isLocked:
                            self.do_recording(body, joints, self.recButton_LRK, 'LRK', x, y)
                        if not self.recButton_RMP.isLocked:
                            self.do_recording(body, joints, self.recButton_RMP, 'RMP', x, y)

                        # check if save button is triggered, do save(), or do nothing.
                        #self.do_saving(body, joints, self.saveButton, x, y, 'database_train.pkl')
                        if not self.saveButton.isLocked:
                            self.do_saving(body, joints, self.saveButton, x, y, 'database.pkl')

                        # check if train button is triggered, do train(), or do nothing.
                        if not self.trainButton.isLocked:
                            self.do_training(body, joints, self.trainButton, x, y, data_filename = 'database.pkl', model_filename = 'model.pkl')

                        # check if test button is triggered, do test(), or do nothing.
                        if not self.testButton.isLocked:
                            self.do_testing(body, joints, self.testButton, x, y, model_filename ='model.pkl')

                        # check if home button is triggerd, do choosing mode, or do nothing
                        if not self.homeButton.isLocked:
                            self.do_choosing_mode(body, joints, joint_points, self.homeButton, x, y)
                        if not self.infoButton.isLocked:
                            self.do_choosing_mode(body, joints, joint_points, self.infoButton, x, y)

                    elif self.mode == 'INFO_REC':

                        self.drawUI_INFO_REC()

                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)

                        self.do_choosing_mode(body, joints, joint_points, self.backButton_REC, x, y)

                    elif self.mode == 'INFO_INTRO':

                        self.drawUI_INFO_INTRO()

                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        self.drawIndicator('indicator.jpg', joints, joint_points, PyKinectV2.JointType_ThumbLeft, self._game_surface)

                        self.do_choosing_mode(body, joints, joint_points, self.backButton_INTRO, x, y)
                    
                    elif self.mode == 'GAME':
                        self.drawUI_GAME()
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._frame_surface)
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._game_surface)

                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        

                        self.do_gaming(body, joints, self.startButton, x, y, data_filename = 'database.pkl', model_filename = 'model.pkl')
                        self.do_choosing_mode(body, joints, joint_points, self.backButton_GAME, x, y)

                # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
                # --- (screen size may be different from Kinect's color frame size) 
            
                target_height = int(self.h_to_w * (self._screen.get_width() // 5))
                game_target_height = int(self.h_to_w * (self._screen.get_width()))

                surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width()//5, target_height));
                game_surface_to_draw = pygame.transform.scale(self._game_surface, (self._screen.get_width(), game_target_height));
                # blit 1st game_surface to the screen, then blit the color frame showing surface to the screen

                
                self._screen.blit(game_surface_to_draw,
                                    (0, 0))
                #self._screen.blit(self._game_surface, (0, 0))
                game_surface_to_draw = None
                
                if self.mode == 'REC':
                    self._screen.blit(surface_to_draw, 
                                    (0 + self._screen.get_width() * 3//4, 
                                    0 + self._screen.get_height() - target_height))


                # --- Go ahead and update the screen with what we've drawn.
                pygame.display.flip()

                # --- Limit to 60 frames per second
                self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = KarateGo();
game.run();

