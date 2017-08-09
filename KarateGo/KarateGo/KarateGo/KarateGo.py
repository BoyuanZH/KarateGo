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
from hmmlearn import hmm

import os
import ctypes
import _ctypes
import pygame
import sys
import UI
import TechRecog

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# color for drawing the skeleton
SKELETON_COLOR = (40, 40, 40)
CIRCLE_COLOR = pygame.color.THECOLORS['orange']
CIRCLE_R = 30
BACKGROUND_COLOR = (232, 232, 232)

BACKGROUND_COLOR_INTRO = (182, 182, 182)


class DrawCircleAtHandGame(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # modes: INTRO, REC, GAME
        self.mode = 'INTRO'

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()

        #self.h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
        self.h_to_w = float(self._infoObject.current_h) / self._infoObject.current_w
        #self.h_to_w = float(self._kinect.body_index_frame_desc.Width) / self._kinect.body_index_frame_desc.Height
        screen_size = (self._infoObject.current_w >> 1, int((self._infoObject.current_w >> 1)*self.h_to_w))
        #screen_size = (self._kinect.body_index_frame_desc.Width >> 1, int((self._kinect.body_index_frame_desc.Width >> 1)*self.h_to_w))
        self._screen = pygame.display.set_mode(screen_size, 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("TP1 TechShow: Pygame and Kinect. DrawCirecleAtHand")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()


        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        # another buffer surface for drawing the game world (not showing the color frame)
        self._game_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        # init a small count down timer for recording movement.
        self.rec_time = 60
        # add a button named record
        # rec round kick
        self.recButton_RK = UI.Button(300, 250, self._game_surface)
        filename_up, filename_down = 'button_up_RK.jpg', 'button_down_RK.jpg'
        self.recButton_RK.setIcon(filename_up, filename_down)
        self.recButton_RK.set_recTime(self.rec_time)
        # rec front kick
        self.recButton_MP = UI.Button(300, 450, self._game_surface)
        filename_up, filename_down = 'button_up_MP.jpg', 'button_down_MP.jpg'
        self.recButton_MP.setIcon(filename_up, filename_down)
        self.recButton_MP.set_recTime(self.rec_time)

        # save button to save database into a pkl file.
        self.saveButton = UI.Button(300, 650, self._game_surface)
        filename_up, filename_down = 'button_up_SAVE.jpg', 'button_down_SAVE.jpg'
        self.saveButton.setIcon(filename_up, filename_down)

        # train button to train the hmm model by using the stored train data set
        self.trainButton = UI.Button(300, 50, self._game_surface)
        filename_up, filename_down = 'button_up_TRAIN.jpg', 'button_down_TRAIN.jpg'
        self.trainButton.setIcon(filename_up, filename_down)

        # test button to test user's input gesture
        self.testButton = UI.Button(500, 50, self._game_surface)
        filename_up, filename_down = 'button_up_TEST.jpg', 'button_down_TEST.jpg'
        self.testButton.setIcon(filename_up, filename_down)
        self.testButton.set_recTime(self.rec_time)
        # init a dict which contain ndarray to store training data set
        # this database dict has keys: 'MP', 'RK'
        self.database = dict()
        # this snapshot is for temporary storing of the whole 60 frames in one recording.
        self.snapshot = []
        # output is text to store the recognized gesture
        self.output = ''


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
            pygame.draw.line(surface, color, start, end, 20)
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
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, surface);   
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight, surface);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight, surface);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft, surface);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft, surface); 
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
        w, h = self._infoObject.current_w, self._infoObject.current_h
        w, h = self._screen.get_width(), self._screen.get_height()
        intro_surface = pygame.image.load(os.path.join('images','intro_bg.jpg'))
        #intro_surface = pygame.transform.scale(intro_surface, (w, h))
        tryButton = UI.Button(300, 650, intro_surface)
        filename_up, filename_down = 'button_up_SAVE.jpg', 'button_down_SAVE.jpg'
        tryButton.setIcon(filename_up, filename_down)
        tryButton.show()
        #print('screen -->',(w, h), 'image -->', (intro_surface.get_width(), intro_surface.get_height()))
        self._screen.fill(BACKGROUND_COLOR_INTRO)
        self._screen.blit(intro_surface, (60, 0))
        pygame.display.flip()

    def drawUI_REC(self):
        # ui including buttons, instructions, etc. should draw on the self._game_surface
        # buttons or other widgets are created here and can be used other place.
        self._game_surface.fill(BACKGROUND_COLOR)
        self.recButton_RK.show()
        self.recButton_MP.show()
        self.saveButton.show()
        self.testButton.show()
        self.trainButton.show()
        font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
        test_result_text = font.render(self.output, True, (200, 0, 0))
        self._game_surface.blit(test_result_text, (self._game_surface.get_width()//2, 150))
        print(self._screen.get_width(), self._screen.get_height())

    # input: button to make the record(Button()), movement type (str), e.g. 'RK', 'MP'.
    # input: recoding time; (x, y): the position of the controlling hand.
    def do_recording(self, body, joints, button, type, x, y):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover:
            button.get_selected()
        elif button.isSelected and button.recTime > 0:
            print('You can record the movement for 1s', joints[PyKinectV2.JointType_AnkleRight].Position.x,
                    joints[PyKinectV2.JointType_AnkleRight].Position.y,
                    joints[PyKinectV2.JointType_AnkleRight].Position.z)
            # store the 1/60 frames result into self.database[type]
            # each snapshot (1/60 total frames) has 5*(x, y, z) of AnkleRigtht, AnkleLeft, wristight, wristleft, spinebase
            snapshot = [joints[PyKinectV2.JointType_AnkleRight].Position.x,
                                joints[PyKinectV2.JointType_AnkleRight].Position.y,
                                joints[PyKinectV2.JointType_AnkleRight].Position.z,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.x,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.y,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.z,
                                joints[PyKinectV2.JointType_WristRight].Position.x,
                                joints[PyKinectV2.JointType_WristRight].Position.y,
                                joints[PyKinectV2.JointType_WristRight].Position.z,
                                joints[PyKinectV2.JointType_WristLeft].Position.x,
                                joints[PyKinectV2.JointType_WristLeft].Position.y,
                                joints[PyKinectV2.JointType_WristLeft].Position.z,
                                joints[PyKinectV2.JointType_SpineBase].Position.x,
                                joints[PyKinectV2.JointType_SpineBase].Position.y,
                                joints[PyKinectV2.JointType_SpineBase].Position.z]
            # ['AR_x','AR_y','AR_z','AL_x', 'AL_y', 'AL_z', 'WR_x', 'WR_y', 'WR_z', 'WL_x', 'WL_y', 'WL_z', 'SB_x', 'SB_y', 'SB_z']
            self.snapshot += snapshot
            snapshot = None
            # blit a text on the game surface
            text = 'Time Remained: %d' % button.recTime
            font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
            rec_timer_text = font.render(text, True, (200, 0, 0))
            self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.recTime -= 1
            # at the end of the recording event, save snapshot in the database as one instance
            if (button.recTime == 0):
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

    # check if the save button is triggered and do saving
    def do_saving(self, body, joints, button, x, y, filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover:
            button.get_selected()
        elif button.isSelected:
            print('Saving database in database.pkl', self.database)
            # save the self.database dictionary in a pickle file named 'database.pkl'
            self.database['recNum'] = 'recNum_MP = %d, recNum_RK = %d' % (self.recButton_MP.recNum, self.recButton_RK.recNum)
            joblib.dump(self.database, filename)
            # blit a info text onto the game surface
            text = 'Save Successfully'
            font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
            rec_timer_text = font.render(text, True, (200, 0, 0))
            self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.get_unselected()
        else:
            button.get_unhover()

    def do_training(self, body, joints, button, x, y, filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover:
            button.get_selected()
        elif button.isSelected:
            TechRecog.train(filename)
            text = 'Model is Ready!'
            font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
            rec_timer_text = font.render(text, True, (200, 0, 0))
            self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.get_unselected()
        else:
            button.get_unhover()

    def do_testing(self, body, joints, button, x, y, filename):
        if (body.hand_left_state == PyKinectV2.HandState_Open and button.rect.collidepoint(x, y)):
            button.get_hover()
        elif body.hand_left_state == PyKinectV2.HandState_Closed and button.isHover:
            button.get_selected()
        elif button.isSelected and button.recTime > 0:
            # store the 1/60 frames result into self.database[type]
            # each snapshot (1/60 total frames) has 5*(x, y, z) of AnkleRigtht, AnkleLeft, wristight, wristleft, spinebase
            snapshot = [joints[PyKinectV2.JointType_AnkleRight].Position.x,
                                joints[PyKinectV2.JointType_AnkleRight].Position.y,
                                joints[PyKinectV2.JointType_AnkleRight].Position.z,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.x,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.y,
                                joints[PyKinectV2.JointType_AnkleLeft].Position.z,
                                joints[PyKinectV2.JointType_WristRight].Position.x,
                                joints[PyKinectV2.JointType_WristRight].Position.y,
                                joints[PyKinectV2.JointType_WristRight].Position.z,
                                joints[PyKinectV2.JointType_WristLeft].Position.x,
                                joints[PyKinectV2.JointType_WristLeft].Position.y,
                                joints[PyKinectV2.JointType_WristLeft].Position.z,
                                joints[PyKinectV2.JointType_SpineBase].Position.x,
                                joints[PyKinectV2.JointType_SpineBase].Position.y,
                                joints[PyKinectV2.JointType_SpineBase].Position.z]
            # ['AR_x','AR_y','AR_z','AL_x', 'AL_y', 'AL_z', 'WR_x', 'WR_y', 'WR_z', 'WL_x', 'WL_y', 'WL_z', 'SB_x', 'SB_y', 'SB_z']
            self.snapshot += snapshot
            snapshot = None
            # blit a text on the game surface
            text = 'Time Remained: %d' % button.recTime
            font = pygame.font.SysFont('comicsansms', 40, bold=True, italic=False)
            rec_timer_text = font.render(text, True, (200, 0, 0))
            self._game_surface.blit(rec_timer_text, (self._game_surface.get_width()//2, 100))
            button.recTime -= 1
            # at the end of the recording event, save snapshot in the database as one instance
            if (button.recTime == 0):
                xTest = np.array(self.snapshot)
                output = TechRecog.test(xTest, filename)
                self.output = output
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
                # --- Game logic should go here
                self.drawUI_REC()
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
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._frame_surface)
                        #self._game_surface.fill(BACKGROUND_COLOR)
                        self.drawUI_REC()
                        self.draw_body(joints, joint_points, SKELETON_COLOR, self._game_surface)

                        # check if rec button is triggered, do recording, or do nothing.
                        #x, y = joint_points[PyKinectV2.JointType_HandLeft].x, joint_points[PyKinectV2.JointType_HandLeft].y
                        x, y = joint_points[PyKinectV2.JointType_ThumbLeft].x, joint_points[PyKinectV2.JointType_ThumbLeft].y
                        self.do_recording(body, joints, self.recButton_RK, 'RK', x, y)
                        self.do_recording(body, joints, self.recButton_MP, 'MP', x, y)

                        # check if save button is triggered, do save(), or do nothing.
                        #self.do_saving(body, joints, self.saveButton, x, y, 'database_train.pkl')
                        self.do_saving(body, joints, self.saveButton, x, y, 'database_test.pkl')

                        # check if train button is triggered, do train(), or do nothing.
                        self.do_training(body, joints, self.trainButton, x, y, 'database_test.pkl')

                        # check if test button is triggered, do test(), or do nothing.
                        self.do_testing(body, joints, self.testButton, x, y, 'model.pkl')


                    


                # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
                # --- (screen size may be different from Kinect's color frame size) 
            
                target_height = int(self.h_to_w * (self._screen.get_width() // 4))
                game_target_height = int(self.h_to_w * (self._screen.get_width()))\

                surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width()//4, target_height));
                game_surface_to_draw = pygame.transform.scale(self._game_surface, (self._screen.get_width(), game_target_height));
                # blit 1st game_surface to the screen, then blit the color frame showing surface to the screen
                self._screen.blit(game_surface_to_draw,
                                  (0, 0))
                game_surface_to_draw = None

                self._screen.blit(surface_to_draw, 
                                  (0 + self._screen.get_width() * 3//4, 
                                   0 + self._screen.get_height() - target_height))
            
                surface_to_draw = None

                # --- Go ahead and update the screen with what we've drawn.
                pygame.display.flip()

                # --- Limit to 60 frames per second
                self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = DrawCircleAtHandGame();
game.run();

