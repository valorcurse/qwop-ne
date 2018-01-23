#!/usr/bin/env python

import pychrome

from PIL import Image
from base64 import b64decode

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from skimage.measure import compare_ssim
from tesserocr import PyTessBaseAPI

import cv2

from enum import Enum

import time
import io
import base64
import re
import os

import json

class Key(Enum):
	Q = 0
	W = 1
	O = 2
	P = 3

options = {
	Key.Q: { \
		"code": "KeyQ",
		"key": "q",
		"text": "q",
		"unmodifiedText": "q",
		"nativeVirtualKeyCode": ord("Q"),
		"windowsVirtualKeyCode": ord("Q")
	},
	Key.W: { \
		"code": "KeyW",
		"key": "w",
		"text": "w",
		"unmodifiedText": "w",
		"nativeVirtualKeyCode": ord("W"),
		"windowsVirtualKeyCode": ord("W")
	},
	Key.O: { \
		"code": "KeyO",
		"key": "o",
		"text": "o",
		"unmodifiedText": "o",
		"nativeVirtualKeyCode": ord("O"),
		"windowsVirtualKeyCode": ord("O")
	},
	Key.P: { \
		"code": "KeyP",
		"key": "p",
		"text": "p",
		"unmodifiedText": "p",
		"nativeVirtualKeyCode": ord("P"),
		"windowsVirtualKeyCode": ord("P")
	},
}


class QWOP:

	def __init__(self):
		self.browser = pychrome.Browser()

		tabs = self.browser.list_tab()

		if not tabs:
			self.tab = self.browser.new_tab()

		else:
			self.tab = tabs[0]

		# self.tab.debug = True

		self.tab.start()
		self.tab.Page.enable()
		self.tab.Page.navigate(url="http://www.foddy.net/Athletics.html?webgl=true")
		
		self.tab.call_method("Page.startScreencast", 
			format="png", quality=25, everyNthFrame=1,
			maxWidth=640, maxHeight=400)
		self.tab.Page.screencastFrame = self.processStreamFrame

		self.gameIntroTemplate = cv2.imread('intro-small.png', 0)
		# self.gameIntroTemplate = cv2.resize(cv2.imread('intro.png', 0), (0,0), fx=0.68, fy=0.68)
		self.gameLostTemplate = cv2.imread('lost.png', 0)

		self.grayImage = None
		self.image = None

		self.previousKey = None

		while self.image is None:
			time.sleep(1)

	def processStreamFrame(self, **kwargs):
		imgData = base64.b64decode(kwargs.get('data'))
		img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgData))), cv2.COLOR_BGR2RGB)

		self.image = img[62:-73, 50:-60]
		self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

		self.tab.Page.screencastFrameAck(sessionId=kwargs.get('sessionId')) 

	def startGame(self):
		if (self.previousKey):
			# self.dispatchKeyEvent("keyUp", options[self.previousKey])
			self.previousKey = None

		if (self.isAtIntro()):
			# print("Starting game.")
			self.click()
			# pass
		elif (self.isAtGameLost()):
			# print("Restarting game.")
			self.actions.key_down(Keys.SPACE).perform()
			self.actions.reset_actions()
			time.sleep(3)
		else:
			self.actions.key_down("r").perform()
			self.actions.reset_actions()
			time.sleep(3)


	def stop(self):
		# self.browser.quit()
		pass

	def isAtIntro(self):
		if (self.grayImage is None):
			return False 
		else:
			# print("---------------------------------")
			# for scale in np.linspace(0.2, 1.0, 20)[::-1]:
			# 	# resize the image according to the scale, and keep track
			# 	# of the ratio of the resizing
			# 	# resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			# 	# print(scale)
			# 	resized = cv2.resize(self.grayImage, (0,0), fx=scale, fy=scale)
			# 	# r = gray.shape[1] / float(resized.shape[1])
		 
			# 	# if the resized image is smaller than the template, then break
			# 	# from the loop
			# 	if resized.shape[0] < self.gameIntroTemplate.shape[0] or resized.shape[1] < self.gameIntroTemplate.shape[1]:
			# 		break

			# 	self.matchTemplate(resized, self.gameIntroTemplate)

			return self.matchTemplate(self.grayImage, self.gameIntroTemplate)
			# return False

	def isAtGameLost(self):
		if (self.grayImage is None):
			return False
		else:
			return self.matchTemplate(self.grayImage, self.gameLostTemplate)

	def isPlayable(self):
		return (not self.isAtIntro() and not self.isAtGameLost())

	def pressKey(self, key):
		if (key != self.previousKey):
			self.holdKey(key)

	# def grabImage(self):

	# 	# testImg = np.array(self.browser.execute_script(self.screenshotScript, self.canvas), dtype=np.uint8).reshape(self.height, self.width, 4)
	# 	testImg = np.array(self.browser.execute_script("return getCanvasPixels()"), dtype=np.uint8).reshape(self.height, self.width, 4)
	# 	testImg = cv2.flip(testImg, 0)
		
	# 	self.grayImage = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)

	def score(self):
		scoreImage = self.grayImage[20:50, 200:400]

		with PyTessBaseAPI() as tesseract:
			tesseract.SetImage(Image.fromarray(scoreImage))
			score = re.search('((-)?\d+(\.\d+)?).*', tesseract.GetUTF8Text())
			# print(tesseract.GetUTF8Text(), score)
			if (score == None):
				print("score is none:", tesseract.GetUTF8Text())
				return 0
				
			return max(0, int(float(score.group(1))*100))

	def runningTrack(self):
		resizedGrey = self.grayImage[50:-15, :]
		grey = cv2.resize(resizedGrey, (0,0), fx=0.50, fy=0.50)
		
		return grey

	def matchTemplate(self, image, template):
		res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
		# res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)

		w, h = template.shape[::-1]

		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		x1, x2 = top_left[0], bottom_right[0]
		y1, y2 = top_left[1], bottom_right[1]
		matchedRegion = image[y1:y2, x1:x2]
		resizedTemplate = cv2.resize(template, (matchedRegion.shape[1], matchedRegion.shape[0]))

		# print(image.shape, template.shape)
		print(compare_ssim(resizedTemplate, matchedRegion))
		# cv2.imshow("matchedRegion", matchedRegion)

		return (compare_ssim(resizedTemplate, matchedRegion) >= 0.9)

	def click(self):
		self.tab.Input.dispatchMouseEvent(
			type="mousePressed",
			button="left",
			x=640/2,
			y=400/2)

	def holdKey(self, key):
		option = options[key]
		self.tab.Input.dispatchKeyEvent(
            type="rawKeyDown",
            key=option["key"],
            code=option["code"],
            text=option["text"],
            unmodifiedText=option["unmodifiedText"],
            nativeVirtualKeyCode=option["nativeVirtualKeyCode"],
            windowsVirtualKeyCode=option["windowsVirtualKeyCode"])