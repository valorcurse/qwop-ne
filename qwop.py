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
	Q 		= 0
	W 		= 1
	O 		= 2
	P 		= 3
	SPACE 	= 4
	R 		= 5

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
	Key.SPACE: { \
		"code": "Space",
		"key": " ",
		"text": " ",
		"unmodifiedText": " ",
		"nativeVirtualKeyCode": ord(" "),
		"windowsVirtualKeyCode": ord(" ")
	},
	Key.R: { \
		"code": "KeyR",
		"key": "r",
		"text": "r",
		"unmodifiedText": "r",
		"nativeVirtualKeyCode": ord("R"),
		"windowsVirtualKeyCode": ord("R")
	}
}


class QWOP:

	def __init__(self, index):
		self.number = index

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
			format="png", quality=25, everyNthFrame=5,
			maxWidth=640, maxHeight=400)
		self.tab.Page.screencastFrame = self.processStreamFrame

		# self.gameIntroTemplate = cv2.resize(cv2.imread('intro.png', 0), (0,0), fx=0.68, fy=0.68)
		# self.gameIntroTemplate = cv2.imread('intro.png', 0)
		self.gameIntroTemplate = cv2.imread('intro-small.png', 0)
		# self.gameLostTemplate = cv2.imread('lost.png', 0)
		self.gameLostTemplate = cv2.imread('lost-small.png', 0)

		self.grayImage = None
		self.image = None
		self.scoreImage = None
		
		self.imageIsSimilar = False
		self.scoreIsSimilar = False

		self.previousKey = None

		# cv2.namedWindow(str(self.number))

		while self.image is None:
			# self.takeScreenshot()
			time.sleep(1)

	def showStream(self):
		cv2.imshow(str(self.number), self.image)
		cv2.waitKey(20)

	def takeScreenshot(self):
		print("taking screenshot")
		kwargs = self.tab.call_method("Page.captureScreenshot", 
			format="png", quality=25, fromSurface=True)

		imgData = base64.b64decode(kwargs.get('data'))
		img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgData))), cv2.COLOR_BGR2RGB)
		# img = np.array(Image.open(io.BytesIO(imgData)))

		# cv2.imshow("intro", img)
		# print(kwargs.get('data'))

		# pilImg = Image.open(io.BytesIO(imgData))
		# img = np.array(pilImg.getdata()).reshape(pilImg.size[1], pilImg.size[0], 4)

		self.image = img[100:-110, 80:-90]
		

		newImg = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
		if (not self.grayImage is None):
			self.imageIsSimilar = compare_ssim(self.grayImage, newImg) > 0.95
		
		self.grayImage = newImg

		cv2.imshow("intro", self.grayImage)
		cv2.waitKey(20)
		# plt.imshow(self.grayImage)
		# plt.show()

		newScoreImg = self.grayImage[15:30, 140:275]
		if (not self.scoreImage is None):
			self.scoreIsSimilar = compare_ssim(self.scoreImage, newScoreImg) > 0.95
		self.scoreImage = self.grayImage[15:30, 140:275]

		# print(kwargs.get('sessionId'))
		# self.tab.Page.screencastFrameAck(sessionId=kwargs.get('sessionId')) 

	def processStreamFrame(self, **kwargs):
		imgData = base64.b64decode(kwargs.get('data'))
		img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgData))), cv2.COLOR_BGR2RGB)

		self.image = img[62:-73, 50:-60]

		newImg = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
		if (not self.grayImage is None):
			self.imageIsSimilar = compare_ssim(self.grayImage, newImg) > 0.95
		
		
		self.grayImage = newImg

		# cv2.imshow("intro", self.grayImage)
		# cv2.waitKey(20)

		newScoreImg = self.grayImage[15:30, 140:275]
		if (not self.scoreImage is None):
			self.scoreIsSimilar = compare_ssim(self.scoreImage, newScoreImg) > 0.95
		self.scoreImage = self.grayImage[15:30, 140:275]

		# print(kwargs.get('sessionId'))
		self.tab.Page.screencastFrameAck(sessionId=kwargs.get('sessionId')) 

	def startGame(self):
		if (self.previousKey):
			self.sendKeyEvent(self.previousKey, "keyUp")
			self.previousKey = None

		if (self.isAtIntro()):
			# print("Starting game.")
			self.click()
		elif (self.isAtGameLost()):
			# print("Restarting game.")
			self.pressKey(Key.SPACE)
			# time.sleep(3)
		else:
			self.pressKey(Key.R)
			# time.sleep(3)


	def stop(self):
		# self.browser.quit()
		pass

	def getNumber(self):
		return self.number

	def getImage(self):
		return self.grayImage

	def isImageSimilar(self):
		return self.imageIsSimilar

	def isAtIntro(self):
		if (self.grayImage is None):
			return False 
		else:
			return self.matchTemplate(self.grayImage, self.gameIntroTemplate)

	def isAtGameLost(self):
		if (self.grayImage is None):
			return False
		else:
			return self.matchTemplate(self.grayImage, self.gameLostTemplate)

	def isPlayable(self):
		return (not self.isAtIntro() and not self.isAtGameLost())

	def pressKey(self, key):
		self.sendKeyEvent(key, "keyDown")
		self.sendKeyEvent(key, "keyUp")

	def holdKey(self, key):
		
		if (key != self.previousKey):
			if (self.previousKey):
				self.sendKeyEvent(self.previousKey, "keyUp")
			
			self.sendKeyEvent(key, "keyDown")
			self.previousKey = key

	def isScoreSimilar(self):
		return self.scoreIsSimilar

	def score(self):
		with PyTessBaseAPI() as tesseract:
			tesseract.SetImage(Image.fromarray(self.scoreImage))
			score = re.search('((-)?\d+(\.\d+)?).*', tesseract.GetUTF8Text())
			# print(tesseract.GetUTF8Text(), score)
			if (score == None):
				# print("score is none:", tesseract.GetUTF8Text())
				return 0
				
			return max(0, int(float(score.group(1))*100))

	def runningTrack(self):
		resizedGrey = self.grayImage[50:-15, :]
		grey = cv2.resize(resizedGrey, (0,0), fx=0.15, fy=0.15)
		# print(grey.size)
		return grey

	def matchTemplate(self, image, template):
		res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

		w, h = template.shape[::-1]

		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		x1, x2 = top_left[0], bottom_right[0]
		y1, y2 = top_left[1], bottom_right[1]
		matchedRegion = image[y1:y2, x1:x2]
		resizedTemplate = cv2.resize(template, (matchedRegion.shape[1], matchedRegion.shape[0]))

		# print(compare_ssim(resizedTemplate, matchedRegion))
		return (compare_ssim(resizedTemplate, matchedRegion) >= 0.9)

	def click(self):
		self.tab.Input.dispatchMouseEvent(
			type="mousePressed",
			button="left",
			x=640/2,
			y=400/2)

	def sendKeyEvent(self, key, mode):
		# print(key)
		# if (key != self.previousKey):
		option = options[key]
		self.tab.Input.dispatchKeyEvent(
            type=mode,
            key=option["key"],
            code=option["code"],
            text=option["text"],
            unmodifiedText=option["unmodifiedText"],
            nativeVirtualKeyCode=option["nativeVirtualKeyCode"],
            windowsVirtualKeyCode=option["windowsVirtualKeyCode"])

		# self.previousKey = key