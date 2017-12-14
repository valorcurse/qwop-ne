#!/usr/bin/env python

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.keys import Keys

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

class Key(Enum):
	Q = 0
	W = 1
	O = 2
	P = 3

class QWOP:

	def __init__(self):

		self.browser = webdriver.Firefox()
		self.browser.set_window_size(640, 480)
		self.browser.get('http://www.foddy.net/Athletics.html?webgl=true')
		self.browser.implicitly_wait(10)
		# os.system("xdotool search 'QWOP - Mozilla Firefox' windowmove 2560 100")

		time.sleep(4)

		self.canvas = self.browser.find_element_by_id('window1')

		location = self.canvas.location
		size = self.canvas.size
		
		self.left = location['x']
		self.top = location['y']
		self.right = location['x'] + size['width']
		self.bottom = location['y'] + size['height']

		self.gameIntroTemplate = cv2.imread('intro.png', 0)
		self.gameLostTemplate = cv2.imread('lost.png', 0)

		self.grayImage = None
		self.image = None

	def startGame(self):
		if (self.isAtIntro()):
			# print("Starting game.")
			self.canvas.click()
			time.sleep(0.25)
		elif (self.isAtGameLost()):
			# print("Restarting game.")
			self.canvas.send_keys(Keys.SPACE)
			time.sleep(0.5)
		else:
			self.canvas.send_keys("r")
			time.sleep(1)


	def stop(self):
		self.browser.quit()

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
		# print("Pressing key " + key)
		actions = ActionChains(self.browser) 
		actions.key_down(key)
		actions.perform()

	def grabImage(self):
		# Convert to PIL
		base64Img = self.browser.get_screenshot_as_base64()
		img = io.BytesIO(base64.b64decode(base64Img))
		im = Image.open(img)

		# Crop image to element
		im = im.crop((self.left, self.top, self.right, self.bottom))
		
		# Converto to mat
		self.image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
		self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

	def score(self):
		scoreImage = self.image[20:50, 200:400]

		with PyTessBaseAPI() as tesseract:
			tesseract.SetImage(Image.fromarray(scoreImage))
			score = re.search('((-)?\d+(\.\d+)?).*', tesseract.GetUTF8Text())
			# print(tesseract.GetUTF8Text(), score)
			if (score == None):
				print("score is none:", tesseract.GetUTF8Text())
				return 0
				
			return max(0, int(float(score.group(1))*100))

	def runningTrack(self):
		# return self.image[75:-15, :]
		# return self.image[75:-15, 100:-275]
		grey = cv2.cvtColor(self.image[75:-15, 100:-275], cv2.COLOR_BGR2GRAY)
		grey = cv2.resize(grey, (0,0), fx=0.15, fy=0.15)
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

		return (compare_ssim(resizedTemplate, matchedRegion) >= 0.9)