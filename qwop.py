#!/usr/bin/env python

from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

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
		# options = Options()
		# options.add_argument('--headless')
		# options.add_argument('--disable-gpu')
		
		# options = Options()
		# options.add_argument('--disable-infobars')
		# options.binary_location = r'/usr/bin/chromium-browser'
		# options.add_argument('window-size=640x480')

		capa = DesiredCapabilities.CHROME;
		capa['chromeOptions'] = {
		  'binary': r'/usr/bin/chromium-browser',
		  'args': ["--disable-infobars"]
		}

		# self.browser = webdriver.Firefox()
		# self.browser = webdriver.Chrome(chrome_options=options)
		self.browser = webdriver.Chrome(desired_capabilities=capa)
		self.browser.set_window_size(640, 480)
		self.browser.get('http://www.foddy.net/Athletics.html?webgl=true')

		# self.canvas = WebDriverWait(self.browser, 20).until(
        	# EC.presence_of_element_located((By.XPATH, "//*[@id=\"window1\"")))


		self.browser.implicitly_wait(10)
		self.canvas = WebDriverWait(self.browser, 20).until(
        	EC.element_to_be_clickable((By.XPATH, "//canvas[@id='window1']")))

		# self.canvas = self.browser.find_element_by_id("window1")
		# self.canvas = self.browser.find_element_by_id("gameContent")
		# self.canvas = self.browser.find_element_by_xpath("//*[@id=\"window1\"]")

		time.sleep(1)

		print(self.canvas)

		location = self.canvas.location
		size = self.canvas.size

		self.actions = ActionChains(self.browser) 
		
		self.left = location['x']
		self.top = location['y']
		self.right = location['x'] + size['width']
		self.bottom = location['y'] + size['height']

		self.gameIntroTemplate = cv2.imread('intro.png', 0)
		self.gameLostTemplate = cv2.imread('lost.png', 0)

		self.grayImage = None
		self.image = None

		self.previousKey = None

	def startGame(self):
		if (self.previousKey):
			self.dispatchKeyEvent("keyUp", options[self.previousKey])
			self.previousKey = None

		if (self.isAtIntro()):
			# print("Starting game.")
			self.canvas.click()
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
		if (key != self.previousKey):
			self.holdKey(key)

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
		grey = cv2.cvtColor(self.image[75:-15, 100:-200], cv2.COLOR_BGR2GRAY)
		grey = cv2.resize(grey, (0,0), fx=0.10, fy=0.10)
		# print("size:", grey.size)
		# cv2.imshow("runningTrack", grey)
		# cv2.waitKey()
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

	def dispatchKeyEvent(self, name, options = {}):
		options["type"] = name
		body = json.dumps({'cmd': 'Input.dispatchKeyEvent', 'params': options})
		resource = "/session/%s/chromium/send_command" % self.browser.session_id
		url = self.browser.command_executor._url + resource
		self.browser.command_executor._request('POST', url, body)

	def holdKey(self, key):
		keyOptions = options[key]

		if (self.previousKey):
			self.dispatchKeyEvent("keyUp", options[self.previousKey])

		self.dispatchKeyEvent("rawKeyDown", keyOptions)
		self.dispatchKeyEvent("char", keyOptions)

		self.previousKey = key