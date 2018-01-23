#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import base64
import pychrome
import threading
import cv2
from PIL import Image
import io
import numpy as np

def close_all_tabs(browser):
    if len(browser.list_tab()) == 0:
        return

    for tab in browser.list_tab():
        try:
            tab.stop()
        except pychrome.RuntimeException:
            pass

        browser.close_tab(tab)

    time.sleep(1)
    assert len(browser.list_tab()) == 0


class Streamcast:
    def __init__(self):
        self.browser = pychrome.Browser()

        # close_all_tabs(browser)

        # tab = browser.new_tab()
        tabs = self.browser.list_tab()

        if not tabs:
            self.tab = self.browser.new_tab()

        else:
            self.tab = tabs[0]

        self.tab.debug = True

        self.tab.start()
        self.tab.Page.enable()
        self.tab.Page.navigate(url="http://www.foddy.net/Athletics.html?webgl=true")
        # self.tab.Page.navigate(url="http://chromium.org")
        
        # self.tab.call_method("Page.startScreencast", format="jpeg", quality=50, maxWidth=640, maxHeight=400, everyNthFrame=50)
        # self.tab.Page.screencastFrame = self.request_will_be_sent

        while True:
            # self.pressKey()
            # print("clicking Mouse")
            self.click()
            time.sleep(1)
        
        # browser.close_tab(tab)

    def request_will_be_sent(self, **kwargs):
        imgData = base64.b64decode(kwargs.get('data'))
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgData))), cv2.COLOR_BGR2RGB)
        cv2.imshow("QWOP", img)
        cv2.waitKey(32)

        self.tab.Page.screencastFrameAck(sessionId=kwargs.get('sessionId')) 
        # self.click()

    def pressKey(self):
        self.tab.Input.dispatchKeyEvent(
            type="rawKeyDown", 
            key="q", 
            code="KeyQ", 
            text="q", 
            unmodifiegText="q",
            nativeVirtualKeyCode=ord("Q"),
            windowsVirtualKeyCode=ord("Q"))

    def click(self):
        self.tab.Input.dispatchMouseEvent(
            type="mousePressed",
            button="left",
            x=0,
            y=0,
            timestamp=int(time.time()),
            clickCount=10)

        self.tab.Input.dispatchMouseEvent(
            type="mouseReleased",
            button="left",
            x=0,
            y=0,
            timestamp=int(time.time()),
            clickCount=1)



if __name__ == '__main__':
    Streamcast()