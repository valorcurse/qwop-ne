#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import base64
import pychrome
import threading
import cv2

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


def main():
    browser = pychrome.Browser()

    # close_all_tabs(browser)

    # tab = browser.new_tab()
    tabs = browser.list_tab()

    if not tabs:
        tab = browser.new_tab()

    else:
        tab = tabs[0]

    tab.debug = True

    tab.start()
    tab.Page.enable()
    # tab.Page.navigate(url="http://www.foddy.net/Athletics.html")
    tab.Page.navigate(url="http://chromium.org")
    
    screen_lock = threading.Lock()
    with screen_lock:
        try:
            # viewport = tab.Viewport
            # viewport.x = 5
            # viewport = tab.viewport(x=50, y=50, width=200, height=200, scale=1)
            # [5, 5, 200, 200, 1]
            tab.Page.getLayoutMetrics()
            data = tab.Page.captureScreenshot(params="clip={viewport:{x: 10, y: 10, width: 200, height: 200}}")
            # data = tab.Page.captureScreenshot()
            # tab.call_method("Page.captureScreenshot", clip="{}")
            # print(data)

            with open("%s.png" % time.time(), "wb") as fd:
                fd.write(base64.b64decode(data['data']))
        finally:
            # tab.stop()
            pass
    
    # browser.close_tab(tab)



if __name__ == '__main__':
    main()