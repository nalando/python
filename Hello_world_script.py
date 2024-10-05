import pyautogui
import random
import time
import pyperclip




_time = 1
for j in range(5):
    print(j)
    time.sleep(1)

for i in range(20):
    pyperclip.copy("笑死公連仔")
    pyautogui.hotkey('Ctrl','V')
    pyautogui.press('enter')
    pyperclip.copy("還在玩呀冰鳥")
    pyautogui.hotkey('Ctrl','V')
    pyautogui.press('enter')
