from ctypes.wintypes import WIN32_FIND_DATAA
import pygame
from random import randint

# 遊戲初始化
pygame.init()
FPS = 60
COLOR = (250, 250, 250)
winx = 960
winy = 540
WINDOWS = (winx, winy)

# 遊戲初始化和視窗建立
screen = pygame.display.set_mode(WINDOWS, pygame.RESIZABLE)
clock = pygame.time.Clock()
pygame.display.set_caption("game")

# 載入圖片
image_surface = pygame.image.load("./test/cat.png").convert()
image_new = pygame.transform.scale(image_surface, (50, 50))
image_new1 = image_new  # 初始時不旋轉，保存圖片副本
position = image_new.get_rect()  # 取得圖片的矩形框座標
position.topleft = (winx // 2, winy // 2)  # 將圖片初始放在螢幕中央

# 長按狀態字典
long_press = {'up': False, 'down': False, 'left': False, 'right': False}

# 建立一個粉紅色的牆面
wall = pygame.Surface((50, 50))
wall.fill(color='pink')

running = True

while running:
    clock.tick(FPS)  # 一秒鐘執行六十次
    screen.fill(COLOR)
    site = [0, 0]  # 每次循環重置位移量

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # 處理鍵盤事件
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                long_press['up'] = True
            if event.key == pygame.K_DOWN:
                long_press['down'] = True
            if event.key == pygame.K_LEFT:
                long_press['left'] = True
            if event.key == pygame.K_RIGHT:
                long_press['right'] = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                long_press['up'] = False
            if event.key == pygame.K_DOWN:
                long_press['down'] = False
            if event.key == pygame.K_LEFT:
                long_press['left'] = False
            if event.key == pygame.K_RIGHT:
                long_press['right'] = False

    # 根據長按狀態更新位移
    if long_press['up']:
        site[1] -= 5  # 小幅移動
        image_new1 = pygame.transform.rotate(image_new, 90)
    if long_press['down']:
        site[1] += 5
        image_new1 = pygame.transform.rotate(image_new, 270)
    if long_press['left']:
        site[0] -= 5
        image_new1 = pygame.transform.rotate(image_new, 180)
    if long_press['right']:
        site[0] += 5
        image_new1 = pygame.transform.rotate(image_new, 0)

    # 更新位置
    position.x += site[0]
    position.y += site[1]

    # 邊緣檢查，避免超出螢幕邊界
    if position.left < 0:
        position.left = 0
    if position.top < 0:
        position.top = 0
    if position.right > winx:
        position.right = winx
    if position.bottom > winy:
        position.bottom = winy

    # 更新遊戲畫面
    screen.blit(wall, (100, 100))  # 顯示牆面
    screen.blit(image_new1, position)  # 更新圖片位置
    pygame.display.update()

# 結束遊戲
pygame.quit()
