from ctypes.wintypes import WIN32_FIND_DATAA
import pygame 
from random import randint
#遊戲初始化
pygame.init()
FPS = 60
COLOR = (250,250,250)
winx = 960
winy = 540
WINDOWS = (winx,winy)
img_x = 100
img_y = 100
img_xy = (img_x,img_y)


#遊戲初始化和視窗建立
screen = pygame.display.set_mode(WINDOWS,pygame.RESIZABLE)
clock = pygame.time.Clock()
pygame.display.set_caption("game")


image_surface = pygame.image.load("D:/睿恩的禍害日記/python/game/cat.png").convert()
image_new = pygame.transform.scale(image_surface,(50,50))
image_new = pygame.transform.rotate(image_new,0)
image_new1 = pygame.transform.rotate(image_new,0)
position = image_new.get_rect()

long_press = {'up': False, 'down': False, 'left': False, 'right': False};

wall = pygame.Surface((50,50))
wall.fill(color='pink')



running = True


while running:
    clock.tick(FPS) #一秒鐘執行六十次
    screen.fill(COLOR)
    site = [0,0]
    for event in pygame.event.get():
         if event.type == pygame.QUIT:
                 running = False
         if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: # 增加长按状态（按下方向键）
                long_press['up'] = True
            if event.key == pygame.K_DOWN:
                long_press['down'] = True
            if event.key == pygame.K_LEFT:
                long_press['left'] = True
            if event.key == pygame.K_RIGHT:
                long_press['right'] = True
         if event.type == pygame.KEYUP: # 取消长按状态（松开按键）
             if event.key == pygame.K_UP:
                long_press['up'] = False
             if event.key == pygame.K_DOWN:
                long_press['down'] = False
             if event.key == pygame.K_LEFT:
                long_press['left'] = False
             if event.key == pygame.K_RIGHT:
                long_press['right'] = False
         
         if long_press['up']:
                site[1] -= 100
                image_new1 = pygame.transform.rotate(image_new,90)
         if long_press['down']:
                site[1] += 100
                image_new1 = pygame.transform.rotate(image_new,270)
         if long_press['left']:
                site[0] -= 100
                image_new1 = pygame.transform.rotate(image_new,180)
         if long_press['right']:
                site[0] += 100
                image_new1 = pygame.transform.rotate(image_new,0)
    if position[0] <= 0: # 碰到左边缘
        position[0] = 1
    if position[1] <= 0: # 碰到上边缘
        position[1] = 1
    if position[0] >= int(winx):
        position[0] = int(winx)
    if position [1] >= int(winy):
        position[1] = int(winy)

                
    #更新遊戲
    position[0] += site[0]
    position[1] += site[1]
    screen.blit(wall, (100,100))
    screen.blit(image_new1,position) # 更新球的位置
    pygame.display.update()
    #畫面提示
    
    

pygame.quit()
 