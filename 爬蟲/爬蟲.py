from selenium import webdriver
import time
driver = webdriver.Chrome() 
driver.get("https://www.instagram.com/")#get to IG
time.sleep(20)
button = driver.find_element_by_class_name("_acan _acap _acas")
button.click()

