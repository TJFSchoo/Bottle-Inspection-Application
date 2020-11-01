#!/usr/bin/env python
# coding: utf-8

# In[8]:

import cv2
import numpy as np
import math
import time
import RPi.GPIO as GPIO
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import subprocess

# Eindopdracht Inleiding Vision
# Tom Schoonbeek 2032257 & Djim Oomes 2122380

RST = None     # on the PiOLED this pin isnt used
# Note the following are only used with SPI:
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0

# 128x32 display with hardware I2C:
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)


# Initialize library.
disp.begin()

# Clear display.
disp.clear()
disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0,0,width,height), outline=0, fill=0)

# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding
bottom = height-padding
# Move left to right keeping track of the current x position for drawing shapes.
# Load default font.
#font = ImageFont.load_default()
font = ImageFont.truetype('Righteous-Regular.ttf', 14)
big_font = ImageFont.truetype('Righteous-Regular.ttf', 20)

# LED setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(17,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
GPIO.setup(24,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.output(17,False)
GPIO.output(18,False)
GPIO.output(22,False)
GPIO.output(23,False)

# Webcam variabele
cam = cv2.VideoCapture(0)

# Camera Capture scherm
cv2.namedWindow("Bottle Inspection")

# Counter voor ondersteuning logica screen-capture maken
img_counter = 0

# Runtime loop
while True:
    ret, frame = cam.read()
    if not ret:
        print("Fout bij verbinden met camera.")
        break
    frame_copy = frame.copy()
    cv2.putText(frame_copy, 'Klaar voor scan', (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, "Druk op de knop", (30, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    cv2.imshow("Bottle Inspection", frame_copy)
    
    draw.rectangle((0,0,width,height), outline=0, fill=0)

    draw.text((0, top),       "Klaar voor scan ",  font=font, fill=255)

    # Display image.
    disp.image(image)
    disp.display()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC
        print(">>  ESC ingedrukt. Sluit applicatie.")
        print("<end>")
        print("")
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        disp.image(image)
        disp.display()
        break
    #elif k%256 == 32 | GPIO.input(24) == GPIO.HIGH:
    elif GPIO.input(24) == GPIO.HIGH:
        # SPACE
        img_name = "fles_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print(">>  {} vastgelegd en geschreven naar bestand".format(img_name))
        
        # Foto inladen die net is opgeslagen
        bottle_3_channel = cv2.imread(img_name)
        
        # Sample images als alternatief
        #bottle_3_channel = cv2.imread('fles_sample_correct.png')
        #bottle_3_channel = cv2.imread('fles_sample_high.png')
        #bottle_3_channel = cv2.imread('fles_sample_low.png')
        #bottle_3_channel = cv2.imread('fles_sample_nolid.png')
        
        # Afbeelding grijswaarde
        bottle_gray = cv2.split(bottle_3_channel)[0]
        
        # Check variabelen voor beide checks later in de code
        bottle_contents_check = 0;
        bottle_cap_check = 0;
        
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        draw.text((0, top),       "Verwerken..",  font=font, fill=255)
        disp.image(image)
        disp.display()
    
        # Gaussian Blur
        # toegepast om de image te blurren en ruis te verminderen.
        gaussian_bottle = cv2.GaussianBlur(bottle_gray, (3, 3), 0)
        print('>>  Gaussian Blur toegepast.')
        gaussian_bottle_screen_output = bottle_gray.copy()
        cv2.putText(gaussian_bottle_screen_output, 'Gaussian Blur', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", gaussian_bottle_screen_output)
        
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        draw.text((0, top),       "Verwerken.. ",  font=font, fill=255)
        draw.text((0, top+11),       ">> Blur ",  font=font, fill=255)
        disp.image(image)
        disp.display()
        cv2.waitKey(500)

        # Thresholding
        # toegepast voor een duidelijke splitsing van hoge en lage waarden van de image d.m.v. zwarte en witte pixels.
        
        # Gehele fles
        gaussian_bottle_copy = gaussian_bottle.copy()
        threshold_value = 90.5
        kernel = np.ones((5,5),np.uint8)
        (T, bottle_threshold_full) = cv2.threshold(gaussian_bottle_copy, threshold_value, 255, cv2.THRESH_BINARY_INV)
        # Closing tussendoor voor witte puntjes
        closing = cv2.morphologyEx(bottle_threshold_full, cv2.MORPH_CLOSE, kernel)
        bottle_threshold_full_screen_output = closing.copy()
        cv2.putText(bottle_threshold_full_screen_output, "fles", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(bottle_threshold_full_screen_output, 'Thresholding', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_threshold_full_screen_output)
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        draw.text((0, top),       "Verwerken.. ",  font=font, fill=255)
        draw.text((0, top+11),       ">> Threshold ",  font=font, fill=255)
        disp.image(image)
        disp.display()
        cv2.waitKey(500)
        
        # Inhoud + dop (apart ivm transparantie fles)
        threshold_value = 198.5
        kernel = np.ones((5,5),np.uint8)
        (T, bottle_threshold) = cv2.threshold(gaussian_bottle, threshold_value, 255, cv2.THRESH_BINARY_INV)
        bottle_threshold_screen_output = bottle_threshold.copy()
        print('>>  Thresholding toegepast.')
        cv2.putText(bottle_threshold_screen_output, "inhoud & dop", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(bottle_threshold_screen_output, 'Thresholding', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_threshold_screen_output)
        cv2.waitKey(500)
        
        # Gradient
        # Zorgt voor een afbeelding met enkel edges, basisweergave contour fles, inhoud en dop

        # Gehele fles 
        contour_bottle_full = cv2.morphologyEx(closing,cv2.MORPH_GRADIENT,kernel)
        contour_bottle_full_screen_output = contour_bottle_full.copy()
        cv2.putText(contour_bottle_full_screen_output, "fles", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.putText(contour_bottle_full_screen_output, 'Gradient', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", contour_bottle_full_screen_output)
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        draw.text((0, top),       "Verwerken.. ",  font=font, fill=255)
        draw.text((0, top+11),       ">> Gradient ",  font=font, fill=255)
        disp.image(image)
        disp.display()
        cv2.waitKey(500)

        # Inhoud + dop (apart ivm transparantie fles)
        contour_bottle = cv2.morphologyEx(bottle_threshold,cv2.MORPH_GRADIENT,kernel)
        print('>>  Gradient toegepast.')
        contour_bottle_screen_output = contour_bottle.copy()
        cv2.putText(contour_bottle_screen_output, "inhoud & dop", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.putText(contour_bottle_screen_output, 'Gradient', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", contour_bottle_screen_output)
        cv2.waitKey(500)
        
        # External contours (fles) berekenen aan de hand van thresholded image
        contours, hierarchy = cv2.findContours(contour_bottle_full, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print('>>  Externe contouren berekend.')
        external_contours = np.zeros(contour_bottle_full.shape)
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1: # checkt laatste value uit elke row van de contours array op value die wel -1 is (dus NIET extern)
                cv2.drawContours(external_contours, contours, i, 255, -1)

        # Contour Fles
        i = len(contours) - 1
        bottle_clone = bottle_3_channel.copy()
        cv2.drawContours(bottle_clone, [contours[i]], -1, (255, 0, 0), 2)
        bottle_clone_screen_output = bottle_clone.copy()
        cv2.putText(bottle_clone_screen_output, 'Contour fles', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        draw.rectangle((0,0,width,height), outline=0, fill=0)
        draw.text((0, top),       "Verwerken.. ",  font=font, fill=255)
        draw.text((0, top+11),       ">> Contouren ",  font=font, fill=255)
        disp.image(image)
        disp.display()
        cv2.waitKey(500)
        
        # Internal contours (vloeistof+flesdop) berekenen aan de hand van thresholded image
        contours, hierarchy = cv2.findContours(contour_bottle, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print('>>  Interne contouren berekend.')
        internal_contours = np.zeros(contour_bottle.shape)
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1: # checkt laatste value uit elke row van de contours array op value die niet -1 is (dus NIET extern)
                cv2.drawContours(internal_contours, contours, i, 255, -1)
                
        # Sorteren per grootte van area, pakt de eennagrootste waarde voor de flesinhoud (grootste waarde is de achtergrond) en de tweenagrootste voor flesdop
        areas = [cv2.contourArea(contour) for contour in contours]
        (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1])) 
        
        # Contour Inhoud fles (tweede-grootste contour)
        i = len(contours) - 2
        bottle_clone = bottle_3_channel.copy()
        cv2.drawContours(bottle_clone, [contours[i]], -1, (255, 0, 0), 2)
        bottle_clone_screen_output = bottle_clone.copy()
        cv2.putText(bottle_clone_screen_output, 'Contour inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(500)
        
        # Contour Flesdop (derde-grootste contour)
        j = len(contours) - 3
        bottle_clone = bottle_3_channel.copy()
        cv2.drawContours(bottle_clone, [contours[j]], -1, (255, 0, 0), 2)
        bottle_clone_screen_output = bottle_clone.copy()
        cv2.putText(bottle_clone_screen_output, 'Contour dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(500)
        
        ## Controle 80-90% flesinhoud
        print('>>  Controle inhoud fles op 80-90% inhoud..')
        
        # Bounding box maken voor berekenen aspect ratio van inhoud fles en goed- en afkeuren
        bottle_clone = bottle_3_channel.copy()
        (x, y, w, h) = cv2.boundingRect(contours[i]) #i vanwege inhoud fles
        aspectRatio = w / float(h)
        
        # Vulpercentage automatisch berekenen
        fullBottleHeight = 300
        percentFinal = (h / fullBottleHeight) * 100
        percentFinalRounded = math.floor(percentFinal*10)/10
        print('>> >>  Fles voor ' + str(percentFinalRounded) + '% gevuld.')
        
        # Validatie
        bottle_clone_screen_output = bottle_clone.copy()
        if percentFinal >= 80 and percentFinal <= 90.0:
            cv2.rectangle(bottle_clone_screen_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, "Correct", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, str(percentFinalRounded) + "%", (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, 'Controle inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            GPIO.output(23,True)
            print('>> >>  PASS!')
            bottle_contents_check = 1;
            
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Inhoud correct",  font=font, fill=255)
            draw.text((0, top+10),     "" + str(percentFinalRounded) + "% vol ", font=big_font, fill=255)
            disp.image(image)
            disp.display()
                    
        else:
            cv2.rectangle(bottle_clone_screen_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, "Incorrect", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, str(percentFinalRounded) + "%", (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, 'Controle inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            GPIO.output(22,True)
            print('>> >>  FAIL!')
            bottle_contents_check = 0;
            
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Inhoud incorrect",  font=font, fill=255)
            draw.text((0, top+10),     "" + str(percentFinalRounded) + "% vol ", font=big_font, fill=255)
            disp.image(image)
            disp.display()
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(500)
        
        ## Controle flesdop via Feature Matching (met FLANN-based matcher)
        print('>>  Controle flesdop op aanwezigheid en correctheid..')
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        
        # Source (afbeelding van correct-geplaatste flesdop) + target
        source = cv2.imread('bottle_cap_sample.png', 0)     
        target = bottle_3_channel.copy()   

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(source,None)
        kp2, des2 = sift.detectAndCompute(target,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []

        # Ratio test
        for i,(match1,match2) in enumerate(matches):
            if match1.distance < 0.3*match2.distance:
                good.append([match1])
        flann_matches = cv2.drawMatchesKnn(source,kp1,target,kp2,good,None,flags=0)

        # Resultaat flesdop controle
        flann_matches_screen_output = flann_matches.copy()
        if not good: # (als good-array NULL is)
            cv2.putText(flann_matches_screen_output, "Geen match", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            print('>> >>  Geen match gevonden.')
            print('>> >>  FAIL!')
            cv2.putText(flann_matches_screen_output, 'Controle dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Bottle Inspection", flann_matches_screen_output)
            bottle_cap_check = 0;
            
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Dop ",  font=font, fill=255)
            draw.text((0, top+10),       "Geen match ",  font=big_font, fill=255)
            disp.image(image)
            disp.display()
            
            GPIO.output(17,True)
        else:
            cv2.putText(flann_matches_screen_output, "Match", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print('>> >>  Match gevonden.')
            print('>> >>  PASS!')
            cv2.putText(flann_matches_screen_output, 'Controle dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Bottle Inspection", flann_matches_screen_output)
            bottle_cap_check = 1;
            
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Dop ",  font=font, fill=255)
            draw.text((0, top+10),       "Match ",  font=big_font, fill=255)
            disp.image(image)
            disp.display()
            
            GPIO.output(18,True)
        cv2.waitKey(2000)
            
        # Final keuring
        if bottle_contents_check == 1 and bottle_cap_check == 1:
            approved_img = cv2.imread('quality_approved.jpg',1)
            print('>>  Alle controles PASSED! Fles goedgekeurd.')
            print('>>  ')
            cv2.imshow("Bottle Inspection", approved_img)
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Fles ",  font=font, fill=255)
            
            
            draw.text((0, top+10),       "goedgekeurd ",  font=big_font, fill=255)
            
            disp.image(image)
            disp.display()
            
            count = 3
            while(count >= 0):
                GPIO.output(17,False)
                GPIO.output(18,False)
                GPIO.output(22,False)
                GPIO.output(23,True)
                cv2.waitKey(500)
                GPIO.output(17,False)
                GPIO.output(18,True)
                GPIO.output(22,False)
                GPIO.output(23,False)
                cv2.waitKey(500)
                count = count - 1
            GPIO.output(17,False)
            GPIO.output(18,False)
            GPIO.output(22,False)
            GPIO.output(23,False)
            cv2.waitKey(0)
        else:
            rejected_img = cv2.imread('quality_rejected.jpg',1)
            print('>>  Eén of meerdere controles FAILED! Fles afgekeurd.')
            print('>>  ')
            cv2.imshow("Bottle Inspection", rejected_img)
            image = Image.new('1', (width, height))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            disp.clear()
            draw.rectangle((0,0,width,height), outline=0, fill=0)
            draw.text((0, top),       "Fles ",  font=font, fill=255)
            draw.text((0, top+10),       "afgekeurd ",  font=big_font, fill=255)
            disp.image(image)
            disp.display()
            count = 3
            while(count >= 0):
                GPIO.output(17,False)
                GPIO.output(18,False)
                GPIO.output(22,True)
                GPIO.output(23,False)
                cv2.waitKey(500)
                GPIO.output(17,True)
                GPIO.output(18,False)
                GPIO.output(22,False)
                GPIO.output(23,False)
                cv2.waitKey(500)
                count = count - 1
            GPIO.output(17,False)
            GPIO.output(18,False)
            GPIO.output(22,False)
            GPIO.output(23,False)
            
        
        GPIO.output(17,False)
        GPIO.output(18,False)
        GPIO.output(22,False)
        GPIO.output(23,False)
        cv2.destroyAllWindows()
        
cam.release()
cv2.destroyAllWindows()


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




