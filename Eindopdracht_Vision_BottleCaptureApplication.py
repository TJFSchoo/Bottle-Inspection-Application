#!/usr/bin/env python
# coding: utf-8

# In[8]:

import cv2
import numpy as np
import math
import time
import RPi.GPIO as GPIO

# Eindopdracht Inleiding Vision
# Tom Schoonbeek 2032257 & Djim Oomes 2122380

# LED setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(17,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)

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
    cv2.putText(frame_copy, "Navigeer met spatie door het proces heen.", (30, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    cv2.imshow("Bottle Inspection", frame_copy)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC
        print(">> ESC ingedrukt. Sluit applicatie.")
        print("<end>")
        print("")
        break
    elif k%256 == 32:
        # SPACE
        img_name = "fles_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print(">> {} vastgelegd en geschreven naar bestand".format(img_name))
        
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
        
        # Gaussian Blur
        # toegepast om de image te blurren en ruis te verminderen.
        gaussian_bottle = cv2.GaussianBlur(bottle_gray, (3, 3), 0)
        print('>> Gaussian Blur toegepast.')
        gaussian_bottle_screen_output = bottle_gray.copy()
        cv2.putText(gaussian_bottle_screen_output, '1. Gaussian Blur', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", gaussian_bottle_screen_output)
        cv2.waitKey(0)

        # Thresholding
        # toegepast voor een duidelijke splitsing van hoge en lage waarden van de image d.m.v. zwarte en witte pixels.
        
        # Gehele fles
        gaussian_bottle_copy = gaussian_bottle.copy()
        threshold_value = 80.5
        kernel = np.ones((5,5),np.uint8)
        (T, bottle_threshold_full) = cv2.threshold(gaussian_bottle_copy, threshold_value, 255, cv2.THRESH_BINARY_INV)
        # Closing tussendoor voor witte puntjes
        closing = cv2.morphologyEx(bottle_threshold_full, cv2.MORPH_CLOSE, kernel)
        bottle_threshold_full_screen_output = closing.copy()
        cv2.putText(bottle_threshold_full_screen_output, "fles", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(bottle_threshold_full_screen_output, '2. Thresholding', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_threshold_full_screen_output)
        cv2.waitKey(0)
        
        # Inhoud + dop (apart ivm transparantie fles)
        threshold_value = 198.5
        kernel = np.ones((5,5),np.uint8)
        (T, bottle_threshold) = cv2.threshold(gaussian_bottle, threshold_value, 255, cv2.THRESH_BINARY_INV)
        bottle_threshold_screen_output = bottle_threshold.copy()
        print('>> Thresholding toegepast.')
        cv2.putText(bottle_threshold_screen_output, "inhoud & dop", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(bottle_threshold_screen_output, '2. Thresholding', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_threshold_screen_output)
        cv2.waitKey(0)

        # Gradient
        # Zorgt voor een afbeelding met enkel edges, basisweergave contour fles, inhoud en dop

        # Gehele fles 
        contour_bottle_full = cv2.morphologyEx(closing,cv2.MORPH_GRADIENT,kernel)
        contour_bottle_full_screen_output = contour_bottle_full.copy()
        cv2.putText(contour_bottle_full_screen_output, "fles", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.putText(contour_bottle_full_screen_output, '3. Gradient', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", contour_bottle_full_screen_output)
        cv2.waitKey(0)

        # Inhoud + dop (apart ivm transparantie fles)
        contour_bottle = cv2.morphologyEx(bottle_threshold,cv2.MORPH_GRADIENT,kernel)
        print('>> Gradient toegepast.')
        contour_bottle_screen_output = contour_bottle.copy()
        cv2.putText(contour_bottle_screen_output, "inhoud & dop", (10, 410), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.putText(contour_bottle_screen_output, '3. Gradient', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", contour_bottle_screen_output)
        cv2.waitKey(0)

        # External contours (fles) berekenen aan de hand van thresholded image
        contours, hierarchy = cv2.findContours(contour_bottle_full, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print('>> Externe contouren berekend.')
        external_contours = np.zeros(contour_bottle_full.shape)
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1: # checkt laatste value uit elke row van de contours array op value die wel -1 is (dus NIET extern)
                cv2.drawContours(external_contours, contours, i, 255, -1)

        # Contour Fles
        i = len(contours) - 1
        bottle_clone = bottle_3_channel.copy()
        cv2.drawContours(bottle_clone, [contours[i]], -1, (255, 0, 0), 2)
        bottle_clone_screen_output = bottle_clone.copy()
        cv2.putText(bottle_clone_screen_output, '4. Contour fles', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(0)
        
        # Internal contours (vloeistof+flesdop) berekenen aan de hand van thresholded image
        contours, hierarchy = cv2.findContours(contour_bottle, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print('>> Interne contouren berekend.')
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
        cv2.putText(bottle_clone_screen_output, '5. Contour inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(0)
        
        # Contour Flesdop (derde-grootste contour)
        j = len(contours) - 3
        bottle_clone = bottle_3_channel.copy()
        cv2.drawContours(bottle_clone, [contours[j]], -1, (255, 0, 0), 2)
        bottle_clone_screen_output = bottle_clone.copy()
        cv2.putText(bottle_clone_screen_output, '6. Contour dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(0)
        
        ## Controle 80-90% flesinhoud
        print('>> Controle inhoud fles op 80-90% inhoud..')
        
        # Bounding box maken voor berekenen aspect ratio van inhoud fles en goed- en afkeuren
        bottle_clone = bottle_3_channel.copy()
        (x, y, w, h) = cv2.boundingRect(contours[i]) #i vanwege inhoud fles
        aspectRatio = w / float(h)
        
        # Vulpercentage automatisch berekenen
        fullBottleHeight = 300
        percentFinal = (h / fullBottleHeight) * 100
        percentFinalRounded = math.floor(percentFinal*10)/10
        print('>>>> Fles voor ' + str(percentFinalRounded) + '% gevuld.')
        
        # Validatie
        bottle_clone_screen_output = bottle_clone.copy()
        if percentFinal >= 80 and percentFinal <= 90.0:
            cv2.rectangle(bottle_clone_screen_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, "Correct", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, str(percentFinalRounded) + "%", (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(bottle_clone_screen_output, '7. Controle inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            GPIO.output(23,True)
            print('>>>> PASS!')
            bottle_contents_check = 1;
        else:
            cv2.rectangle(bottle_clone_screen_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, "Incorrect", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, str(percentFinalRounded) + "%", (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(bottle_clone_screen_output, '7. Controle inhoud', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            GPIO.output(22,True)
            print('>>>> FAIL!')
            bottle_contents_check = 0;
        cv2.imshow("Bottle Inspection", bottle_clone_screen_output)
        cv2.waitKey(0)
        
        ## Controle flesdop via Feature Matching (met FLANN-based matcher)
        print('>> Controle flesdop op aanwezigheid en correctheid..')
        
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
            print('>>>> Geen match gevonden.')
            print('>>>> FAIL!')
            cv2.putText(flann_matches_screen_output, '8. Controle dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Bottle Inspection", flann_matches_screen_output)
            bottle_cap_check = 0;
            GPIO.output(17,True)
            cv2.waitKey(0)
        else:
            cv2.putText(flann_matches_screen_output, "Match", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print('>>>> Match gevonden.')
            print('>>>> PASS!')
            cv2.putText(flann_matches_screen_output, '8. Controle dop', (10,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Bottle Inspection", flann_matches_screen_output)
            bottle_cap_check = 1;
            GPIO.output(18,True)
            cv2.waitKey(0)
            
        # Final keuring
        if bottle_contents_check == 1 and bottle_cap_check == 1:
            approved_img = cv2.imread('quality_approved.jpg',1)
            print('>> Alle controles PASSED! Fles goedgekeurd.')
            print('>> ')
            cv2.imshow("Bottle Inspection", approved_img)
            cv2.waitKey(0)
        else:
            rejected_img = cv2.imread('quality_rejected.jpg',1)
            print('>> Eén of meerdere controles FAILED! Fles afgekeurd.')
            print('>> ')
            cv2.imshow("Bottle Inspection", rejected_img)
            cv2.waitKey(0)
        
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




