# Building an Interactive Window (GUI): -----> tkinter
# Predict_dig() ---> method takes the pic as input and activates the model to predict the digit

## Importing Libraries:
from asyncio import events
import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import * 
from PIL import Image, ImageDraw, ImageGrab     # Python Imaging Library (PIL)

# Load the saved Handwritten Recog model:
from keras.models import load_model

model = load_model('handwritten_digits.h5')
print('\n' + "Model loaded successfully....." + '\n')

'''
Tkinter ===> We'll be using: 1. Canvas - widget for drawing graphics (here, drawing the digits),
                             2. Button - to execute user actions
'''
# Function pool:
def clear_screen():
    global cv
    cv.delete('all')   # To delete every on the canvas

'''
In our application, we use the bind method of the Canvas widget to bind a 
activate_event() function to an event called <Button-1> and inside this callback function we bind another function, 
which is draw_lines() to an event called <B1-Motion>
'''
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)  ## executes draw_lines func after B1 pressed
    '''
    The mouse is moved, with mouse button 1 being held down and the current 
    position of the mouse pointer is provided in the x and y members of the event 
    object passed to the callback.
    '''
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', 
                    capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

def Recog_Digit():
    '''
    we will use ImageGrab module to copy the contents of the screen or the clipboard to a 
    PIL (Python Imaging Library) image memory. Basically, it takes a snapshot of the screen.
    After taking the screen snapshot we will use a crop method which takes four coordinates as input and 
    returns a rectangular region from the image (which is a snapshot in the case) and 
    then we will save the image under the given filename in png format.
    '''
    global img_num
    pred = []
    percentages = []
    filename = "{}.png".format(img_num)
    widget = cv

    # get the widgets coordinates:
    x = root.winfo_rootx() + widget.winfo_rootx()
    y = root.winfo_rooty() + widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Grab the image and crop acc. to the requirement and save it png format:
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename) 

    '''
    use OpenCV to find contours of an image that we saved earlier. 
    Contours can be explained simply as a curve joining all the continuous points 
    (along the boundary), having the same color or intensity. 
    It is a useful tool for object detection and recognition.

    Binary images --> better accuracy. So before finding contours, apply a threshold.
    '''

    # Read the image in color format:
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    # Convert the image to grayscale:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying the Otsu thresholding:
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # function helps in extracting the contours from the image:
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    '''
    We are creating bounding boxes for contours and extract ROI (Region Of Interest). 
    After extracting ROI, we will preprocess (resize, reshape and normalize)the image to support our model input. 
    Then execute the model.predict() method to recognize the handwritten digit 
    and draw the bounding box surrounding each digit present in the image with predicted value and percentage.
    '''
    for cnt in contours:
        # Get bounding box and extract ROI:
        x,y,w,h = cv2.boundingRect(cnt)
        # Create Rectangle:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)

        # Extract the image ROI:
        roi = th[y-top:y+h+bottom, x-left:x+w+right]

        # Resize ROI Image to 28x28 pixels:
        img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)

        # Reshaping the image to fit in our model input:
        img = img.reshape(1,28,28,1)

        # Normalizing the image to fit in our model input:
        img = img/255.0

        # Predict the result:
        pred = model.predict([img])[0]

        final_pred = np.argmax(pred)  ## np.argmax returns the indices of the maximum values

        data = str(final_pred) + " " + str(int(max(pred)*100)) + "%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)  ## To put text string on image
        
        # Showing the predicted result on New Window:
        cv2.imshow('image' ,image)
        cv2.waitKey(0)


# Creating a main window:
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognizer App --- A₹JUN")

# Initialize few variables:
lastx, lasty = None, None
img_num = 0

# Create Canvas for drawing:
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Tkinter provides a mechanism to deal with events and bindings:
cv.bind('<Button-1>', activate_event)                           ## define this function later

# Add Buttons and Labels:
btn_save = Button(text='Recognize Digit', command=Recog_Digit)    ## define Recog_Digit later
btn_save.grid(row=2, column=0, pady=1, padx=1)
btn_clear = Button(text='Clear Screen', command=clear_screen)     ## define clear_screen later
btn_clear.grid(row=2, column=1, pady=1, padx=1)

# mainloop() ---> used when the app is ready to run
root.mainloop()