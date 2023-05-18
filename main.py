import numpy as np
from keras.models import load_model
import cv2

model = load_model('E://file linh tinh//All//hoc ky 2 - nam 3//He thong nhung//VS CODE//AI_FinalTest//HumanDetectionModel.h5') 

# Note: medium (1366x768), white background, people standing apart at a distance
img = cv2.imread('E://file linh tinh//All//hoc ky 2 - nam 3//He thong nhung//VS CODE//AI_FinalTest//images//BlackPink.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Blur photo
edges = cv2.Canny(blurred, 30, 100) # Edge detection in image

# Dilate to increase the dimension
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contour
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #EXTERNAL, TREE,...

# Split each object and store it in an array
subimages = []
for contour in contours:
    if cv2.contourArea(contour) > 5000:
        x,y,w,h = cv2.boundingRect(contour)
        subimg = img[y:y+h, x:x+w]
        subimages.append((subimg, (x, y, w, h)))

# Detect "Human" or "Non Human" using trained Model
human_count = 0
for subimg, (x, y, w, h) in subimages:
    resized = cv2.resize(subimg, (40, 40))
    resized = np.expand_dims(resized, axis=0)
    resized = resized.astype('float32') / 255
    pred = (model.predict(resized).argmax())

    class_name=['Not Human','Human']
    if(class_name[pred] == 'Human'):
        human_count += 1

    # Draw bounding box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display Human or nonHuman
    label = f'{class_name[pred]}'
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # else:
    #     # Draw bounding box
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #     # Display Human or nonHuman
    #     label = f'Not Human'
    #     cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the number of detected person
text = f"People Counting: {human_count}"
cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)  
# (20, 20) -> (x, y)
# 0.8 -> 0.8 times original font size
# thickness = 3

# Display the result image
cv2.imshow('People Detection', img)

cv2.waitKey(0)
cv2.destroyAllWindows()