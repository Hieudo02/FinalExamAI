import numpy as np
import time
import streamlit as st
from keras.models import load_model
import cv2
from PIL import Image

# Load model
model = load_model("PeopleCountingWeb\HumanDetectionModel.h5")

def main():
    # Title the web
    st.title(":blue[PEOPLE COUNTING]")

    # Background
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://i.imgur.com/50DuJG3.jpg");
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Create menu tab
    tabs = ["TOOL", "MORE"]
    active_tab = st.sidebar.radio("Select tab", tabs)

    # Display the content (depend on the following tab)
    if active_tab == "TOOL":
        tool_tab()
    elif active_tab == "MORE":
        more_tab()

def tool_tab():
    # Display the header and "Drag and drop" bar
    st.header("Detect and count the number of people in the picture")
    st.write(":warning: Note about the image:") 
    st.write(":heavy_check_mark: _SHOULD be in medium size (1366x768)._")
    st.write(":heavy_check_mark: _MUST have white background._")
    st.write(":heavy_check_mark: _The objects (human, things, etc) MUST stand apart at a distance._")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png"])
    
    # Process the file (if true)
    if uploaded_file is not None:
        successMessage = st.empty()
        successMessage = st.success("Upload image successfully!", icon="✅")
        # Create button
        click = st.button("Click here to count right now!")
        img_PIL = Image.open(uploaded_file)
        img_array = np.array(img_PIL) # np.array -> RGB
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if(click):
            successMessage.empty()
            # status element: spinner
            with st.spinner('Loading...'):
                time.sleep(2)
            st.success('Done!')

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
            
            # Display the number of detected person
            text = f"People Counting: {human_count}"
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)  
            # (20, 20) -> (x, y)
            # 0.8 -> 0.8 times original font size
            # thickness = 3

            st.image(img, channels="BGR", caption="Result")
def more_tab():
    st.header(":point_down: Check my Git for more")
    st.markdown("[Link is here](https://github.com/Hieudo02)")

if __name__ == "__main__":
    main()

