import streamlit as st
from PIL import Image
import requests
import json
import cv2


def main():
    st.title("Bildverarbeitung mit Drag-and-Drop und Objekterkennung")
    st.markdown("---")
    tabs = ["Tab 1", "Tab 2"]
    active_tab = st.sidebar.radio("Tabs", tabs)

    if active_tab == "Tab 1":
        tab1()
    elif active_tab == "Tab 2":
        tab2()


def image_to_base64(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_base64


def process_image(image):
    response = requests.post(
        "https://api.clarifai.com/v2/models/general-image-recognition/versions/aa7f35c01e0642fda5cf400f543e7c40/outputs",
        headers={
            "Authorization": "Key 6598557ba29943b8b18c4f96cfe3a757",
            "Content-Type": "application/json",
        },
        data=json.dumps({"inputs": [{"data": {"image": {"base64": image}}}]})
    )
    if response.status_code == 200:
        data = response.json()
        predictions = data["outputs"][0]["data"]["concepts"]
        st.success("Objekterkennung erfolgreich!")

        for prediction in predictions:
            st.write(f"- {prediction['name']}: {prediction['value']:.2f}")
    else:
        error_message = response.json()["status"]["details"]
        st.error(f"Fehler bei der Objekterkennung: {error_message}")


def tab1():
    uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Bild verarbeiten"):
            with st.spinner("Bild wird verarbeitet..."):
                # Bild als Base64 kodieren
                image_base64 = image_to_base64(image)
                process_image(image_base64)


def tab2():
    st.title("Gesichtserkennung mit OpenCV und Streamlit")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    video_widget = st.empty()

    while True:
        success, img = cap.read()
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            st.write("Fehler beim Laden des Bildes.")
            break

        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        eyes = eyeCascade.detectMultiScale(img_gray)


        for (ex, ey, ew, eh) in eyes:
            img = cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)

        # Zeigen Sie das Video-Widget in Streamlit an
        video_widget.image(img, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
