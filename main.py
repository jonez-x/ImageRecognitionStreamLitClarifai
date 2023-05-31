import streamlit as st
from PIL import Image
import requests
import json

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

def main():
    st.title("Bildverarbeitung mit Drag-and-Drop und Objekterkennung")
    st.markdown("---")

    uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Bild verarbeiten"):
            with st.spinner("Bild wird verarbeitet..."):
                # Bild als Base64 kodieren
                image_base64 = image_to_base64(image)
                process_image(image_base64)

def image_to_base64(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_base64

if __name__ == "__main__":
    main()
