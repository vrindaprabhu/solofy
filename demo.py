# Imports
import streamlit as st
import ray
from streamlit.proto.Selectbox_pb2 import Selectbox
from PIL import Image


from src.bbox_detect import yolov3_main
from src.saliency_detect import u2net_main
from src.inpaint import WGAN_tester
from src.create_mask import create_mask_bgreplace, create_mask_inpaint
from utils.rescale_image import rescale_image
from config.solofy import *


# Session States
if "upload" not in st.session_state:
    st.session_state.upload = False

if "bboxes" not in st.session_state:
    st.session_state.bboxes = {}

if "preprocess" not in st.session_state:
    st.session_state.preprocess = False

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "inpaint" not in st.session_state:
    st.session_state.inpaint = False

if "inimage" not in st.session_state:
    st.session_state.inimage = None

# Utility function to rename the input file name to default name
def save_uploaded_file(uploadedfile):
    with open(INIMAGE, "wb") as f:
        f.write(uploadedfile.getbuffer())

    img = Image.open(INIMAGE)
    st.session_state.inimage = img.resize(RESIZE_TO, Image.ANTIALIAS)
    return st.success("Upload successful!")


# Title
st.title("SOLOFY!")
st.write("Removing strangers from your photos. :sunglasses:")


# Image Upload
st.sidebar.title("Upload Image")
image_file_buffer = st.sidebar.file_uploader("", type=["png", "jpeg", "jpg"])

if image_file_buffer is not None:
    # Hacky way to avoid inferencing on the same image twice
    if image_file_buffer.name != st.session_state.filename:
        st.session_state.filename = image_file_buffer.name
        # More info on the issue in below link:
        # https://github.com/streamlit/streamlit/issues/897#issuecomment-739268247
        save_uploaded_file(image_file_buffer)
        st.session_state.upload = True
        st.session_state.bboxes = {}
        st.session_state.preprocess = False
        st.session_state.inpaint = False

        image_file_buffer.seek(0)


# Every file upload pre-process the input. This includes getting the detection
# output and saliency output.
# The detected image will be then diplayed to help make the selection
if st.session_state.upload:
    ray.init()
    with st.spinner("preprocessing the input image..."):

        # detection and saliency
        ret_func1 = yolov3_main.remote()
        ret_func2 = u2net_main.remote()

        ret1, ret2 = ray.get([ret_func1, ret_func2])

        bboxes, img_shape = ret1
        original_h, original_w, _ = img_shape

        # utility functions
        create_mask_inpaint(list(bboxes.values()), img_shape)
        scaled_boxes, resized_detection = rescale_image(
            bboxes,
            original_h,
            original_w,
        )

        bboxes = dict(zip(bboxes.keys(), scaled_boxes.tolist()))

    st.session_state.bboxes = bboxes
    st.session_state.upload = False
    st.session_state.preprocess = True
    ray.shutdown()


# Once the pre-processing is completed, the background replacement kicks in.
# The selection helps in choosing the person who has to be in the photgraph
if st.session_state.preprocess:
    # display layouts
    st.header("Preprocessed Image")
    col1 = st.beta_columns(1)[0]
    col1.image(DETIMAGE, width=RESIZE_TO[0], use_column_width=True)

    st.header("Final Image")
    col2, col3 = st.beta_columns(2)
    col2.image(
        st.session_state.inimage,
        use_column_width=True,
        caption="Original Input Image",
    )

    # selection of the person
    st.sidebar.title("Foreground Selection")
    selection_values = ["<select>"] + list(st.session_state.bboxes.keys())
    selection = st.sidebar.selectbox("Display", selection_values)

    if selection != "<select>":
        # Algorithm runs only once at the initial selection and then the result is
        # saved. First run takes long.
        with st.spinner("generating the final output..."):
            if not st.session_state.inpaint:
                st.warning(
                    "Note: First time execution takes longer time than usual."
                )
                ray.init()
                ray.get(WGAN_tester.remote())
                st.session_state.inpaint = True
                ray.shutdown()

            # creating the final output and displaying the result
            create_mask_bgreplace(st.session_state.bboxes[selection])
            col3.image(OUTIMAGE, use_column_width=True, caption="Output Image")
