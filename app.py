import streamlit as st
import pandas as pd
from vehicle_detector import (
    analyze_image,
    analyze_video,
    save_csv_report,
    generate_graph
)
from PIL import Image
import tempfile
import cv2

st.set_page_config(
    page_title="Vehicle Detection & Traffic Density Analysis",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection & Traffic Density Analysis System")

st.write(
    "Upload an image or video to detect vehicles, analyze traffic density, and generate reports."
)

uploaded_file = st.file_uploader(
    "Choose an Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type

    # IMAGE ANALYSIS
    if file_type.startswith("image"):

        st.subheader("Uploaded Image")

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        if st.button("Analyze Image"):

            data, annotated_image = analyze_image(image_path)

            st.subheader("Detection Result")
            st.image(
                annotated_image,
                caption="Detected Vehicles",
                use_container_width=True
            )

            st.subheader("Traffic Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Cars", data["Cars"])
            col2.metric("Buses", data["Buses"])
            col3.metric("Trucks", data["Trucks"])

            col1.metric(
                "Motorcycles",
                data["Motorcycles"]
            )

            col2.metric(
                "Total Vehicles",
                data["Total Vehicles"]
            )

            col3.metric(
                "Traffic Density",
                data["Traffic Density"]
            )

            csv_file = save_csv_report(
                data,
                "traffic_report.csv"
            )

            graph_file = generate_graph(
                data,
                "traffic_graph.png"
            )

            st.subheader("Vehicle Distribution")

            st.image(
                graph_file,
                use_container_width=True
            )

            with open(csv_file, "rb") as f:

                st.download_button(
                    "📥 Download CSV Report",
                    f,
                    file_name="traffic_report.csv",
                    mime="text/csv"
                )

    # VIDEO ANALYSIS
    elif file_type.startswith("video"):

        st.subheader("Uploaded Video")

        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(uploaded_file.read())

            video_path = tmp.name

        if st.button("Analyze Video"):

            with st.spinner(
                "Analyzing video... Please wait."
            ):

                data = analyze_video(video_path)

            st.success("Analysis Completed!")

            st.subheader("Traffic Statistics")

            col1, col2 = st.columns(2)

            col1.metric(
                "Frames Processed",
                data["Frames Processed"]
            )

            col2.metric(
                "Traffic Density",
                data["Traffic Density"]
            )

            st.subheader("Detection Statistics")

            st.write(pd.DataFrame([data]))

            csv_file = save_csv_report(
                data,
                "video_report.csv"
            )

            graph_file = generate_graph(
                data,
                "video_graph.png"
            )

            st.subheader("Vehicle Distribution")

            st.image(
                graph_file,
                use_container_width=True
            )

            with open(csv_file, "rb") as f:

                st.download_button(
                    "📥 Download CSV Report",
                    f,
                    file_name="video_report.csv",
                    mime="text/csv"
                )