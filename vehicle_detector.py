from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


def calculate_density(total):

    if total < 10:
        return "Low"

    elif total < 25:
        return "Medium"

    else:
        return "High"


def count_vehicles(results):

    car_count = 0
    bus_count = 0
    truck_count = 0
    motorcycle_count = 0

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 2:
                car_count += 1

            elif cls == 3:
                motorcycle_count += 1

            elif cls == 5:
                bus_count += 1

            elif cls == 7:
                truck_count += 1

    total = (
        car_count
        + bus_count
        + truck_count
        + motorcycle_count
    )

    density = calculate_density(total)

    return {
        "Cars": car_count,
        "Buses": bus_count,
        "Trucks": truck_count,
        "Motorcycles": motorcycle_count,
        "Total Vehicles": total,
        "Traffic Density": density
    }


def analyze_image(image_path):

    results = model(image_path, conf=0.5)

    data = count_vehicles(results)

    annotated_image = results[0].plot()

    return data, annotated_image


def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    total_car = 0
    total_bus = 0
    total_truck = 0
    total_motorcycle = 0

    vehicle_per_frame = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=0.5)

        frame_data = count_vehicles(results)

        total_car += frame_data["Cars"]
        total_bus += frame_data["Buses"]
        total_truck += frame_data["Trucks"]
        total_motorcycle += frame_data["Motorcycles"]

        vehicle_per_frame.append(
            frame_data["Total Vehicles"]
        )

    cap.release()

    avg_vehicles = (
        sum(vehicle_per_frame) / len(vehicle_per_frame)
        if vehicle_per_frame
        else 0
    )

    max_vehicles = (
        max(vehicle_per_frame)
        if vehicle_per_frame
        else 0
    )

    density = calculate_density(avg_vehicles)

    return {
        "Frames Processed": frame_count,
        "Car Detections": total_car,
        "Bus Detections": total_bus,
        "Truck Detections": total_truck,
        "Motorcycle Detections": total_motorcycle,
        "Average Vehicles Per Frame": round(avg_vehicles, 2),
        "Maximum Vehicles In A Frame": max_vehicles,
        "Traffic Density": density
    }


def save_csv_report(data, filename):

    df = pd.DataFrame([data])

    df.to_csv(filename, index=False)

    return filename


def generate_graph(data, filename):

    labels = []
    counts = []

    for key, value in data.items():

        if isinstance(value, int):

            labels.append(key)
            counts.append(value)

    plt.figure(figsize=(8, 5))

    plt.bar(labels, counts)

    plt.title("Vehicle Analysis")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()

    return filename


if __name__ == "__main__":

    choice = input(
        "Enter image or video: "
    ).lower()

    if choice == "image":

        path = input(
            "Enter image path: "
        )

        data, annotated = analyze_image(path)

        print("\nIMAGE ANALYSIS\n")

        for k, v in data.items():
            print(k, ":", v)

        save_csv_report(
            data,
            "image_report.csv"
        )

        generate_graph(
            data,
            "image_graph.png"
        )

        print("\nReport Saved")

    elif choice == "video":

        path = input(
            "Enter video path: "
        )

        data = analyze_video(path)

        print("\nVIDEO ANALYSIS\n")

        for k, v in data.items():
            print(k, ":", v)

        save_csv_report(
            data,
            "video_report.csv"
        )

        generate_graph(
            data,
            "video_graph.png"
        )

        print("\nReport Saved")