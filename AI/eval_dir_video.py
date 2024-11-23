import os
import cv2
import numpy as np
import requests
import base64

def call_analyze_api_for_videos(api_url, directory_path, output_directory):
    """
    Process all videos in the directory, and output labeled videos.

    :param api_url: The URL of the analyze API endpoint.
    :param directory_path: Path to the directory containing videos.
    :param output_directory: Path to save the labeled videos.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if not file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"Skipping non-video file: {file_name}")
            continue

        print(f"Processing video: {file_name}")
        try:
            cap = cv2.VideoCapture(file_path)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            output_path = os.path.join(output_directory, f"labeled_{file_name}")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1

                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)

                with open(temp_frame_path, "rb") as image_file:
                    response = requests.post(api_url, files={"file": image_file})

                if response.status_code == 200:
                    # Decode JSON response and extract Base64 image
                    json_response = response.json()
                    labeled_image_base64 = json_response["labeled_image_base64"]

                    # Convert Base64 string to NumPy array
                    labeled_image_data = base64.b64decode(labeled_image_base64)
                    labeled_frame = cv2.imdecode(np.frombuffer(labeled_image_data, np.uint8), cv2.IMREAD_COLOR)

                    if labeled_frame is None:
                        print(f"Error: Failed to decode labeled frame for frame {frame_count}")
                        continue

                    if labeled_frame.shape[:2] != (height, width):
                        print(f"Error: Mismatched dimensions for frame {frame_count}. Skipping.")
                        continue

                    out.write(labeled_frame)
                else:
                    print(f"Error: API call failed for frame {frame_count} in {file_name}")
                    print(response.text)

            print(f"Labeled video saved to: {output_path}")
            print("-" * 30)

            cap.release()
            out.release()

        except Exception as e:
            print(f"Error processing video {file_name}: {e}")

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000/analyze-image"  # Use the Base64 JSON API
    directory_path = "../Datasets/User Test/videos"  # Path to your input videos
    output_directory = "../labeled_videos"  # Path to save the labeled videos
    call_analyze_api_for_videos(api_url, directory_path, output_directory)
