import os
import cv2
import numpy as np
import requests

def call_analyze_api_for_videos(api_url, directory_path, output_directory):
    """
    Process all videos in the directory, send frames to the API, and save labeled videos.

    :param api_url: The URL of the analyze API endpoint.
    :param directory_path: Path to the directory containing videos.
    :param output_directory: Path to save the labeled videos.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in the directory
    for i, file_name in enumerate(os.listdir(directory_path)):
        if i == 0:
            continue
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is a video
        if not file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"Skipping non-video file: {file_name}")
            continue

        print(f"Processing video: {file_name}")
        try:
            cap = cv2.VideoCapture(file_path)

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Initialize video writer
            output_path = os.path.join(output_directory, f"labeled_{file_name}")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1

                # Save frame as a temporary image
                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)

                # Send frame to API
                with open(temp_frame_path, "rb") as image_file:
                    response = requests.post(api_url, files={"file": image_file})

                if response.status_code == 200:
                    # Decode labeled frame
                    labeled_frame = cv2.imdecode(
                        np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
                    )

                    # Check for valid labeled frame
                    if labeled_frame is None:
                        print(f"Error: Failed to decode labeled frame for frame {frame_count}")
                        continue

                    # Ensure dimensions match
                    if labeled_frame.shape[:2] != (height, width):
                        print(f"Error: Mismatched dimensions for frame {frame_count}. Skipping.")
                        continue

                    # Write labeled frame to video
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
    api_url = "http://127.0.0.1:8000/analyze-and-return-image"  # API URL
    directory_path = "../Datasets/User Test/videos"  # Input videos directory
    output_directory = "../labeled_videos"  # Output directory for labeled videos
    call_analyze_api_for_videos(api_url, directory_path, output_directory)
