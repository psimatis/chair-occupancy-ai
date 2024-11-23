import os
import requests

def call_analyze_api(api_url, directory_path):
    """
    Iterate through all images in the directory and call the analyze API.

    :param api_url: The URL of the analyze API endpoint.
    :param directory_path: Path to the directory containing images.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            print(f"Skipping non-image file: {file_name}")
            continue

        try:
            with open(file_path, "rb") as image_file:
                response = requests.post(api_url, files={"file": image_file})

            if response.status_code == 200:
                result = response.json()
                print(result)
            else:
                print(f"Error: API call failed for {file_name} with status code {response.status_code}")
                print(response.text)

            print("-" * 30)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000/analyze-image"
    directory_path = "../Datasets/User Test/images"
    call_analyze_api(api_url, directory_path)
