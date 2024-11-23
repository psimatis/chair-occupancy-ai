import os
import requests

def call_analyze_api(api_url, directory_path):
    """
    Iterate through all images in the directory and call the analyze API.

    :param api_url: The URL of the analyze API endpoint.
    :param directory_path: Path to the directory containing images.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        # Construct the full path to the file
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is an image (by extension)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            print(f"Skipping non-image file: {file_name}")
            continue

        try:
            # Open the file and send it to the API
            with open(file_path, "rb") as image_file:
                response = requests.post(api_url, files={"file": image_file})

            # Check the response
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
    # Example usage
    api_url = "http://127.0.0.1:8000/analyze-and-label"  # Change to your API's URL
    directory_path = "../Datasets/User Test"
    call_analyze_api(api_url, directory_path)
