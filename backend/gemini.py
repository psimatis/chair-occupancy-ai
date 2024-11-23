import os
import PIL.Image
from dotenv import load_dotenv
import google.generativeai as genai

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Load environment variables from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

genai.configure(api_key=GEMINI_API_KEY)


def analyze_image(image_path):
    """
    Analyze an image using the Gemini API.
    :param image_path: Path to the image to be analyzed.
    :return: Analysis results from the Gemini API.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = PIL.Image.open(image_path)

        model = genai.GenerativeModel("gemini-1.5-pro")  # Initialize the generative model

        # Formulate the request
        prompt = "Keep it brief. Say if there many or few empty chairs. Suggest where to find an empty chair. Give demographics."
        result = model.generate_content([prompt, img], generation_config=genai.types.GenerationConfig(max_output_tokens=50))
        return result.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    image_path = "/home/panos/Desktop/chair-occupancy-ai/Datasets/User Test/images/Pasted image (2).png"
    print(analyze_image(image_path))
    
