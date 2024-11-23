const fileInput = document.getElementById("fileInput");
const selectedImage = document.getElementById("selectedImage");
const labeledImage = document.getElementById("labeledImage");
const statsDiv = document.getElementById("stats");
const labeledContainer = document.getElementById("labeledContainer");
const errorDiv = document.getElementById("error");
const galleryImages = document.querySelectorAll(".gallery img");

console.log("Script loaded successfully!");

// Analyze an image (uploaded or from gallery)
const analyzeImage = async (imageSrc, isUploaded = false) => {
    try {
      let formData = new FormData();
  
      if (isUploaded) {
        // For uploaded images, directly append the file
        formData.append("file", imageSrc);
      } else {
        // For gallery images, fetch the image as a blob and append it
        const response = await fetch(imageSrc);
        if (!response.ok) {
          throw new Error(`Failed to fetch gallery image: ${response.status}`);
        }
        const blob = await response.blob();
        formData.append("file", blob, "gallery_image.jpg");
      }
  
      // Call the backend API
      const response = await fetch("http://127.0.0.1:8000/analyze-image", {
        method: "POST",
        body: formData,
      });
  
      if (response.ok) {
        const data = await response.json();
  
        // Display stats
        statsDiv.style.display = "block";
        statsDiv.innerHTML = `
          <h3>Analysis Results</h3>
          <p><strong>People Detected:</strong> ${data.people}</p>
          <p><strong>Chairs Detected:</strong> ${data.chairs}</p>
          <p><strong>Chairs Taken:</strong> ${data.chairs_taken}</p>
          <p><strong>Empty Chairs:</strong> ${data.empty_chairs}</p>
          <p><strong>Min Occupancy Estimate:</strong> ${data.min_occupancy.toFixed(2)}%</p>
          <p><strong>Max Occupancy Estimate:</strong> ${data.max_occupancy.toFixed(2)}%</p>
        `;
  
        // Display labeled image
        labeledImage.src = `data:image/jpeg;base64,${data.labeled_image_base64}`;
        labeledImage.style.display = "block";
        labeledContainer.style.display = "block";
      } else {
        throw new Error(`API Error: ${response.status}`);
      }
    } catch (error) {
      console.error(error);
      errorDiv.style.display = "block";
      errorDiv.textContent = `Error: ${error.message}`;
    }
  };
  

// Event listener for uploaded images
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      // Display selected image
      const reader = new FileReader();
      reader.onload = () => {
        selectedImage.src = reader.result;
        selectedImage.style.display = "block";
      };
      reader.readAsDataURL(file);
  
      // Analyze uploaded image
      analyzeImage(file, true); // Pass the file correctly to analyzeImage
    } else {
      console.error("No file selected for upload.");
      errorDiv.style.display = "block";
      errorDiv.textContent = "Please select a file to upload.";
    }
  });

// Event listener for gallery images
galleryImages.forEach((img) => {
    img.addEventListener("click", (event) => {
      const src = event.target.getAttribute("data-src");
  
      if (src) {
        console.log(`Selected gallery image: ${src}`); // Debugging log
  
        // Display selected image
        selectedImage.src = src;
        selectedImage.style.display = "block";
  
        // Analyze selected gallery image
        analyzeImage(src);
      } else {
        console.error("Gallery image source is missing.");
        errorDiv.style.display = "block";
        errorDiv.textContent = "Unable to process the selected gallery image.";
      }
    });
  });