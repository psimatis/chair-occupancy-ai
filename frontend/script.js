const fileInput = document.getElementById("fileInput");
const labeledImage = document.getElementById("labeledImage");
const labeledContainer = document.getElementById("labeledContainer");
const statsDiv = document.getElementById("stats");
const geminiResultsDiv = document.getElementById("geminiResults");
const geminiText = document.getElementById("geminiText");
const errorDiv = document.getElementById("error");
const galleryImages = document.querySelectorAll(".gallery img");

const analyzeImage = async (imageSrc, isUploaded = false) => {
  try {
    let formData = new FormData();

    if (isUploaded) {
      formData.append("file", imageSrc);
    } else {
      const response = await fetch(imageSrc);
      const blob = await response.blob();
      formData.append("file", blob, "selected_image.jpg");
    }

    // Call the analyze API
    const analyzeResponse = await fetch("http://127.0.0.1:8000/analyze-image", {
      method: "POST",
      body: formData,
    });

    if (analyzeResponse.ok) {
      const data = await analyzeResponse.json();

      // Display labeled image
      labeledImage.src = `data:image/jpeg;base64,${data.labeled_image_base64}`;
      labeledImage.style.display = "block";
      labeledContainer.style.display = "block";

      // Display stats
      statsDiv.style.display = "block";
      statsDiv.innerHTML = `
        <h3>Analysis Results</h3>
        <p><strong>Persons Detected:</strong> ${data.people}</p>
        <p><strong>Chairs Detected:</strong> ${data.chairs}</p>
        <p><strong>Chairs Taken:</strong> ${data.chairs_taken}</p>
        <p><strong>Empty Chairs:</strong> ${data.empty_chairs}</p>
        <p><strong>Min Occupancy Estimate:</strong> ${data.min_occupancy.toFixed(2)}%</p>
        <p><strong>Max Occupancy Estimate:</strong> ${data.max_occupancy.toFixed(2)}%</p>
      `;

      // Call the Gemini API
      const geminiResponse = await fetch("http://127.0.0.1:8000/llm-analyze", {
        method: "POST",
        body: formData,
      });

      if (geminiResponse.ok) {
        const geminiData = await geminiResponse.json();
        geminiResultsDiv.style.display = "block";
        geminiText.textContent = geminiData.gemini_analysis;
        console.log(geminiText.textContent);
      } else {
        throw new Error(`Gemini API Error: ${geminiResponse.status}`);
      }
    } else {
      throw new Error(`Analyze API Error: ${analyzeResponse.status}`);
    }
  } catch (error) {
    console.error(error);
    errorDiv.style.display = "block";
    errorDiv.textContent = `Error: ${error.message}`;
  }
};

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    analyzeImage(file, true);
  }
});

galleryImages.forEach((img) => {
  img.addEventListener("click", (event) => {
    const src = event.target.getAttribute("data-src");
    analyzeImage(src);
  });
});
