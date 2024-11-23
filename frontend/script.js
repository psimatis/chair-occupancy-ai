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
        <p><strong>${data.people}</strong> people detected</p>
        <p><strong>${data.chairs_taken}</strong>  out of chairs  <strong>${data.chairs}</strong> are taken </p>
        <p><strong>Occupancy estimate:</strong> ${data.min_occupancy.toFixed(2)}% - ${data.max_occupancy.toFixed(2)}%</p>
      `;

      // Call the Gemini API
      geminiText.innerHTML = "<em>AI is thinking...</em>";
      geminiResultsDiv.style.display = "block";

      const geminiResponse = await fetch("http://127.0.0.1:8000/llm-analyze", {
        method: "POST",
        body: formData,
      });

      if (geminiResponse.ok) {
        const geminiData = await geminiResponse.json();
        geminiText.innerHTML = geminiData.gemini_analysis;
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
