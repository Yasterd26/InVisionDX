document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.getElementById("modelSelect");
  const fileInput = document.getElementById("fileInput");
  const predictBtn = document.getElementById("predictBtn");
  const resultDiv = document.getElementById("result");

  // Get the current server URL
  const serverUrl = window.location.origin;

  // Endpoints
  const ENDPOINT_COVID = `${serverUrl}/predict_covid`;
  const ENDPOINT_PNEUMONIA = `${serverUrl}/predict_pneumonia`;
  const ENDPOINT_TB = `${serverUrl}/predict_tb`;
  const ENDPOINT_LUNG = `${serverUrl}/predict_lung`;
  const ENDPOINT_ALZ = `${serverUrl}/predict_alz`;

  predictBtn.addEventListener("click", async () => {
    resultDiv.innerText = "Processing...";

    if (!fileInput.files.length) {
      resultDiv.innerText = "Please select a file first.";
      return;
    }

    let endpoint;
    switch (modelSelect.value) {
      case "covid":
        endpoint = ENDPOINT_COVID;
        break;
      case "pneumonia":
        endpoint = ENDPOINT_PNEUMONIA;
        break;
      case "tb":
        endpoint = ENDPOINT_TB;
        break;
      case "lung":
        endpoint = ENDPOINT_LUNG;
        break;
      case "alz":
        endpoint = ENDPOINT_ALZ;
        break;
      default:
        resultDiv.innerText = "Unknown model selection.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      console.log(`Sending request to: ${endpoint}`);
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      // First check if the response is ok
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Try to parse the response as JSON
      let data;
      try {
        const text = await response.text();
        console.log("Raw response:", text);
        data = JSON.parse(text);
      } catch (e) {
        console.error("Failed to parse JSON:", e);
        throw new Error("Invalid JSON response from server");
      }

      if (data.error) {
        console.error(`Server error: ${data.error}`);
        resultDiv.innerHTML = `Error: ${data.error}`;
      } else {
        const { prediction, confidence, prediction_vector } = data;
        resultDiv.innerHTML = `
          <p><strong>Prediction:</strong> ${prediction}</p>
          <p><strong>Confidence:</strong> ${
            (confidence !== undefined) ? confidence.toFixed(4) : "N/A"
          }</p>
          ${
            prediction_vector
            ? `<p><strong>Prediction Vector:</strong> ${JSON.stringify(prediction_vector)}</p>`
            : ""
          }
        `;
      }
    } catch (error) {
      console.error("Error:", error);
      resultDiv.innerText = `Error: ${error.message}. Please check if the server is running and try again.`;
    }
  });
});
