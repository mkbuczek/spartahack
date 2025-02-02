const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictionText = document.getElementById("predictionText");
const predictionImg = document.getElementById("predictionImg");

// Variable to track whether the user is drawing
let drawing = false;

// Set canvas size dynamically
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

// Adjust line width for smooth drawing
ctx.lineWidth = 10;
ctx.lineCap = "round";

// Helper function to get cursor position (mouse or touch)
function getCursorPosition(event) {
    const canvasRect = canvas.getBoundingClientRect();
    let x, y;

    if (event.touches) {
        x = event.touches[0].clientX - canvasRect.left;
        y = event.touches[0].clientY - canvasRect.top;
    } else {
        x = event.clientX - canvasRect.left;
        y = event.clientY - canvasRect.top;
    }

    return { x, y };
}

// Start drawing (Mouse + Touch)
function startDrawing(event) {
    event.preventDefault();
    drawing = true;
    ctx.beginPath();
    const { x, y } = getCursorPosition(event);
    ctx.moveTo(x, y);
}

// Draw on canvas (Mouse + Touch)
function draw(event) {
    if (!drawing) return;
    event.preventDefault();
    const { x, y } = getCursorPosition(event);
    ctx.lineTo(x, y);
    ctx.stroke();
}

// Stop drawing
function stopDrawing() {
    drawing = false;
}

// Mouse event listeners
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

// Touch event listeners (Mobile support)
canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);
canvas.addEventListener("touchcancel", stopDrawing);

// Submit canvas drawing
function submitCanvas() {
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");

    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.drawImage(canvas, 0, 0);

    const imageData = tempCanvas.toDataURL("image/png");

    console.log(imageData);

    // Send to tracenlearn.co API
    fetch("https://tracenlearn.co/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Prediction result:", data);
            predictionText.innerHTML = data.prediction;
            predictionText.style.display = "flex";
            predictionImg.src = `images/${data.prediction}.jpg`;
            predictionImg.style.display = "block";
        })
        .catch((error) => {
            console.error("Error", error);
        });

    // Send to local Flask backend
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Prediction result:", data);
            predictionText.innerHTML = data.prediction;
            predictionText.style.display = "flex";
            predictionImg.src = `images/${data.prediction}.jpg`;
            predictionImg.style.display = "block";
        })
        .catch((error) => {
            console.error("Error", error);
        });

    return imageData;
}

// Clear canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Convert canvas to Tensor for ML processing
function getCanvasTensor() {
    return tf.browser
        .fromPixels(canvas, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
}
