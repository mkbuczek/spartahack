const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictionText = document.getElementById("predictionText");
const predictionWord = document.getElementById("predictionWord");
const predictionImg = document.getElementById("predictionImg");

//variable to determine whether the user is currently
//drawing or not
let drawing = false;

canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

//adjust line width
ctx.lineWidth = 5;

//detects mouse being held down on canvas
canvas.addEventListener("mousedown", (event) => {
    drawing = true;
    ctx.beginPath();

    const canvasRect = canvas.getBoundingClientRect();
    const cursorX = event.clientX - canvasRect.left;
    const cursorY = event.clientY - canvasRect.top;    

    ctx.moveTo(cursorX, cursorY);
});

//detects the mouse moving while held down
//to begin drawing
canvas.addEventListener("mousemove", (event) => {
    if (drawing) {

        const canvasRect = canvas.getBoundingClientRect();
        const cursorX = event.clientX - canvasRect.left;
        const cursorY = event.clientY - canvasRect.top;    

        ctx.lineTo(cursorX, cursorY);
        ctx.stroke();
    }
});

//detects mouse not held down, no longer drawing
canvas.addEventListener("mouseup", () => {
    drawing = false;
});

function submitCanvas(){
    const imageData = canvas.toDataURL('image/png');

    //debug
    //predictionImg.src = imageData;

    predictionText.style.display = "flex";
    predictionImg.style.display = "block";

    return imageData;
}

//clears the canvas
function clearCanvas(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getCanvasTensor(){
    return tf.browser.fromPixels(canvas, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();
}