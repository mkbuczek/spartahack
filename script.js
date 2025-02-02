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
ctx.lineWidth = 10;

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

    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");

    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    tempCtx.fillStyle= "white";
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    tempCtx.drawImage(canvas, 0, 0);

    const imageData = tempCanvas.toDataURL('image/png');

    console.log(imageData);

    fetch('tracenlearn.co/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Prediction result:', data)

        predictionText.innerHTML = data.prediction;
        predictionText.style.display = "flex";
        predictionImg.src = `images/${data.prediction}.jpg`;
        predictionImg.style.display = "block";

    })
    .catch(error => {
        console.error('Error', error);
    });

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Prediction result:', data)

        predictionText.innerHTML = data.prediction;
        predictionText.style.display = "flex";
        predictionImg.src = `images/${data.prediction}.jpg`;
        predictionImg.style.display = "block";

    })
    .catch(error => {
        console.error('Error', error);
    });


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