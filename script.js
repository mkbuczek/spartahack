const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

//variable to determine whether the user is currently
//drawing or not
let drawing = false;

canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

//adjust line width
ctx.lineWidth = 5;

canvas.addEventListener("mousedown", (event) => {
    drawing = true;
    ctx.beginPath();

    const canvasRect = canvas.getBoundingClientRect();
    const cursorX = event.clientX - canvasRect.left;
    const cursorY = event.clientY - canvasRect.top;    

    ctx.moveTo(cursorX, cursorY);
});

canvas.addEventListener("mousemove", (event) => {
    if (drawing) {

        const canvasRect = canvas.getBoundingClientRect();
        const cursorX = event.clientX - canvasRect.left;
        const cursorY = event.clientY - canvasRect.top;    

        ctx.lineTo(cursorX, cursorY);
        ctx.stroke();
    }
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
});

