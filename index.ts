import * as tf from '@tensorflow/tfjs';

// Safely gets an element using a query selector. Throws an error if element does not exist
function getElement(id: string): HTMLElement {
    let element = document.querySelector(id);
    if (!element)
        throw new Error(`Cannot find element "${id}"`);
    return element as HTMLElement;
}

// Make sure all necessary HTML elements are present and acquire them
function loadElements(): [HTMLElement, HTMLElement, HTMLElement, HTMLSelectElement, HTMLInputElement] {
    let output = getElement("#output")
    let grid = getElement(".grid")
    let clearBtn = getElement("#clear")
    let select = getElement('#select')
    let slider = getElement('#brush-slider')

    return [output, grid, clearBtn, select as HTMLSelectElement, slider as HTMLInputElement];
}

// Convert a 2d array into a TensorFlow Tensor
function getImageInputTensor(array: number[][]): tf.Tensor {
    return tf.tensor(array).reshape([1, 28, 28, 1]);
}

// Use the TensorFlow model to take a 2d array and output predictions
function showPredictions(model: tf.LayersModel, imgArray: number[][], output: HTMLElement) {
    // Get and shape input
    let inputTensor = getImageInputTensor(imgArray);

    // Get result Tensor
    let resultTensor = model.predict(inputTensor) as tf.Tensor;

    // Convert result Tensor to a JS array
    let resultArray = (resultTensor.arraySync() as number[][])[0];

    // Get the largest element
    let maxResult = Math.max(...resultArray);

    // Create a string that will be displayed to user
    let resultString = resultArray.map((n, i) => {
        let row: string = `${i} : ${n.toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}`

        if (n == maxResult) {
            row = `<b>${row}</b>`
        }
        return row;
    }).join('      \r\n');

    // Place the prediction string into the output HTML element
    output.innerHTML = resultString;
}

function createInputCanvas(grid: HTMLElement, updateFunc: () => void, brushSize: () => number): [HTMLCanvasElement, CanvasRenderingContext2D, HTMLCanvasElement, CanvasRenderingContext2D] {
    grid.innerHTML = "";
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d')!;

    let auxCanvas = document.createElement('canvas');
    let auxCtx = auxCanvas.getContext('2d')!;

    canvas.style.border = "1px solid black";
    canvas.style.imageRendering = "crisp-edges";

    canvas.width = 28;
    canvas.height = 28;
    canvas.style.width = "280px";
    canvas.style.height = "280px";

    auxCanvas.width = 280;
    auxCanvas.height = 280;

    grid.appendChild(canvas);

    // Set up some variables
    var drawing = false;
    var lastX: number;
    var lastY: number;

    // Add event listeners for mouse events
    canvas.addEventListener("mousedown", start);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stop);
    canvas.addEventListener("mouseout", stop);

    // Function to start drawing
    function start(e: MouseEvent) {
        drawing = true;
        lastX = e.clientX - canvas.offsetLeft;
        lastY = e.clientY - canvas.offsetTop;
    }

    // Function to draw
    function draw(e: MouseEvent) {
        if (!drawing) return;
        var x = e.clientX - canvas.offsetLeft;
        var y = e.clientY - canvas.offsetTop;

        auxCtx.beginPath();

        function lerp(start: number, end: number, progress: number) {
            return start + (end - start) * progress;
        }

        // Tried using squared norm to save computation. Made page crash 
        let frames = Math.floor(Math.hypot(x - lastX, y - lastY));

        let brush = brushSize();

        for (let i = 0; i < frames; i += 1) {
            auxCtx.ellipse(lerp(lastX, x, i / frames), lerp(lastY, y, i / frames), brush, brush, 0, 0, Math.PI * 2);
        }

        auxCtx.fill();

        ctx.clearRect(0, 0, 28, 28);
        ctx.drawImage(auxCanvas, 0, 0, 28, 28);

        lastX = x;
        lastY = y;

        updateFunc();
    }

    // Function to stop drawing
    function stop() {
        drawing = false;
    }

    return [canvas, ctx, auxCanvas, auxCtx];
}

function canvasImageArray(canvas: HTMLCanvasElement): number[][] {

    let ctx = canvas.getContext('2d')!;

    // Get the pixel data of the new canvas
    const pixelData = ctx.getImageData(0, 0, 28, 28).data;

    // Create a 28x28 array to hold the pixel values
    const pixelArray = new Array(28).fill(0).map(() => new Array(28).fill(0));

    // Loop through the pixel data and populate the array with grayscale values
    for (let i = 0; i < pixelData.length; i += 4) {
        const grayValue = pixelData[i + 3] / 255;
        const x = Math.floor((i / 4) % 28);
        const y = Math.floor(i / 4 / 28);
        pixelArray[y][x] = grayValue;
    }

    return pixelArray;
}

function createModelOptions(selection_element: HTMLSelectElement, options: string[], updateFunc: (selection: string) => void) {
    for (const option of options) {
        let option_element = document.createElement("option");
        option_element.text = option;
        selection_element.add(option_element);
    }

    selection_element.addEventListener("change", () => {
        updateFunc(selection_element.value);
    });
}

function getModelPath(model: string) {
    return './' + model + '/model.json';
}

// Main function
window.addEventListener('load', async () => {

    // Get HTML elements
    let [output, grid, clearBtn, select, slider] = loadElements();

    let modelStrings = ["miniModel", "mnistModel", "outputModel2", "outputModel3", "outputModel4"];

    // Load initial model from filesystem
    let model = await tf.loadLayersModel(getModelPath(modelStrings[0]));

    let brushSize = 10;

    // Create canvas element
    let [canvas, ctx, auxCanvas, auxCtx] = createInputCanvas(grid, () => {
        showPredictions(model, canvasImageArray(canvas), output);
    },
        () => {
            // This is ugly. TODO: refactor
            return brushSize;
        })

    // Create model selection options and callback
    createModelOptions(select, modelStrings, async (newModel) => {
        //TODO Signify that model is loading
        canvas.style.filter = "opacity(50%)";
        canvas.style.pointerEvents = "none";
        model = await tf.loadLayersModel(getModelPath(newModel));
        showPredictions(model, canvasImageArray(canvas), output);
        canvas.style.filter = "none";
        canvas.style.pointerEvents = "auto";
    });

    // Create initial predictions to get the model "warmed up"
    // (The first prediction of the model is significantly slower. I'm guessing that it doesn't fully load until the first prediction is made)
    showPredictions(model, canvasImageArray(canvas), output);


    // Clear button functionality
    clearBtn.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        auxCtx.clearRect(0, 0, auxCanvas.width, auxCanvas.height);
        showPredictions(model, canvasImageArray(canvas), output);
    })

    // Brush Slider
    slider.addEventListener('input', () => {
        brushSize = +slider.value;
    })


})