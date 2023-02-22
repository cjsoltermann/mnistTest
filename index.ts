import * as tf from '@tensorflow/tfjs';

// Safely gets an element using a query selector. Throws an error if element does not exist
function getElement(id: string): HTMLElement {
    let element = document.querySelector(id);
    if (!element)
        throw new Error(`Cannot find element "${id}"`);
    return element as HTMLElement;
}

// Make sure all necessary HTML elements are present and acquire them
function loadElements(): [HTMLElement, HTMLElement, HTMLElement, HTMLSelectElement] {
    let output = getElement("#output")
    let grid = getElement(".grid")
    let clearBtn = getElement("#clear")
    let select = getElement('#select')

    return [output, grid, clearBtn, select as HTMLSelectElement];
}

// Convert the array of checkbox elements into an array of 0's and 1's
function checkboxesImageArray(checkboxes: HTMLInputElement[]) {
    let array = [];
    for (let i = 0; i < 28; i++) {
        let row = [];
        for (let j = 0; j < 28; j++) {
            let checkboxIndex = i * 28 + j;
            row.push(checkboxes[checkboxIndex].checked ? 1 : 0);
        }
        array.push(row);
    }

    return array;
}

function getImageInputTensor(array: number[][]): tf.Tensor {
    return tf.tensor(array).reshape([1, 28, 28, 1]);
}

// Use the TensorFlow model to take the array of checkbox elements and output predictions
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
        let row = `${i} : ${n.toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}`

        if (n == maxResult) {
            row = `<b>${row}</b>`
        }
        return row;
    }).join('      \r\n');

    // Place the prediction string into the output HTML element
    output.innerHTML = resultString;
}

function createInputCanvas(grid: HTMLElement, updateFunc: () => void): [HTMLCanvasElement, CanvasRenderingContext2D] {
    grid.innerHTML = "";
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d')!;

    canvas.style.border = "1px solid black";

    canvas.width = 280;
    canvas.height = 280;

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
        // ctx.beginPath();
        // ctx.moveTo(lastX, lastY);
        // ctx.lineTo(x, y);
        // ctx.strokeStyle = "black";
        // ctx.lineWidth = 8;
        // ctx.stroke();
        ctx.beginPath();

        function lerp(start: number, end: number, progress: number) {
            return start + (end - start) * progress;
        }

        // Tried using squared norm to save computation. Made page crash 
        let frames = Math.floor(Math.hypot(x - lastX, y - lastY));

        for (let i = 0; i < frames; i += 1) {
            ctx.ellipse(lerp(lastX, x, i / frames), lerp(lastY, y, i / frames), 10, 10, 0, 0, Math.PI * 2);
        }

        ctx.fill();


        lastX = x;
        lastY = y;

        updateFunc();
    }

    // Function to stop drawing
    function stop() {
        drawing = false;
    }

    return [canvas, ctx];
}

function canvasImageArray(canvas: HTMLCanvasElement): number[][] {

    // Create a new canvas element with a 28x28 size
    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;

    // Get the context of the new canvas
    const resizedCtx = resizedCanvas.getContext("2d")!;

    // Draw the original canvas onto the new canvas, scaled down to 28x28
    resizedCtx.drawImage(canvas, 0, 0, 28, 28);

    // Get the pixel data of the new canvas
    const pixelData = resizedCtx.getImageData(0, 0, 28, 28).data;

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

// Create the 28x28 array of checkboxes and setup mouse input
function createInputCheckboxes(grid: HTMLElement, updateFunc: () => void) {

    // Create the 28x28 checkboxes
    let checkboxes: HTMLInputElement[] = [];
    grid.innerHTML = "";
    for (let i = 0; i < 28 * 28; i++) {
        let checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkboxes.push(checkbox);
        grid.appendChild(checkbox);
    }

    // Events for keeping track of mouse state
    let isMouseDown = false;
    document.addEventListener('mousedown', function () {
        isMouseDown = true;
    });
    document.addEventListener('mouseup', function () {
        isMouseDown = false;
    });

    // Add a mouse event to each checkbox. On mouseover each checkbox will enable both itself and some neighbors.
    // updateFunc is the callback that will be called when any checkbox's state is changed
    for (let i = 0; i < checkboxes.length; i++) {
        checkboxes[i].addEventListener('mouseover', function () {
            if (isMouseDown) {
                this.checked = true;
                if (i + 1 < 28 * 28)
                    checkboxes[i + 1].checked = true;
                if (i - 28 >= 0)
                    checkboxes[i - 28].checked = true;
                updateFunc();
            }
        });
    }

    return checkboxes;
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

function allowCheckboxes(checkboxes: HTMLInputElement[], allow: boolean) {
    for (const checkbox of checkboxes) {
        checkbox.disabled = !allow;
    }
}

function getModelPath(model: string) {
    return './' + model + '/model.json';
}

// Main function
window.addEventListener('load', async () => {

    // Get HTML elements
    let [output, grid, clearBtn, select] = loadElements();

    let modelStrings = ["mnistModel", "outputModel2", "outputModel3", "outputModel4"];

    // Load initial model from filesystem
    let model = await tf.loadLayersModel(getModelPath(modelStrings[0]));

    // Create canvas element
    let [canvas, ctx] = createInputCanvas(grid, () => {
        showPredictions(model, canvasImageArray(canvas), output);
    })

    // Create model selection options and callback
    createModelOptions(select, modelStrings, async (newModel) => {
        //allowCheckboxes(checkboxes, false);
        model = await tf.loadLayersModel(getModelPath(newModel));
        //allowCheckboxes(checkboxes, true);
        showPredictions(model, canvasImageArray(canvas), output);
    });

    // Create initial predictions to get the model "warmed up"
    // (The first prediction of the model is significantly slower. I'm guessing that it doesn't fully load until the first prediction is made)
    showPredictions(model, canvasImageArray(canvas), output);


    // Clear button functionality
    clearBtn.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // showPredictions(model, checkboxesImageArray(checkboxes), output);
    })


})