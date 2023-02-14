import * as tf from '@tensorflow/tfjs';

function getElement(id: string): HTMLElement {
    let element = document.querySelector(id);
    if (!element)
        throw new Error(`Cannot find element "${id}"`);
    return element as HTMLElement;
}

// Make sure all necessary HTML elements are present and acquire them
function loadElements(): [HTMLElement, HTMLElement, HTMLElement] {
    let output = getElement("#output")
    let grid = getElement(".grid")
    let clearBtn = getElement("#clear")

    return [output, grid, clearBtn];
}

// Convert the array of checkbox elements into an array of 0's and 1's
function getImageArray(checkboxes: HTMLInputElement[]) {
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

// Use the TensorFlow model to take the array of checkbox elements and output predictions
function showPredictions(model: tf.LayersModel, checkboxes: HTMLInputElement[], output: HTMLElement) {
    // Get and shape input
    let inputTensor = tf.tensor(getImageArray(checkboxes)).reshape([1, 28, 28, 1]);

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

// Main function
window.addEventListener('load', () => {

    // Get HTML elements
    let [output, grid, clearBtn] = loadElements();

    // Load model from filesystem.
    tf.loadLayersModel('./outputModel2/model.json').then((model) => {

        // Create checkbox elements
        let checkboxes = createInputCheckboxes(grid, () => {
            showPredictions(model, checkboxes, output);
        });

        // Create initial predictions to get the model "warmed up"
        // (The first prediction of the model is significantly slower. I'm guessing that it doesn't fully load until the first prediction is made)
        showPredictions(model, checkboxes, output);


        // Clear button functionality
        clearBtn.addEventListener('click', () => {
            for (let i = 0; i < checkboxes.length; i++) {
                checkboxes[i].checked = false;
            }
            showPredictions(model, checkboxes, output);
        })

    });


})