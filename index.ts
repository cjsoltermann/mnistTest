import * as tf from '@tensorflow/tfjs';

// Make sure all necessary HTML elements are present and acquire them
function loadElements(): [HTMLElement, HTMLElement, HTMLElement] {
    let output = document.getElementById('output');
    if (!output) {
        throw new Error("Cannot find output element");
    }
    output = output as HTMLElement;
    let grid = document.querySelector('.grid') as HTMLElement | null;
    if (!grid) {
        throw new Error("Cannot find input grid");
    }
    grid = grid as HTMLElement;

    let clearBtn = document.getElementById('clear');
    if (!clearBtn) {
        throw new Error("Cannot find clear button");
    }
    clearBtn = clearBtn as HTMLElement;

    return [output, grid, clearBtn];
}

// Convert the array of checkbox elements into an array of 0's and 1's
function getImageArray(checkboxes: HTMLInputElement[]) {
    let numpyArray = [];
    for (let i = 0; i < 28; i++) {
        let row = [];
        for (let j = 0; j < 28; j++) {
            let checkboxIndex = i * 28 + j;
            row.push(checkboxes[checkboxIndex].checked ? 1 : 0);
        }
        numpyArray.push(row);
    }

    return numpyArray;
}

// Use the TensorFlow model to take the array of checkbox elements and output predictions
function showPredictions(model: tf.LayersModel, checkboxes: HTMLInputElement[], output: HTMLElement) {
    let tensor = tf.tensor(getImageArray(checkboxes)).reshape([1, 28, 28, 1]);
    let resultTensor = model.predict(tensor) as tf.Tensor;
    let result = (resultTensor.arraySync() as number[][])[0];
    let maxResult = Math.max(...result);
    let resultString = result.map((n, i) => (n == maxResult ? "<b>" : "") + i + " : " + Number(n).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 }) + (n == maxResult ? "</b>" : "")).join('      \r\n');

    output.innerHTML = resultString;
}

// Create the 28x28 array of checkboxes and setup mouse input
function createInputCheckboxes(grid: HTMLElement, updateFunc: () => void) {
    let checkboxes: HTMLInputElement[] = [];
    grid.innerHTML = "";
    for (let i = 0; i < 28 * 28; i++) {
        let checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkboxes.push(checkbox);
        grid.appendChild(checkbox);
    }

    let isMouseDown = false;
    document.addEventListener('mousedown', function () {
        isMouseDown = true;
    });
    document.addEventListener('mouseup', function () {
        isMouseDown = false;
    });
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