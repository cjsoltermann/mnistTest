import * as tf from '@tensorflow/tfjs';

//tf.loadLayersModel('./mnistModel/model.json').then((model) => {
tf.loadLayersModel('./outputModel2/model.json').then((model) => {
    let output = document.getElementById('output');

    function showPredictions() {
        let tensor = tf.tensor(getImageArray()).reshape([1, 28, 28, 1]);
        let result = model.predict(tensor).arraySync()[0];
        let maxResult = Math.max(...result);
        let resultString = result.map((n, i) => (n == maxResult ? "<b>" : "") + i + " : " + Number(n).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 }) + (n == maxResult ? "</b>" : "")).join('      \r\n');

        output.innerHTML = resultString;
    }

    let checkboxes = [];
    document.querySelector('.grid').innerHTML = "";
    for (let i = 0; i < 28 * 28; i++) {
        let checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkboxes.push(checkbox);
        document.querySelector('.grid').appendChild(checkbox);
    }

    showPredictions();

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
                showPredictions();
            }
        });
    }

    let clearBtn = document.getElementById('clear');
    clearBtn.addEventListener('click', () => {
        for (let i = 0; i < checkboxes.length; i++) {
            checkboxes[i].checked = false;
        }
        showPredictions();
    })

    async function copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            console.log('Text copied to clipboard');
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    }

    let exportBtn = document.getElementById('export')
    exportBtn.addEventListener('click', () => {
        let numpyArray = getImageArray();

        let numpyArrayString = 'np.array([\n' + numpyArray.map(row => '  ' + JSON.stringify(row)).join(',\n') + '\n])\n';

        copyToClipboard(numpyArrayString);
    })

    function getImageArray() {
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
});

