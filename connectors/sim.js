function connect() {
    let { PythonShell } = require("python-shell");
    var path = require("path");

    var mpSpace = document.getElementById('mpSpace').value
    document.getElementById('mpSpace').value = "";

    var mpTime = document.getElementById('mpTime').value
    document.getElementById('mpTime').value = "";

    var kwt = document.getElementById('kwt').value
    document.getElementById('kwt').value = "";

    var kwb = document.getElementById('kwb').value
    document.getElementById('kwb').value = "";

    var Time = document.getElementById('Time').value
    document.getElementById('Time').value = "";

    document.getElementById('createCharts').disabled = true;

    var options = {
        scriptPath: path.join(__dirname, '/backend/'),
        args: [mpSpace, mpTime, kwt, kwb, Time]
    }

    console.log(options.args);

    document.getElementById("gif").style.display = "inline"

    var testRetrieval = new PythonShell('simulationScript.py', options);
    testRetrieval.on('message', function (message) {
        if (message == "OK") {
            CreateCharts();
        }
        else {
            HandleError();
        }
    });
}

function CreateCharts() {
    var img1 = document.getElementById("testImage1");
    img1.style.width = "400px"
    img1.style.height = "320px"
    img1.src = "./images/figure.png"
    var cap1 = document.getElementById("img1Caption");
    cap1.innerHTML = "Test1"

    var img2 = document.getElementById("testImage2");
    img2.style.width = "400px"
    img2.style.height = "320px"
    img2.src = "./images/OL.png"
    var cap2 = document.getElementById("img2Caption");
    cap2.innerHTML = "Test2"

    var img3 = document.getElementById("testImage3");
    img3.style.width = "400px"
    img3.style.height = "320px"
    img3.src = "./images/bike.png"
    var cap3 = document.getElementById("img3Caption");
    cap3.innerHTML = "Test3"

    var img4 = document.getElementById("testImage4");
    img4.style.width = "400px"
    img4.style.height = "320px"
    img4.src = "./images/sq.png"
    var cap4 = document.getElementById("img4Caption");
    cap4.innerHTML = "Test4"

    document.getElementById("gif").style.display = "none"
}

function HandleError() {
    document.getElementById("gif").style.display = "none"
    alert("An exception occurred");
}