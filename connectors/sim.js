function connect() {
    let { PythonShell } = require("python-shell");
    var path = require("path");

    var Width = document.getElementById('Width').value
    document.getElementById('Width').value = "";

    var Depth = document.getElementById('Depth').value
    document.getElementById('Depth').value = "";

    var Temperature = document.getElementById('Temperature').value
    document.getElementById('Temperature').value = "";

    var Pressure = document.getElementById('Pressure').value
    document.getElementById('Pressure').value = "";

    var Timestep = document.getElementById('Timestep').value
    document.getElementById('Timestep').value = "";

    document.getElementById('createCharts').disabled = true;

    var options = {
        scriptPath: path.join(__dirname, '/backend/'),
        args: [Width, Depth, Temperature, Pressure, Timestep]
    }

    console.log(options.args);

    document.getElementById("gif").style.display = "inline"

    var testRetrieval = new PythonShell('simulationScript.py', options);
    testRetrieval.on('message', function (message) {
        CreateCharts();
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