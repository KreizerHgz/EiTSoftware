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

    var testRetrieval = new PythonShell('testRetrieval.py', options);
    testRetrieval.on('message', function (message) {
        CreateCharts();
    });
}

function CreateCharts() {

}