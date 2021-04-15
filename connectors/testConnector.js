
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
        CreateChart(message);
    });
}


function CreateChart(inputData) {

    var Chart = require('chart.js');
    var ctx = document.getElementById('chart').getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                //data: [12, 19, 3, 5, 2, 3],
                data: JSON.parse(inputData),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            }
        }
    });
}