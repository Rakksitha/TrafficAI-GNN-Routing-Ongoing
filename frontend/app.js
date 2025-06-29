
// Initialize Stick Map (Empty)
function initializeStickMap() {
    let stickMap = document.getElementById("stick-map");
    stickMap.innerHTML = "<h3>Waiting for Route...</h3>"; 
}

// Fetch Route & Update Stick Map
function fetchRoute() {
    let start_x = document.getElementById("start_x").value;
    let start_y = document.getElementById("start_y").value;
    let dest_x = document.getElementById("dest_x").value;
    let dest_y = document.getElementById("dest_y").value;
    let priority = document.getElementById("priority").value;

    fetch(`http://127.0.0.1:5000/get_route?start_x=${start_x}&start_y=${start_y}&dest_x=${dest_x}&dest_y=${dest_y}&priority=${priority}`)
        .then(response => response.json())
        .then(data => {
            console.log("Route Data Received:", data); // Debugging log

            if (data.route && data.route.length > 0) {
                updateStickMap(data.route);
            } else {
                alert("No route found!");
            }
        })
        .catch(error => {
            console.error("Error fetching route:", error);
            alert("Failed to fetch route. Check backend connection.");
        });
}

// Update Stick Map with Route Path
function updateStickMap(route) {
    let stickMap = document.getElementById("stick-map");
    stickMap.innerHTML = "";  // Clear previous content

    route.forEach((point, index) => {
        let node = document.createElement("div");
        node.innerText = `(${point[0].toFixed(4)}, ${point[1].toFixed(4)})`;
        node.classList.add("stick-node");
        stickMap.appendChild(node);

        if (index < route.length - 1) {
            let line = document.createElement("div");
            line.classList.add("stick-line");
            stickMap.appendChild(line);
        }
    });
}

// Generate Line Chart for Peak Hours
function generateLineChart() {
    let hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    let trafficIntensity = [10, 15, 20, 25, 50, 80, 100, 90, 85, 75, 65, 55, 45, 40, 35, 30, 28, 32, 50, 70, 85, 95, 100, 80];

    let trace = {
        x: hours,
        y: trafficIntensity,
        mode: "lines+markers",
        type: "scatter",
        name: "Traffic Intensity",
        line: { color: "red" }
    };

    let layout = {
        title: "Line Chart with Peak Hours During a Day",
        xaxis: { title: "Time (Hours)" },
        yaxis: { title: "Traffic Intensity" }
    };

    Plotly.newPlot("traffic-flow", [trace], layout);
}

function generateHeatmap() {
    let heatData = [
        [10, 30, 50, 70, 90],
        [20, 40, 60, 80, 100],
        [15, 35, 55, 75, 95],
        [5, 25, 45, 65, 85],
        [0, 20, 40, 60, 80]
    ];

    let trace = {
        z: heatData,
        type: "heatmap",
        colorscale: "Reds"
    };

    let layout = {
        title: "Traffic Congestion Levels",
        xaxis: { title: "Junctions" },
        yaxis: { title: "Time of Day" }
    };

    Plotly.newPlot("congestion-map", [trace], layout);
}

// Initialize Page
generateLineChart();
generateHeatmap();
initializeStickMap();

