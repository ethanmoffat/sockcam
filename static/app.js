var socket = io();

socket.on("count_change", function(message) {
    var countElem = document.getElementById("count");
    countElem.innerHTML = "People watching: " + message["data"];
});
