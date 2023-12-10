function authenticateUser(event) {
    event.preventDefault(); // Prevent the default form submission

    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // Create a new XMLHttpRequest object
    const xhr = new XMLHttpRequest();

    // Configure the request
    xhr.open("POST", "/authenticate", true);
    xhr.setRequestHeader("Content-Type", "application/json");

    // Set up the callback function to handle the response
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            const responseText = xhr.responseText;
            alert(responseText); // Display the response
	    
		const bananabreads = document.querySelectorAll(".bananabreads");
        }
    };

    // Send the request with the data "bananabread"
    xhr.send(JSON.stringify({ username, password, secret: "bananabread" }));
}

