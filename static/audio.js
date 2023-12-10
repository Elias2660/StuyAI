let mediaRecorder;
let audioChunks = [];
let recordingInterval;
let recordingDuration = 0;

document.getElementById('recordButton').addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            // Reset and start the recording duration counter
            recordingDuration = 0;
            document.getElementById('duration').textContent = recordingDuration;
            recordingInterval = setInterval(() => {
                recordingDuration++;
                document.getElementById('duration').textContent = recordingDuration;
            }, 1000);

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                clearInterval(recordingInterval); // Stop the duration counter
                const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
                let formData = new FormData();
                formData.append('audio', audioBlob);
                
                // Enable the upload button
                document.getElementById('uploadButton').disabled = false;
            });

            // Enable the stop button
            document.getElementById('stopButton').disabled = false;
        });
});

document.getElementById('stopButton').addEventListener('click', () => {
    mediaRecorder.stop();
    document.getElementById('stopButton').disabled = true;
    document.getElementById('recordButton').disabled = false;
});

document.getElementById('uploadButton').addEventListener('click', () => {

// Updated part of the JavaScript code
const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
let formData = new FormData();
formData.append('file', audioBlob, 'recording.mp3');

fetch('/models/audio', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Upload successful');
    alert(data.prediction);
}).catch(error => {
    console.error('Error:', error);
});
});

