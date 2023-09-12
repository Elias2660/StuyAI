function stuyActivities() {
                location.href = 'https://stuyactivities.org/stuyai'
        }

function openBlock(header_id, content_id) {
    let header = document.getElementById(header_id);
    let content = document.getElementById(content_id);
    let content_boxes = document.getElementsByClassName("content");
    let header_boxes = document.getElementsByClassName("exercise_header");

    // Close all other content boxes
    for (let i = 0; i < content_boxes.length; i++) {
        if (content_boxes[i].id !== content_id) {
            content_boxes[i].style.display = "none";
        }
        if (header_boxes[i].id !== header_id) {
            header_boxes[i].style.borderRadius = "5px";
            header_boxes[i].style.borderBottomColor = "white";
        }
      }


    if (content.style.display === "none" || content.style.display === "") {
        content.style.display = "block";
        content.style.height = "450px";
        header.style.borderRadius = "5px 5px 0px 0px";
        content.style.borderRadius = "0px 0px 5px 5px";

        content.style.transition = "height 3s";
        header.style.borderBottomColor = "black";
        content.style.borderTopColor = "black";
    } else {
        content.style.height = "0px";

        content.style.transition = "height 3s";
        content.style.display = "none";
        header.style.borderRadius = "5px";

        header.style.borderBottomColor = "white";
        content.style.borderTopColor = "white";
    }
}


var textarea = document.querySelector('textarea');

textarea.addEventListener('keydown', function(event) {
  if (event.key === 'Tab') {
    event.preventDefault();

    // Get the current cursor position
    var start = this.selectionStart;
    var end = this.selectionEnd;

    // Insert a tab character at the current cursor position
    this.value = this.value.substring(0, start) + '\t' + this.value.substring(end);

    // Move the cursor position after the inserted tab character
    this.selectionStart = this.selectionEnd = start + 1;
  }
});
