let moves = 0;

var board = Chessboard('myBoard', {
  draggable: true,
  position: 'start',
  onDrop: handleMove,
  pieceTheme: '/static/chessboardjs-1.0.0/img/chesspieces/wikipedia/{piece}.png'
  // other configuration options...
});

function handleMove(source, target) {
    // Get the current board state in FEN
    var currentBoardFEN = board.fen();

    // Prepare data to send, including the move and the current board state
    var dataToSend = {
        move: source + target,
        currentBoard: currentBoardFEN
    };

    // AJAX request to the backend
    $.ajax({
        url: '/make_move',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(dataToSend),
        success: function(response) {
	    console.log("Success");
	    if (response.new_board === "Checkmate") {
		alert(`Checkmate in ${moves} moves!`);
	    } else {
	    	console.log(response.new_board);
            	// Update the board with the response
            	board.position(response.new_board);
		moves += 1;
	    }
        },
        error: function(error) {
            // Handle errors (e.g., invalid move)
            console.log(error);
	    alert("Move not valid");
	    board.position(currentBoardFEN);
        }
    });
}

document.getElementById('myBoard').addEventListener('touchmove', function(e) {
    e.preventDefault();
}, { passive: false });

