import tempfile
import re
import subprocess
import os
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
import csv
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from io import BytesIO
from starlette.responses import FileResponse
import torch
import shutil
import json
import chess
from io import StringIO
from pydub import AudioSegment

app = FastAPI()

# Load the model outside of the endpoint to avoid loading it multiple times
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best50.pt', trust_repo=True, force_reload=True)

# Define a list of safe modules and functions
#SAFE_MODULES = ['math', 'random']
#SAFE_FUNCTIONS = ['print', 'len']

# Set resource limits
#MEMORY_LIMIT = 256  # Memory limit in MB
#CPU_LIMIT = 5  # CPU time limit in seconds


# NLP Stuff
import pandas as pd
import re
import string
import random
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import StandardScaler

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt')

# Text preprocessing function
def preprocess_text(text):
    # Normalize text to lowercase
    print(text)
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words if len(word) > 2])


# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the templates directory
templates = Jinja2Templates(directory="templates")

def load_csv_data():
    data = {}
    with open('data.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            month_topic = row['Month Topic']
            lesson_name = row['Lesson Name']
            problem = row['Problem']
            hint = row['Hint']
            solution = row['Solution'].replace('\\n', '<br>').replace('\\t', '&emsp;')
            another_solution = row['Another Solution'].replace('\\n', '<br>').replace('\\t', '&emsp;')
            third_solution = row['Third Solution (optional)'].replace('\\n', '<br>').replace('\\t', '&emsp;')
            
            print(row["Solution"])
            print(solution)
            print(row["Another Solution"])
            print(another_solution)

            lesson_data = {
                'Problem': problem,
                'Hint': hint,
                'Solutions': [solution, another_solution]
            }

            if third_solution:
                lesson_data['Solutions'].append(third_solution)

            if month_topic not in data:
                data[month_topic] = {}
            if lesson_name not in data[month_topic]:
                data[month_topic][lesson_name] = []

            data[month_topic][lesson_name].append(lesson_data)
    return data

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Read the webhook secret from a file
with open("webhook_secret.txt", "r") as secret_file:
    WEBHOOK_SECRET = secret_file.read().strip()

from subprocess import Popen, PIPE

# Define a function to update files from the GitHub repository and restart your FastAPI app
def update_files_and_restart():
    try:
        # Change the working directory to your project's home directory
        os.chdir("/root/website")

        # Use Git to pull the latest changes from the main branch
        git_pull = Popen(["git", "pull", "origin", "main"], stdout=PIPE, stderr=PIPE)
        stdout, stderr = git_pull.communicate()

        if git_pull.returncode != 0:
            print("failed")
            return {"message": f"Git pull failed: {stderr.decode('utf-8')}", "status_code": 500}

        # Restart your FastAPI app
        uvicorn_process = Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"])
        uvicorn_process.wait()

        return {"message": "Files updated and FastAPI app restarted", "status_code": 200}
    except Exception as e:
        print("failed")
        return {"message": str(e), "status_code": 500}

@app.post("/webhook")
async def github_webhook(request: Request):
    # Verify the webhook secret (replace with your actual secret)
    secret_key = request.headers.get("X-Hub-Signature")
    print(WEBHOOK_SECRET)
    print(secret_key)
    if secret_key != WEBHOOK_SECRET:
        return {"message": "Invalid secret", "status_code": 403}

    # Parse the JSON payload
    payload = await request.json()

    # Handle different GitHub webhook events as needed
    event_type = request.headers.get("X-GitHub-Event")
    if event_type == "push":
        # Handle push event (e.g., update files and restart)
        update_files_and_restart()
        return {"message": "Webhook processed", "status_code": 200}
    
    # Handle other event types as needed
    return {"message": "Webhook not processed", "status_code": 200}

@app.get("/blog")
async def read_blog(request: Request):
    return templates.TemplateResponse("blog.html", {"request": request})


@app.get("/about")
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/models")
async def models(request: Request):
    return templates.TemplateResponse("models.html", {"request": request})

@app.get("/practice")
async def read_practice(request: Request):
    data = load_csv_data()
    return templates.TemplateResponse("practice.html", {"request": request, "data": data})

@app.get("/resources")
async def read_practice(request: Request):
    return templates.TemplateResponse("resources.html", {"request": request})

@app.get("/bananabread")
async def bananabread(request: Request):
    data = load_csv_data()
    return templates.TemplateResponse("bananabread.html", {"request": request, "data": data})

# This should be replaced with actual user data handling
users_data = {
    "user123": {
        "password": "password123"
    }
}

class UserCredentials(BaseModel):
    username: str
    password: str
    secret: str

@app.post("/authenticate")
async def authenticate_user(credentials: UserCredentials):
    username = credentials.username
    password = credentials.password

    # Check if the provided username and password match
    if username in users_data and users_data[username]["password"] == password:
        return {"message": credentials.secret}
    else:
        raise HTTPException(status_code=400, detail="Incorrect login information")

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/upload")
async def privacy(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def determine_brightness(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Resize image using PIL methods
    img = image.resize((384, 384))

    # Convert PIL image to numpy array for the model
    img_array = np.array(img)

    # Ensure the input is in the format the model expects
    results = model([img_array])  # The model expects a batch of images

    # Extract the PIL image with boxes and labels
    pil_img = results.render()[0]  # results.render() returns a list of images
    annotated_img = Image.fromarray(pil_img)

    # Save the annotated image
    annotated_img.save('result.jpg')

    return FileResponse('result.jpg', media_type='image/jpeg')

@app.get("/train")
async def train(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/getImageList")
async def get_image_list():
    image_list = []
    files = os.listdir("./static/door_raw_data/")
    other_files = os.listdir("./static/annotations/")
    print(len(other_files)/ len(files) * 100)
    for file in files:
        if (file[:-4] + ".txt") not in other_files:
            image_list.append(file)
    return {"images": image_list}

@app.post("/train")
async def upload_data(
    file_name: str = Form(...),
    annotations: str = Form(...)
):
    annotations_data = json.loads(annotations)
    formatted_data = f"{file_name}\n"
    for ann in annotations_data:
        formatted_data += f"{round(ann[0]['x'])} {round(ann[0]['y'])} {round(ann[1]['x'])} {round(ann[1]['y'])}\n"
    with open(f"static/annotations/{file_name[:-4]}.txt", "w") as label_file:
        label_file.write(formatted_data)
    return {"status": "success", "message": "Data uploaded successfully"}

@app.post("/test")
async def test_post(request: Request):
    # Replace the following line with your desired response
    return {"message": "This is a test POST endpoint"}

@app.post('/waitlist')
async def add_to_waitlist(data: dict):
    try:
        # Open the CSV file in append mode
        with open('waitlist.csv', 'a', newline='') as csvfile:
            fieldnames = ['fullName', 'stuyvesantEmail', 'name1', 'email1', 'name2', 'email2', 'name3', 'email3']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Create a dictionary with all fieldnames and their values
            entry = {fieldname: data.get(fieldname, '') for fieldname in fieldnames}

            # Write the data to the CSV file
            writer.writerow(entry)

        return JSONResponse(content={'message': 'Data added to waitlist'}, status_code=200)
    except Exception as e:
        return JSONResponse(content={'message': 'Error adding to waitlist'}, status_code=500)

@app.post("/execute")
async def execute_code(request: Request):
    code = await request.json()
    code = code.get("code")
    print("code: ", code) 

    # Compile the code to check for syntax errors
    compile_error = compile_code(code)
    if compile_error:
        return {"output": compile_error}
    
    # Execute the code with time limit
    result = execute_code(code)
    print("result: ", result)

    if result is not None:
        return {"output": result}
    else:
        return {"output": "Code execution timed out"}

# Sanitize and validate the code
def sanitize_code(code):
    # Remove any potentially harmful statements
    code = re.sub(r'import\s+os', '', code)
    code = re.sub(r'subprocess\.', '', code)

    # Restrict function calls to safe functions only
    #for function in SAFE_FUNCTIONS:
    #    code = re.sub(rf'(?<!\w){function}\b', '', code)

    # Restrict module imports to safe modules only
    #for module in SAFE_MODULES:
    #    code = re.sub(rf'(?<!\w)import\s+{module}\b', '', code)

    return code.strip()

# Compile the code to check for syntax errors
def compile_code(code):
    try:
        compile(code, filename='<string>', mode='exec')
    except SyntaxError as e:
        return str(e)

    return None

# Execute the code with PyPy using subprocess.run
def execute_code(code):
    code = sanitize_code(code)
    print("santizized code: ", code)

    # Create a temporary file to store the code
    with open('code.py', 'w') as file:
        file.write(code)

    # Execute the code with PyPy using subprocess.run and capture the output in a file
    output_file = 'output.txt'
    with open(output_file, 'w') as file:
        try:
            subprocess.run(['pypy', 'code.py'], stdout=file, stderr=file, timeout=CPU_LIMIT)
        except subprocess.TimeoutExpired:
            return None

    # Read the output from the file
    with open(output_file, 'r') as file:
        result = file.read()

    # Remove the temporary files
    os.remove('code.py')
    os.remove(output_file)

    result = re.sub("\n", "<br>", result)

    return result

# This will be the code for the chess game (AI bot)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from scipy import signal
from scipy.io import wavfile

ACTION_SPACE_SIZE = 4672

# Define the piece values
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 30}

class EnhancedChessCNN(nn.Module):
    def __init__(self, action_size):
        super(EnhancedChessCNN, self).__init__()

        # Input is now 12 channels, one for each piece type for both colors
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # You might experiment with the number of fully connected layers and their sizes
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(p=0.3)  # Adjust dropout rate as needed
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Function to encode a move into an integer
def encode_move(move):
    return move.from_square * 64 + move.to_square

# Function to decode an integer back into a move
def decode_move(encoded_move, legal_moves):
    from_square = encoded_move // 64
    to_square = encoded_move % 64
    for move in legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            return move
    return None

# Function to encode the board state
def encode_board(board):
    # Define a mapping from pieces to channels
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    # Create a 12x8x8 numpy array to represent the board
    board_array = np.zeros((12, 8, 8), dtype=np.float32)

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            channel = piece_to_channel[piece.symbol()]
            board_array[channel, i // 8, i % 8] = 1  # Place the piece in the corresponding channel

    return board_array

# Load White's model
model_white = EnhancedChessCNN(ACTION_SPACE_SIZE)
model_white.load_state_dict(torch.load('chess_model1000_5_black_575.pth'))
model_white.eval()

# Load Black's model
model_black = EnhancedChessCNN(ACTION_SPACE_SIZE)
model_black.load_state_dict(torch.load('chess_model1000_5_black_575.pth'))
model_black.eval()

# Function to choose a move with the model
def choose_move_with_model(board, model):
    state = encode_board(board)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)

    legal_moves = list(board.legal_moves)
    best_move = None
    best_q_value = -float('inf')

    for move in legal_moves:
        encoded_move = encode_move(move)
        if encoded_move < ACTION_SPACE_SIZE:
            q_value = q_values[0, encoded_move].item()
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = move

    return best_move

@app.get("/chess")
async def train(request: Request):
    return templates.TemplateResponse("chess.html", {"request": request})

class ChessMove(BaseModel):
    move: str
    currentBoard: str

@app.post("/make_move")
async def make_move(chess_move: ChessMove):
    # Extract move and current board state from the request
    move = chess_move.move
    current_board_fen = chess_move.currentBoard

    print(move)
    print(current_board_fen)

    # Initialize a chess board with the current state
    board = chess.Board(current_board_fen)

    # Check if the move is a pawn promotion without specified promotion piece
    if len(move) == 4 and board.piece_at(chess.parse_square(move[:2])).piece_type == chess.PAWN:
        if move[1] == '7' and move[3] == '8':  # White pawn promotion
            move += 'q'  # Promote to queen by default
        elif move[1] == '2' and move[3] == '1':  # Black pawn promotion
            move += 'q'  # Promote to queen by default

    if board.is_game_over():
        return {"new_board": "Checkmate"}

    # Print the board to the console
    print("Current Board:")
    print(board)
    print(board.legal_moves)

    # Validate and execute the move
    try:
        move_obj = board.parse_san(move)
        print("Parsed Move Object:", move_obj)
        if move_obj not in board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        board.push(move_obj)
    except Exception as e:
        print("Error occurred:", e)
        return JSONResponse(status_code=400, content={"message": str(e)})

    if board.is_game_over():
        return {"new_board": "Checkmate"}

    # Computer makes its move
    move = choose_move_with_model(board, model_black)
    board.push(move)

    if board.is_game_over():
        return {"new_board": "Checkmate"}

    # Return the new board position
    return {"new_board": board.fen()}

# NLP

@app.get("/models/nlp")
async def train(request: Request):
    return templates.TemplateResponse("nlp_sent_model.html", {"request": request})

@app.post("/models/nlp")
async def nlp_sent(file: UploadFile = File(None), text: str = Form(None)):
    # If a file is uploaded, use the file
    if file:
        contents = await file.read()
        dataset = pd.read_csv(StringIO(contents.decode('utf-8')))
    # If text is provided in the textbox, assume it's CSV formatted and convert it to a DataFrame
    elif text:
        dataset = pd.read_csv(StringIO(text))
    else:
        return {"error": "No input provided"}

    # Apply text preprocessing to the first column of the dataset
    dataset['Processed'] = dataset.iloc[:, 0].apply(preprocess_text)

    # Splitting the dataset, assuming the first column is the text and the second column is the labels
    X_train, X_test, y_train, y_test = train_test_split(
        dataset['Processed'],   # Text column (preprocessed)
        dataset.iloc[:, 1],     # Label column
        test_size=0.2, 
        random_state=42
    )


    # Feature extraction with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Classifier - Logistic Regression with increased max_iter
    model = LogisticRegression(max_iter=1500)

    # Hyperparameter tuning using Grid Search
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_tfidf, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate the model
    predictions = best_model.predict(X_test_tfidf)
    evaluation_report = classification_report(y_test, predictions)

    # Print the evaluation report
    print(evaluation_report)

    # Function to generate a random 6 character string
    def generate_random_string(length=6):
        letters_and_digits = string.ascii_letters + string.digits
        return ''.join(random.choice(letters_and_digits) for i in range(length))

    # Generate a random filename
    filename = generate_random_string() + '.pkl'

    # Save the best model to file
    joblib.dump(best_model, "./nlp/" + filename)
    joblib.dump(tfidf_vectorizer, f'./nlp/tfidf_vectorizer{filename}')

    # Print the filename for confirmation
    print("Model saved as:", filename)

    return {"message": filename[:-4], "report": evaluation_report}

@app.post("/models/run_nlp")
async def run_nlp(modelName: str = Form(None), new_comment: str = Form(None)):
    if not modelName or not new_comment:
        raise HTTPException(status_code=400, detail="Model name and comment must be provided")

    try:
        # Load the model
        model = joblib.load(f'./nlp/{modelName}.pkl')
        print("found model")
        # Load the vectorizer - ensure this is the same vectorizer used during training
        tfidf_vectorizer = joblib.load(f'./nlp/tfidf_vectorizer{modelName}.pkl')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model or Vectorizer not found")

    # Preprocess and vectorize the new comment
    processed_comment = preprocess_text(new_comment)
    vectorized_comment = tfidf_vectorizer.transform([processed_comment])

    # Make a prediction
    prediction = model.predict(vectorized_comment)

    return {"prediction": prediction[0], "processed_comment": processed_comment}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = f"{file_name}"
    return FileResponse(path=file_path, filename=file_name)


@app.get("/suno_ai")
async def redirect_to_drive():
    return RedirectResponse(url="https://drive.google.com/drive/folders/1fj0DgDX3gWYcq2zKYqIyUPUrldgshEcB?usp=drive_link")

@app.get("/adobe_podcast")
async def redirect_to_drive_folder():
    return RedirectResponse(url="https://drive.google.com/drive/folders/17AAaVVL3O5cvIojCBpYwn2-t4tszhVrx?usp=sharing")

@app.get("/hi_bye")
async def drive_hi_bye():
    return RedirectResponse(url="https://drive.google.com/drive/folders/1tF4qGo1Bn1f4xEKNgh_B2W0yqLjU2LpS?usp=drive_link")

@app.get("/heartbeat")
async def heartbeat():
    return RedirectResponse(url="https://drive.google.com/drive/folders/1C_OVRjDOUEkHw_5_sKA_663x3T-w-2Ve?usp=drive_link")

@app.get("/music")
async def music():
    return RedirectResponse(url="https://drive.google.com/drive/folders/1UdopsUna_rNoKPvwkZVuNAPoiKjhun1Q?usp=sharing")

# Cat model
max_length = 1283
class Animal_Sound_Model(nn.Module):
    def __init__(self):
        super(Animal_Sound_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)  # Conv2d layer
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(self.calculate_linear_input_size(), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the linear layer
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

    def calculate_linear_input_size(self):
        # Temporary input for size calculation
        temp_input = torch.zeros(1, 1, 129, max_length)
        temp_out = self.conv1(temp_input)
        temp_out = self.pool(temp_out)
        temp_out = self.conv2(temp_out)
        temp_out = self.pool(temp_out)
        temp_out = self.conv3(temp_out)
        temp_out = self.pool(temp_out)
        return temp_out.view(temp_out.size(0), -1).shape[1]
import librosa
def process_audio(file_path, max_length):
    # Load MP3 file
    samples, sample_rate = librosa.load("download.mp3", sr=None, mono=True)  # sr=None ensures original sample rate is used

    # Spectrogram processing
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    padded_spectrogram = np.pad(spectrogram, ((0, 0), (0, max_length - spectrogram.shape[1])), mode='constant', constant_values=0)

    return torch.tensor(padded_spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Load the saved model
model = Animal_Sound_Model()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

@app.get("/models/audio")
async def train(request: Request):
    return templates.TemplateResponse("audio.html", {"request": request})

@app.post("/models/audio")
async def classify_audio(file: UploadFile = File(...)):
    max_length = 1283

    # Open the file in binary write mode
    with open("download.mp3", "wb") as f:
        contents = await file.read()  # Read the file asynchronously
        f.write(contents)  # Write the binary data to a file


    # Create a temporary file for the MP3
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        contents = await file.read()  # Read the file asynchronously
        temp_file.write(contents)
        temp_file_path = temp_file.name

    # Process audio file
    processed_audio = process_audio(temp_file_path, max_length)

    # Classify the audio
    with torch.no_grad():
        output = model(processed_audio)
        prediction = torch.argmax(output, dim=1)
        result = "Dog" if prediction.item() == 1 else "Cat"

    return {"prediction": result}

@app.get("/models/audio_code")
async def make_code(request: Request):
        return templates.TemplateResponse("audio_code.html", {"request": request})

class ConfigData(BaseModel):
    num_layers: int
    dropout_rate: float
    l2_reg: float
    batch_size: int
    learning_rate: float
    num_epochs: int
    model_type: int

@app.post("/audio_code_make")
async def create_config(data: ConfigData):
    # Formatting the sample string with the received configuration
    formatted_string = f"""import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import time
from sklearn.model_selection import train_test_split
import os

# Paths for the two classes
class1_path = "/content/data/jazz/"  # path
class2_path = "/content/data/hiphop/"  # path

# Neural network configuration
num_layers = {data.num_layers}  # Define the number of convolutional layers
dropout_rate = {data.dropout_rate}  # Dropout rate for regularization
l2_reg = {data.l2_reg}  # L2 regularization factor

batch_size = {data.batch_size}  # Number of samples per batch
learning_rate = {data.learning_rate}  # Learning rate for the optimizer
num_epochs = {data.num_epochs}  # Total number of training epochs

# Function to read audio files and convert them into spectrograms
def get_waveforms_labels(path, label):
    features, labels = [], []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            try:
                # Reading the WAV file
                sample_rate, samples = wavfile.read(os.path.join(path, filename))
                # Generating a spectrogram
                frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
                features.append(spectrogram)
                labels.append(label)
            except Exception as e:
                print(f"Error reading file")
    return features, labels

# Function to pad the spectrograms to a uniform size
def pad_spec(spectrogram, max_length):
    return np.pad(spectrogram, ((0, 0), (0, max_length - spectrogram.shape[1])), mode='mean')

# Data preprocessing
print("Data preprocessing...")
class1_features, class1_labels = get_waveforms_labels(class1_path, 0)
class2_features, class2_labels = get_waveforms_labels(class2_path, 1)

# Combining the features and labels
all_features = class1_features + class2_features
all_labels = class1_labels + class2_labels

# Determining the maximum length of spectrograms
max_length = max(s.shape[1] for s in all_features)

# Padding spectrograms and converting to numpy arrays
X = np.array([pad_spec(s, max_length) for s in all_features])
y = np.array(all_labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Length X train", len(X_train))

# Convert to tensors and create DataLoader for training and testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparing training data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Preparing testing data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Neural network model definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Dynamically adding convolutional layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 1 if i == 0 else 8 * 2**(i-1)
            out_channels = 8 * 2**i
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))  # Added batch normalization
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer for binary classification
        self.fc = nn.Linear(self.calculate_linear_input_size(), 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # Removed sigmoid activation

    # Function to calculate the input size of the fully connected layer
    def calculate_linear_input_size(self):
        temp_input = torch.zeros(1, 1, 129, max_length)
        for layer in self.layers:
            temp_input = layer(temp_input)
        return temp_input.view(temp_input.size(0), -1).shape[1]

# Initializing the model, criterion, and optimizer
model = Model().to(device)
criterion = nn.BCEWithLogitsLoss()
# Changing optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

def calculate_accuracy(loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            predicted = (outputs.data > 0.5).float()  # Convert to binary predictions
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Training loop
print("Training...")
all_loss = []
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    train_losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Calculate training loss and accuracy
    avg_train_loss = np.mean(train_losses)
    all_loss.append(avg_train_loss)
    train_acc = calculate_accuracy(train_loader, model)

    # Calculate validation accuracy
    val_acc = calculate_accuracy(test_loader, model)  # Using test_loader as a proxy for validation

    print("Epoch: " + str(epoch+1) + ", Loss-train: " + str(format(avg_train_loss, '1.3f')) + 
      ", Train Acc: " + str(format(train_acc, '.3f')) + ", Val Acc: " + str(format(val_acc, '.3f')) + 
      ", Time: " + str(format(time.time() - start_time, '2.2f')) + "s")

# Calculate and display test set accuracy
test_acc = calculate_accuracy(test_loader, model)
print("Test Accuracy: " + str(test_acc))

# Saving the trained model
torch.save(model.state_dict(), 'model.pth')

# Plotting training loss
plt.title("Training Loss")
plt.plot(range(num_epochs), all_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
"""
    if data.model_type == 1:
        formatted_string = formatted_string.replace("jazz", "normal").replace("hiphop", "murmur")

    return {"message": formatted_string}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

