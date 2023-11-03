import re
import subprocess
import os
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import csv
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from io import BytesIO
from starlette.responses import FileResponse
import torch
import shutil

app = FastAPI()

# Load the model outside of the endpoint to avoid loading it multiple times
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best50.pt', trust_repo=True, force_reload=True)


# Define a list of safe modules and functions
SAFE_MODULES = ['math', 'random']
SAFE_FUNCTIONS = ['print', 'len']

# Set resource limits
MEMORY_LIMIT = 256  # Memory limit in MB
CPU_LIMIT = 5  # CPU time limit in seconds

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
async def privacy(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train")
async def upload_data(
    image: UploadFile = Form(...),
    name: str = Form(...),
    status: str = Form(...),
    timestamp: str = Form(...),
    point1_x: str = Form(...),
    point1_y: str = Form(...),
    point2_x: str = Form(...),
    point2_y: str = Form(...)
):
    # Directories
    BASE_DIR = "user_collected"
    IMAGE_DIR = f"{BASE_DIR}/images"
    LABEL_DIR = f"{BASE_DIR}/labels"
    directory_path = "./user_collected/images"
    extension = "jpg"
    number_of_files = len([f for f in os.listdir(directory_path) if f.endswith(extension) and os.path.isfile(os.path.join(directory_path, f))])

    # Save the uploaded image
    img_path = f"{IMAGE_DIR}/img_{number_of_files}.jpg"
    with open(img_path, "wb") as buffer:
        buffer.write(image.file.read())

    # Convert non-jpg images to jpg
    img_format = Image.open(img_path).format
    if img_format != "JPEG":
        image_obj = Image.open(img_path)
        rgb_im = image_obj.convert('RGB')
        rgb_im.save(img_path, "JPEG")

    # Save label data
    label_data = f"{timestamp} {status} {name} {point1_x} {point1_y} {point2_x} {point2_y}\n"
    with open(f"{LABEL_DIR}/{number_of_files}.txt", "w") as label_file:
        label_file.write(label_data)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

