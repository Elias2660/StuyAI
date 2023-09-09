import re
import subprocess
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import csv

app = FastAPI()

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

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

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

