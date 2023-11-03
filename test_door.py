import torch
from PIL import Image

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

def process_image(image_path):
    # Load image
    img = Image.open(image_path)
    img = img.resize((384, 384))
    # Perform inference
    results = model(img)
    
    # Extract the PIL image with boxes and labels
    pil_img = results.render()[0]  # results.render() returns a list of images
    annotated_img = Image.fromarray(pil_img)

    # Save the annotated image
    annotated_img.save('result.jpg')

if __name__ == '__main__':
    process_image('odoor.jpeg')

