from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from ConvNet import ConvNet


DEVICE = torch.device("cpu")
CHECKPOINT_PATH = Path(__file__).resolve().parent / "model.pth"
CLASS_NAMES = {
    0: "No screw",
    1: "Flat-head 1.5 cm",
    2: "Round-head 2.5 cm",
    3: "Flat-head 3.0 cm",
    4: "Flat-head 3.5 cm",
    5: "Flat-head 6.0 cm",
    6: "Round-head 7.5 cm",
}
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

# The checkpoint contains the full model state, including the ResNet backbone.
MODEL = ConvNet(num_classes=len(CLASS_NAMES), pretrained=False).to(DEVICE)
MODEL.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
)
MODEL.eval()


def model_predict(image: Image.Image) -> str:
    """Preprocess one uploaded image and return its predicted class name."""
    image_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(image_tensor)
        predicted_class = logits.argmax(dim=1).item()

    return CLASS_NAMES[predicted_class]


demo = gr.Interface(
    fn=model_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Screw Prediction",
    description="Upload an image to classify the screw.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
