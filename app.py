import gradio as gr
import torch
import cv2
import PostProcess
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torchmodel = torch.jit.load("best.torchscript", map_location=device)
torchmodel.eval()

def preprocess_image(image_ori) -> torch.Tensor:
    image = cv2.resize(image_ori, (640, 640))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image_rgb

def run_model(image):
    start_time = time.time()
    result = torchmodel(image)[0]
    end_time = time.time()
    return result, (end_time - start_time) * 1000

def run_video():
    cap = cv2.VideoCapture("evaluation.mp4")
    postprocessor = PostProcess.PostProcessor()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            image, image_rgb = preprocess_image(frame)
            image = image.to(device)

            result, duration_ms = run_model(image)

            postprocessor.set_image(image_rgb)
            postprocessor.set_time(duration_ms)
            postprocessor.set_result(result)

            output_frame = postprocessor.get_frame()

            # Convert BGR → RGB for Gradio
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            yield output_frame

    cap.release()


demo = gr.Interface(
    fn=run_video,
    inputs=[],
    outputs=gr.Image(streaming=True),
    live=True
)

demo.launch()