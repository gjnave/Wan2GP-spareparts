import gradio as gr
import os
import sys
from huggingface_hub import hf_hub_download
import torch
from mmgp import offload

# Add the GGF directory to the Python path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.wan.any2video import WanAny2V
from shared.utils import files_locator as fl
from shared.utils.audio_video import save_video

# Model configuration from t2v_fusionix.json
model_config = {
    "name": "Wan2.1 Text2video FusioniX 14B",
    "architecture": "t2v",
    "description": "A powerful merged text-to-video model based on the original WAN 2.1 T2V model, enhanced using multiple open-source components and LoRAs to boost motion realism, temporal consistency, and expressive detail.",
    "URLs": [
        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_fp16.safetensors",
        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_fp16_int8.safetensors",
        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_bf16_int8.safetensors"
    ],
    "auto_quantize": True
}

# --- Model Loading and Downloading ---
def download_file(repo_id, filename, subfolder=None):
    """Downloads a file from Hugging Face if it's not already downloaded."""
    local_dir = "."
    if subfolder:
        local_dir = os.path.join(local_dir, subfolder)

    if not os.path.exists(os.path.join(local_dir, os.path.basename(filename))):
        print(f"Downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, subfolder=subfolder)
        print("Download complete.")

def load_video_model():
    """Loads the video generation model."""
    print("Starting model load...")

    # Download all necessary model files
    print("Downloading main model...")
    download_file("DeepBeepMeep/Wan2.1", "Wan14BT2VFusioniX_fp16.safetensors")
    print("Downloading text encoder...")
    download_file("DeepBeepMeep/Wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors", subfolder="umt5-xxl")
    print("Downloading VAE...")
    download_file("DeepBeepMeep/Wan2.1", "Wan2.1_VAE.safetensors")
    print("Downloading CLIP model...")
    download_file("DeepBeepMeep/Wan2.1", "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", subfolder="xlm-roberta-large")

    fl.set_checkpoints_paths([".", "GGF", "GGF/models"])

    from models.wan.configs import WAN_CONFIGS
    cfg = WAN_CONFIGS['t2v-14B']

    print("Instantiating WanAny2V...")
    wan_model = WanAny2V(
        config=cfg,
        checkpoint_dir=".",
        model_filename=["Wan14BT2VFusioniX_fp16.safetensors"],
        submodel_no_list=[1],
        model_type="t2v_fusionix",
        model_def=model_config,
        base_model_type="t2v",
        text_encoder_filename="umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors",
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False
    )
    print("WanAny2V instantiated.")

    pipe = {"transformer": wan_model.model, "text_encoder": wan_model.text_encoder.model, "vae": wan_model.vae.model}
    print("Profiling with offload...")
    offload.profile(pipe, profile_no=4)
    print("Profiling complete.")

    return wan_model

# --- Video Generation ---
def generate_video_for_gradio(prompt):
    """Generates a video based on the user's prompt."""
    global video_model
    if video_model is None:
        print("Model not loaded. Loading now...")
        video_model = load_video_model()
        print("Model loaded.")

    # Simplified generation logic
    print(f"Generating video for prompt: {prompt}")
    output_tensor, _ = video_model.generate(
        input_prompt=prompt,
        frame_num=16,
        sampling_steps=20,
        guide_scale=7.5,
    )

    # The output from generate is a tensor, we need to save it to a file
    video_path = "output.mp4"
    print(f"Saving video to {video_path}...")
    save_video(output_tensor.cpu(), video_path)
    print("Video saved.")
    return video_path

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Get Going Fast - FusionX")
    gr.Markdown("### For those who just want to get going fast")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt")
            generate_button = gr.Button("Generate")

    with gr.Row():
        video_output = gr.Video()

    generate_button.click(
        fn=generate_video_for_gradio,
        inputs=prompt_input,
        outputs=video_output,
    )

if __name__ == "__main__":
    video_model = None
    print("Launching Gradio app...")
    demo.launch()
