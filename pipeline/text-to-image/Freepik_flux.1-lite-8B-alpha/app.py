import torch
from diffusers import FluxPipeline

base_model_id = "Freepik/flux.1-lite-8B-alpha"
torch_dtype = torch.bfloat16
device = "cuda"
cache_dir = "fine_tuned_models/text-to-image/Freepik_flux.1-lite-8B-alpha/base_model"

# Load the pipe
model_id = "Freepik/flux.1-lite-8B-alpha"
pipe = FluxPipeline.from_pretrained(
    model_id, torch_dtype=torch_dtype,cache_dir=cache_dir
).to(device)

# Inference
prompt = "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"

guidance_scale = 3.5
n_steps = 28
seed = 11

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        height=1024,
        width=1024,
    ).images[0]
image.save("output.png")
