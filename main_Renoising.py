import os
import argparse
import json
import glob
import subprocess
import imageio
import imageio_ffmpeg
import numpy as np
import torch
import torchvision.transforms as transforms

from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from imap.video_utils import make_saliency_map_video, make_individual_videos

model_dict = {
    "2b": "THUDM/CogVideoX-2b",
    "5b": "THUDM/CogVideoX-5b",
    "hunyuan": "hunyuanvideo-community/HunyuanVideo",
    "1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
}

def load_model(model_id: str, device: str):
    assert model_id in model_dict, f"Model ID {model_id} not recognized."
    model_path = model_dict[model_id]
    dtype = torch.bfloat16
    
    if "CogVideoX" in model_path:
        from imap.cogvideox.modified_dit import ModifiedCogVideoXTransformer3DModel
        from imap.cogvideox.pipeline import RenoiseCogVideoXPipeline
        
        transformer = ModifiedCogVideoXTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype
        )
        pipe = RenoiseCogVideoXPipeline.from_pretrained(
            model_path, 
            transformer=transformer,
            torch_dtype=dtype
        ).to(device)
        negative_prompt = ""

    elif "HunyuanVideo" in model_path:
        from imap.hunyuanvideo.modified_dit import ModifiedHunyuanVideoTransformer3DModel
        from imap.hunyuanvideo.pipeline import RenoiseHunyuanVideoPipeline
        
        transformer = ModifiedHunyuanVideoTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype
        )
        pipe = RenoiseHunyuanVideoPipeline.from_pretrained(
            model_path, 
            transformer=transformer,
            torch_dtype=dtype
        ).to(device)
        negative_prompt = ""

    elif "Wan2.1" in model_path:
        from imap.wan.modified_dit import ModifiedWanTransformer3DModel
        from imap.wan.pipeline import RenoiseWanPipeline
        
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        transformer = ModifiedWanTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype
        )
        pipe = RenoiseWanPipeline.from_pretrained(
            model_path, 
            transformer=transformer,
            vae=vae, 
            torch_dtype=dtype
        ).to(device)
        pipe.scheduler = scheduler
        # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        negative_prompt = ""
        
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe, negative_prompt

def encode_video(pipe, video_path: str, device: str, model_id: str):
    model_name = model_dict[model_id]
    if "CogVideoX" in model_name or "HunyuanVideo" in model_name:
        dtype = torch.bfloat16
    elif "Wan2.1" in model_name:
        dtype = torch.float32
    
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).unsqueeze(0).to(dtype)
    frames_tensor = pipe.video_processor.preprocess_video(frames_tensor)
    
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
    if "Wan2.1" in model_name or "HunyuanVideo" in model_name:
        return encoded_frames
    return pipe.vae_scaling_factor_image * encoded_frames

def set_inputs(args):
    input_dicts = []

    with open(args.input_json, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
        for item in data:
            video_path = item.get("video_path", None)
            prompt = item.get("caption", "")
            concepts_s = item.get("concepts", ", ")
            concepts_t = item.get("object", ", ")
            concepts = []
            if args.concept_option in ["s", "st"]:
                concepts.extend(concepts_s.split(", "))
            if args.concept_option in ["t", "st"]:
                concepts.extend(concepts_t.split(", "))
            input_dict = {
                "video_path": video_path,
                "prompt": prompt,
                "concepts": concepts
            }
            input_dicts.append(input_dict)
    return input_dicts

# def mp4_to_gif(mp4_path: str, fps: int):
#     gif_path = os.path.splitext(mp4_path)[0] + ".gif"
#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-i",
#             mp4_path,
#             "-vf",
#             f"fps={fps},scale=iw:-1:flags=lanczos",
#             "-loop",
#             "0",
#             gif_path,
#         ],
#         check=True,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )
    
def mp4_to_gif(mp4_path: str, fps: int):
    gif_path = os.path.splitext(mp4_path)[0] + ".gif"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            mp4_path,
            "-vf",
            f"fps={fps},scale=iw:-1:flags=lanczos",
            "-loop",
            "0",
            gif_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
def save_gif(i, input_dict, fps=8, fps_latent=2):
    mp4_files = set()
    mp4_files.update(glob.glob(os.path.join(args.output_dir, f"{i:03d}-{input_dict['prompt'][:20]}", "*.mp4")))
    if len(input_dict["concepts"]) > 0:
        mp4_files.update(glob.glob(os.path.join(args.output_dir, f"{i:03d}-{input_dict['prompt'][:20]}", "cross_attentions", "*.mp4")))
    for mp4 in sorted(mp4_files):
        fps_for_gif = fps if os.path.basename(mp4) == "output.mp4" else fps_latent
        mp4_to_gif(mp4, fps_for_gif)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe, negative_prompt = load_model(args.model_id, device=device)
    
    maps_timestep_range = list(range(args.maps_timesteps[0], args.maps_timesteps[1]))
    maps_layer_range = list(range(args.maps_layers[0], args.maps_layers[1]))

    input_dicts = set_inputs(args)
        
    for i, input_dict in enumerate(input_dicts):
        
        # 0. make output directory
        sample_dir = f"{args.output_dir}/{i:03d}-{input_dict['prompt'][:20]}"
        save_video_path = f"{sample_dir}/output.mp4"
        os.makedirs(f"{sample_dir}", exist_ok=True)
        
        # 1. check if output video already exists
        if os.path.exists(save_video_path.replace(".mp4", ".gif")):
            print(f"[SKIP] Sample {i} already processed, skipping...")
            continue
        
        # 2. video encoding
        encoded_video = encode_video(pipe, input_dict["video_path"], device=device, model_id=args.model_id)


        # 3. save args + sample-level metadata
        meta = dict(vars(args))
        meta.update({
            "video_path": input_dict.get("video_path"),
            "prompt": input_dict.get("prompt"),
            "concepts": input_dict.get("concepts", []),
            "renoise_timestep": args.renoise_timestep,
            "test_full_denoise": args.test_full_denoise,
        })
        with open(os.path.join(sample_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        
        # 3-1. prompt tokenization
        tokens = pipe.tokenizer.tokenize(input_dict["prompt"])
        tokens.append('<EOS>')
        
        # 4. inference
        print(f"Processing sample {i}: {input_dict['prompt'][:50]}...")
        video, maps_dict = pipe(
            encoded_video=encoded_video,
            renoise_timestep=args.renoise_timestep,
            test_full_denoise=args.test_full_denoise,
            prompt=input_dict["prompt"],
            negative_prompt=negative_prompt,
            concepts=input_dict["concepts"],
            guidance_scale=args.guidance_scale,
            # use_dynamic_cfg=True, 
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            saliency_map_kwargs={
                "tokens": tokens,
                "except_softmax": args.maps_except_softmax,
                "timesteps": maps_timestep_range,
                "layers": maps_layer_range,
                "concept_avg_until_eos": args.concept_avg_until_eos,
            },
            imap_layer=args.imap_layer,
            imap_sep_score=args.imap_sep_score,
            imap_sep_topk=args.imap_sep_topk,
            imap_qk_matching_target=args.imap_qk_matching_target,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
        video = video.frames[0]

        # 5. save videos & concept attention maps
        if len(input_dict["concepts"]) > 0:
            for key in maps_dict:
                if key.startswith("_"):
                    continue
                maps = maps_dict[key]
                if "imap" in key and args.imap_qk_matching_target == "prompt":
                    text_lists = tokens
                elif args.model_id in ["1.3b", "14b"] and args.imap_qk_matching_target == "prompt":
                    text_lists = tokens
                else:
                    text_lists = input_dict["concepts"]
                
                # make_saliency_map_video(text_lists, maps,
                #                              save_path=f"{sample_dir}/{key}.mp4", color_map="plasma")
                # os.makedirs(f"{sample_dir}/{key}", exist_ok=True)
                # make_individual_videos(text_lists, maps,
                #                        save_dir=f"{sample_dir}/{key}", fps=args.fps_latent, color_map="plasma")
                
                os.makedirs(f"{sample_dir}/npy", exist_ok=True)
                np.save(f"{sample_dir}/npy/{key}.npy", maps.float().cpu().numpy().astype(np.float32))

            export_to_video(video, 
                            save_video_path, fps=args.fps)
        else:
            print("No concepts provided, skipping concept attention visualization.")
            export_to_video(video, save_video_path, fps=args.fps)
        print(f"Saved MP4 for sample {i}.")

        # save gif
        save_gif(i, input_dict, fps=args.fps, fps_latent=args.fps_latent)
        print(f"Saved GIF for sample {i}.")
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes","y","true","t","1"):
        return True
    if v in ("no","n","false","f","0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean string (true/false).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="5b", help="Model ID to use.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file containing prompts and concepts.")
    parser.add_argument("--renoise_timestep", type=int, nargs="+", default=[20], help="Timesteps to apply Renoise denoising.")
    parser.add_argument("--test_full_denoise", type=str2bool, nargs='?', const=True, default=False, help="Whether to test full denoising (no Renoise).")
    
    parser.add_argument("--concept_option", type=str, default="st", help="Which concept option to use: s, t, or st.")
    parser.add_argument("--height", type=int, default=480, help="Height of the generated video.")
    parser.add_argument("--width", type=int, default=720, help="Width of the generated video.")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video.")
    parser.add_argument("--fps_latent", type=int, default=1, help="Frames per second for the latent video.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--maps_except_softmax", type=str2bool, nargs='?', const=True, default=False, help="Whether to apply softmax to concept attention maps.")
    parser.add_argument("--maps_timesteps", type=int, nargs="+", default=[0, 50], help="Timesteps to extract concept attention maps.")
    parser.add_argument("--maps_layers", type=int, nargs="+", default=[0, 30], help="Layers to extract concept attention maps.")
    parser.add_argument("--concept_avg_until_eos", type=str2bool, nargs='?', const=True, default=False, help="Whether to average concept attention maps until the first EOS token.")

    parser.add_argument("--imap_layer", type=int, nargs="+", required=True, help="Layers to apply IMAP.")
    parser.add_argument("--imap_sep_score", type=str, default="CHI", choices=["Silhouette", "DBI", "CHI", "Fisher"], help="Separation score method for IMAP")
    parser.add_argument("--imap_sep_topk", type=int, default=5, help="Top-K heads to select in IMAP")
    parser.add_argument("--imap_qk_matching_target", type=str, default="concepts", choices=["prompt", "concepts"], help="Target to use for QK_matching in IMAP.")
    args = parser.parse_args()

    main(args)
    