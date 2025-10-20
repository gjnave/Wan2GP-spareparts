import os
os.environ["GRADIO_LANG"] = "en"
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)    
import time
import sys
import threading
import argparse
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
import random
import json
import numpy as np
import importlib
from shared.utils import notification_sound
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask
from shared.utils.utils import calculate_new_dimensions, get_outpainting_frame_location, get_outpainting_full_area_dimensions
from shared.utils.utils import has_video_file_extension, has_image_file_extension
from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, combine_and_concatenate_video_with_audio_tracks, cleanup_temp_audio_files,  save_video, save_image
from shared.utils.audio_video import save_image_metadata, read_image_metadata
from shared.match_archi import match_nvidia_architecture
from shared.attention import get_attention_modes, get_supported_attention_modes
from huggingface_hub import hf_hub_download, snapshot_download
from shared.utils import files_locator as fl 
import torch
import gc
import traceback
import math 
import typing
import asyncio
import inspect
from shared.utils import prompt_parser
import base64
import io
from PIL import Image
import zipfile
import tempfile
import atexit
import shutil
import glob
import cv2
from transformers.utils import logging
logging.set_verbosity_error
from preprocessing.matanyone  import app as matanyone_app
from tqdm import tqdm
import requests
from shared.gradio.gallery import AdvancedMediaGallery
from collections import defaultdict

# import torch._dynamo as dynamo
# dynamo.config.recompile_limit = 2000   # default is 256
# dynamo.config.accumulated_recompile_limit = 2000  # or whatever limit you want

global_queue_ref = []
AUTOSAVE_FILENAME = "queue.zip"
PROMPT_VARS_MAX = 10
target_mmgp_version = "3.6.3"
WanGP_version = "8.996"
settings_version = 2.39
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None

from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
vmc_event_handler = matanyone_app.get_vmc_event_handler()
unique_id = 0
unique_id_lock = threading.Lock()
gen_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
reload_needed = True

def clear_gen_cache():
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]

def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None    
    clear_gen_cache()
    offload.shared_state
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            torch._C._host_emptyCache()
        except:
            pass
        reload_needed = True
    else:
        gc.collect()

def get_unique_id():
    global unique_id  
    with unique_id_lock:
        unique_id += 1
    return str(time.time()+unique_id)

def download_ffmpeg():
    if os.name != 'nt': return
    exes = ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']
    if all(os.path.exists(e) for e in exes): return
    api_url = 'https://api.github.com/repos/GyanD/codexffmpeg/releases/latest'
    r = requests.get(api_url, headers={'Accept': 'application/vnd.github+json'})
    assets = r.json().get('assets', [])
    zip_asset = next((a for a in assets if 'essentials_build.zip' in a['name']), None)
    if not zip_asset: return
    zip_url = zip_asset['browser_download_url']
    zip_name = zip_asset['name']
    with requests.get(zip_url, stream=True) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        with open(zip_name, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    with zipfile.ZipFile(zip_name) as z:
        for f in z.namelist():
            if f.endswith(tuple(exes)) and '/bin/' in f:
                z.extract(f)
                os.rename(f, os.path.basename(f))
    os.remove(zip_name)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{seconds:.1f}s"

def format_generation_time(seconds):
    """Format generation time showing raw seconds with human-readable time in parentheses when over 60s"""
    raw_seconds = f"{int(seconds)}s"
    
    if seconds < 60:
        return raw_seconds
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        human_readable = f"{hours}h {minutes}m {secs}s"
    else:
        human_readable = f"{minutes}m {secs}s"
    
    return f"{raw_seconds} ({human_readable})"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None

    if isinstance(pil_image, str):
        from shared.utils.utils import get_video_frame
        pil_image = get_video_frame(pil_image, 0)

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def compute_sliding_window_no(current_video_length, sliding_window_size, discard_last_frames, reuse_frames):
    left_after_first_window = current_video_length - sliding_window_size + discard_last_frames
    return 1 + math.ceil(left_after_first_window / (sliding_window_size - discard_last_frames - reuse_frames))


def clean_image_list(gradio_list):
    if not isinstance(gradio_list, list): gradio_list = [gradio_list]
    gradio_list = [ tup[0] if isinstance(tup, tuple) else tup for tup in gradio_list ]        

    if any( not isinstance(image, (Image.Image, str))  for image in gradio_list): return None
    if any( isinstance(image, str) and not has_image_file_extension(image) for image in gradio_list): return None
    gradio_list = [ convert_image( Image.open(img) if isinstance(img, str) else img  ) for img in gradio_list  ]        
    return gradio_list



def process_prompt_and_add_tasks(state, model_choice):
    def ret():
        return gr.update(), gr.update()
    
    if state.get("validate_success",0) != 1:
        ret()
    
    state["validate_success"] = 0
    model_filename = state["model_filename"]
    model_type = state["model_type"]
    inputs = get_model_settings(state, model_type)

    if model_choice != model_type or inputs ==None:
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
    
    inputs["state"] =  state
    gen = get_gen_info(state)
    inputs["model_type"] = model_type
    inputs.pop("lset_name")
    if inputs == None:
        gr.Warning("Internal state error: Could not retrieve inputs for the model.")
        queue = gen.get("queue", [])
        return ret()
    model_def = get_model_def(model_type)
    model_handler = get_model_handler(model_type)
    image_outputs = inputs["image_mode"] > 0
    any_steps_skipping = model_def.get("tea_cache", False) or model_def.get("mag_cache", False)
    model_type = get_base_model_type(model_type)
    inputs["model_filename"] = model_filename
    
    mode = inputs["mode"]
    if mode.startswith("edit_"):
        edit_video_source =gen.get("edit_video_source", None)
        edit_overrides =gen.get("edit_overrides", None)
        _ , _ , _, frames_count = get_video_info(edit_video_source)
        if frames_count > max_source_video_frames:
            gr.Info(f"Post processing is not supported on videos longer than {max_source_video_frames} frames. Output Video will be truncated")
            # return
        for k in ["image_start", "image_end", "image_refs", "video_guide", "audio_guide", "audio_guide2", "audio_source" , "video_mask", "image_mask"]:
            inputs[k] = None    
        inputs.update(edit_overrides)
        del gen["edit_video_source"], gen["edit_overrides"]
        inputs["video_source"]= edit_video_source 
        prompt = []

        spatial_upsampling = inputs.get("spatial_upsampling","")
        if len(spatial_upsampling) >0: prompt += ["Spatial Upsampling"]
        temporal_upsampling = inputs.get("temporal_upsampling","")
        if len(temporal_upsampling) >0: prompt += ["Temporal Upsampling"]
        if has_image_file_extension(edit_video_source)  and len(temporal_upsampling) > 0:
            gr.Info("Temporal Upsampling can not be used with an Image")
            return ret()
        film_grain_intensity  = inputs.get("film_grain_intensity",0)
        film_grain_saturation  = inputs.get("film_grain_saturation",0.5)        
        # if film_grain_intensity >0: prompt += [f"Film Grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}"]
        if film_grain_intensity >0: prompt += ["Film Grain"]
        MMAudio_setting = inputs.get("MMAudio_setting",0)
        repeat_generation= inputs.get("repeat_generation",1)
        if mode =="edit_remux":
            audio_source = inputs["audio_source"]
            if  MMAudio_setting== 1:
                prompt += ["MMAudio"]
                audio_source = None 
                inputs["audio_source"] = audio_source
            else:
                if audio_source is None:
                    gr.Info("You must provide a custom Audio")
                    return ret()
                prompt += ["Custom Audio"]
                repeat_generation == 1

        seed = inputs.get("seed",None)
        if len(prompt) == 0:
            if mode=="edit_remux":
                gr.Info("You must choose at least one Remux Method")
            else:
                gr.Info("You must choose at least one Post Processing Method")
            return ret()
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        return ret()

    if hasattr(model_handler, "validate_generative_settings"):
        error = model_handler.validate_generative_settings(model_type, model_def, inputs)
        if error is not None and len(error) > 0:
            gr.Info(error)
            return ret()
    if inputs.get("cfg_star_switch", 0) != 0 and inputs.get("apg_switch", 0) != 0:
        gr.Info("Adaptive Progressive Guidance and Classifier Free Guidance Star can not be set at the same time")
        return ret()
    prompt = inputs["prompt"]
    if len(prompt) ==0:
        gr.Info("Prompt cannot be empty.")
        gen = get_gen_info(state)
        queue = gen.get("queue", [])
        return ret()
    prompt, errors = prompt_parser.process_template(prompt)
    if len(errors) > 0:
        gr.Info("Error processing prompt template: " + errors)
        return ret()
    model_filename = get_model_filename(model_type)  
    prompts = prompt.replace("\r", "").split("\n")
    prompts = [prompt.strip() for prompt in prompts if len(prompt.strip())>0 and not prompt.startswith("#")]
    if len(prompts) == 0:
        gr.Info("Prompt cannot be empty.")
        gen = get_gen_info(state)
        queue = gen.get("queue", [])
        return ret()

    resolution = inputs["resolution"]
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    image_start = inputs["image_start"]
    image_end = inputs["image_end"]
    image_refs = inputs["image_refs"]
    image_prompt_type = inputs["image_prompt_type"]
    audio_prompt_type = inputs["audio_prompt_type"]
    if image_prompt_type == None: image_prompt_type = ""
    video_prompt_type = inputs["video_prompt_type"]
    if video_prompt_type == None: video_prompt_type = ""
    force_fps = inputs["force_fps"]
    audio_guide = inputs["audio_guide"]
    audio_guide2 = inputs["audio_guide2"]
    audio_source = inputs["audio_source"]
    video_guide = inputs["video_guide"]
    image_guide = inputs["image_guide"]
    video_mask = inputs["video_mask"]
    image_mask = inputs["image_mask"]
    speakers_locations = inputs["speakers_locations"]
    video_source = inputs["video_source"]
    frames_positions = inputs["frames_positions"]
    keep_frames_video_guide= inputs["keep_frames_video_guide"] 
    keep_frames_video_source = inputs["keep_frames_video_source"]
    denoising_strength= inputs["denoising_strength"]     
    sliding_window_size = inputs["sliding_window_size"]
    sliding_window_overlap = inputs["sliding_window_overlap"]
    sliding_window_discard_last_frames = inputs["sliding_window_discard_last_frames"]
    video_length = inputs["video_length"]
    num_inference_steps= inputs["num_inference_steps"]
    skip_steps_cache_type= inputs["skip_steps_cache_type"]
    MMAudio_setting = inputs["MMAudio_setting"]
    image_mode = inputs["image_mode"]
    switch_threshold = inputs["switch_threshold"]
    loras_multipliers = inputs["loras_multipliers"]
    activated_loras = inputs["activated_loras"]
    guidance_phases= inputs["guidance_phases"]
    model_switch_phase = inputs["model_switch_phase"]    
    switch_threshold = inputs["switch_threshold"]
    switch_threshold2 = inputs["switch_threshold2"]
    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    video_guide_outpainting = inputs["video_guide_outpainting"]

    outpainting_dims = get_outpainting_dims(video_guide_outpainting)

    if server_config.get("fit_canvas", 0) == 2 and outpainting_dims is not None and any_letters(video_prompt_type, "VKF"):
        gr.Info("Output Resolution Cropping will be not used for this Generation as it is not compatible with Video Outpainting")

    if len(activated_loras) > 0:
        error = check_loras_exist(model_type, activated_loras)
        if len(error) > 0:
            gr.Info(error)
            return ret()
        
    if len(loras_multipliers) > 0:
        _, _, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases= guidance_phases)
        if len(errors) > 0: 
            gr.Info(f"Error parsing Loras Multipliers: {errors}")
            return ret()
    if guidance_phases == 3:
        if switch_threshold < switch_threshold2:
            gr.Info(f"Phase 1-2 Switch Noise Level ({switch_threshold}) should be Greater than Phase 2-3 Switch Noise Level ({switch_threshold2}). As a reminder, noise will gradually go down from 1000 to 0.")
            return ret()
    else:
        model_switch_phase = 1
        
    if not any_steps_skipping: skip_steps_cache_type = ""
    if not model_def.get("lock_inference_steps", False) and model_type in ["ltxv_13B"] and num_inference_steps < 20:
        gr.Info("The minimum number of steps should be 20") 
        return ret()
    if skip_steps_cache_type == "mag":
        if num_inference_steps > 50:
            gr.Info("Mag Cache maximum number of steps is 50")
            return ret()
        
    if image_mode > 0:
        audio_prompt_type = ""

    if "B" in audio_prompt_type or "X" in audio_prompt_type:
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if len(error) > 0:
            gr.Info(error)
            return ret()

    if MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and video_length <16: #should depend on the architecture
        gr.Info("MMAudio can generate an Audio track only if the Video is at least 1s long")
    if "F" in video_prompt_type:
        if len(frames_positions.strip()) > 0:
            positions = frames_positions.replace(","," ").split(" ")
            for pos_str in positions:
                if not pos_str in ["L", "l"] and len(pos_str)>0: 
                    if not is_integer(pos_str):
                        gr.Info(f"Invalid Frame Position '{pos_str}'")
                        return ret()
                    pos = int(pos_str)
                    if pos <1 or pos > max_source_video_frames:
                        gr.Info(f"Invalid Frame Position Value'{pos_str}'")
                        return ret()
    else:
        frames_positions = None

    if audio_source is not None and MMAudio_setting != 0:
        gr.Info("MMAudio and Custom Audio Soundtrack can't not be used at the same time")
        return ret()
    if len(filter_letters(image_prompt_type, "VLG")) > 0 and len(keep_frames_video_source) > 0:
        if not is_integer(keep_frames_video_source) or int(keep_frames_video_source) == 0:
            gr.Info("The number of frames to keep must be a non null integer") 
            return ret()
    else:
        keep_frames_video_source = ""

    if image_outputs:
        image_prompt_type = image_prompt_type.replace("V", "").replace("L", "")

    if "V" in image_prompt_type:
        if video_source == None:
            gr.Info("You must provide a Source Video file to continue")
            return ret()
    else:
        video_source = None

    if "A" in audio_prompt_type:
        if audio_guide == None:
            gr.Info("You must provide an Audio Source")
            return ret()
        if "B" in audio_prompt_type:
            if audio_guide2 == None:
                gr.Info("You must provide a second Audio Source")
                return ret()
        else:
            audio_guide2 = None
    else:
        audio_guide = None
        audio_guide2 = None
        
    if model_type in ["vace_multitalk_14B"] and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        if not "I" in video_prompt_type and not not "V" in video_prompt_type:
            gr.Info("To get good results with Multitalk and two people speaking, it is recommended to set a Reference Frame or a Control Video (potentially truncated) that contains the two people one on each side")

    if model_def.get("one_image_ref_needed", False):
        if image_refs  == None :
            gr.Info("You must provide an Image Reference") 
            return ret()
        if len(image_refs) > 1:
            gr.Info("Only one Image Reference (a person) is supported for the moment by this model") 
            return ret()
    if model_def.get("at_least_one_image_ref_needed", False):
        if image_refs  == None :
            gr.Info("You must provide at least one Image Reference") 
            return ret()
        
    if "I" in video_prompt_type:
        if image_refs == None or len(image_refs) == 0:
            gr.Info("You must provide at least one Reference Image")
            return ret()
        image_refs = clean_image_list(image_refs)
        if image_refs == None :
            gr.Info("A Reference Image should be an Image") 
            return ret()
    else:
        image_refs = None

    if "V" in video_prompt_type:
        if image_outputs:
            if image_guide is None:
                gr.Info("You must provide a Control Image")
                return ret()
        else:
            if video_guide is None:
                gr.Info("You must provide a Control Video")
                return ret()
        if "A" in video_prompt_type and not "U" in video_prompt_type:             
            if image_outputs:
                if image_mask is None:
                    gr.Info("You must provide a Image Mask")
                    return ret()
            else:
                if video_mask is None:
                    gr.Info("You must provide a Video Mask")
                    return ret()
        else:
            video_mask = None
            image_mask = None

        if "G" in video_prompt_type:
                if denoising_strength < 1.:
                    gr.Info(f"With Denoising Strength {denoising_strength:.1f}, denoising will start at Step no {int(round(num_inference_steps * (1. - denoising_strength),4))} ")
        else: 
            denoising_strength = 1.0
        if len(keep_frames_video_guide) > 0 and model_type in ["ltxv_13B"]:
            gr.Info("Keep Frames for Control Video is not supported with LTX Video")
            return ret()
        _, error = parse_keep_frames_video_guide(keep_frames_video_guide, video_length)
        if len(error) > 0:
            gr.Info(f"Invalid Keep Frames property: {error}")
            return ret()
    else:
        video_guide = None
        image_guide = None
        video_mask = None
        image_mask = None
        keep_frames_video_guide = ""
        denoising_strength = 1.0
    
    if image_outputs:
        video_guide = None
        video_mask = None
    else:
        image_guide = None
        image_mask = None


    if "S" in image_prompt_type:
        if image_start == None or isinstance(image_start, list) and len(image_start) == 0:
            gr.Info("You must provide a Start Image")
            return ret()
        image_start = clean_image_list(image_start)        
        if image_start == None :
            gr.Info("Start Image should be an Image") 
            return ret()
        if  multi_prompts_gen_type == 1 and len(image_start) > 1:
            gr.Info("Only one Start Image is supported") 
            return ret()       
    else:
        image_start = None

    if not any_letters(image_prompt_type, "SVL"):
        image_prompt_type = image_prompt_type.replace("E", "")
    if "E" in image_prompt_type:
        if image_end == None or isinstance(image_end, list) and len(image_end) == 0:
            gr.Info("You must provide an End Image") 
            return ret()
        image_end = clean_image_list(image_end)        
        if image_end == None :
            gr.Info("End Image should be an Image") 
            return ret()
        if multi_prompts_gen_type == 0:
            if video_source is not None:
                if len(image_end)> 1:
                    gr.Info("If a Video is to be continued and the option 'Each Text Prompt Will create a new generated Video' is set, there can be only one End Image")
                    return ret()        
            elif len(image_start or []) != len(image_end or []):
                gr.Info("The number of Start and End Images should be the same when the option 'Each Text Prompt Will create a new generated Video'")
                return ret()    
    else:        
        image_end = None


    if test_any_sliding_window(model_type) and image_mode == 0:
        if video_length > sliding_window_size:
            if model_type in ["t2v", "t2v_2_2"] and not "G" in video_prompt_type :
                gr.Info(f"You have requested to Generate Sliding Windows with a Text to Video model. Unless you use the Video to Video feature this is useless as a t2v model doesn't see past frames and it will generate the same video in each new window.") 
                return ret()
            full_video_length = video_length if video_source is None else video_length +  sliding_window_overlap -1
            extra = "" if full_video_length == video_length else f" including {sliding_window_overlap} added for Video Continuation"
            no_windows = compute_sliding_window_no(full_video_length, sliding_window_size, sliding_window_discard_last_frames, sliding_window_overlap)
            gr.Info(f"The Number of Frames to generate ({video_length}{extra}) is greater than the Sliding Window Size ({sliding_window_size}), {no_windows} Windows will be generated")
    if "recam" in model_filename:
        if video_guide == None:
            gr.Info("You must provide a Control Video")
            return ret()
        computed_fps = get_computed_fps(force_fps, model_type , video_guide, video_source )
        frames = get_resampled_video(video_guide, 0, 81, computed_fps)
        if len(frames)<81:
            gr.Info(f"Recammaster Control video should be at least 81 frames once the resampling at {computed_fps} fps has been done")
            return ret()

    if "hunyuan_custom_custom_edit" in model_filename:
        if len(keep_frames_video_guide) > 0: 
            gr.Info("Filtering Frames with this model is not supported")
            return ret()

    if inputs["multi_prompts_gen_type"] != 0:
        if image_start != None and len(image_start) > 1:
            gr.Info("Only one Start Image must be provided if multiple prompts are used for different windows") 
            return ret()

        # if image_end != None and len(image_end) > 1:
        #     gr.Info("Only one End Image must be provided if multiple prompts are used for different windows") 
        #     return

    override_inputs = {
        "image_start": image_start[0] if image_start !=None and len(image_start) > 0 else None,
        "image_end": image_end, #[0] if image_end !=None and len(image_end) > 0 else None,
        "image_refs": image_refs,
        "audio_guide": audio_guide,
        "audio_guide2": audio_guide2,
        "audio_source": audio_source,
        "video_guide": video_guide,
        "image_guide": image_guide,
        "video_mask": video_mask,
        "image_mask": image_mask,
        "video_source": video_source,
        "frames_positions": frames_positions,
        "keep_frames_video_source": keep_frames_video_source,
        "keep_frames_video_guide": keep_frames_video_guide,
        "denoising_strength": denoising_strength,
        "image_prompt_type": image_prompt_type,
        "video_prompt_type": video_prompt_type,        
        "audio_prompt_type": audio_prompt_type,
        "skip_steps_cache_type": skip_steps_cache_type,
        "model_switch_phase": model_switch_phase,
    } 

    if inputs["multi_prompts_gen_type"] == 0:
        if image_start != None and len(image_start) > 0:
            if inputs["multi_images_gen_type"] == 0:
                new_prompts = []
                new_image_start = []
                new_image_end = []
                for i in range(len(prompts) * len(image_start) ):
                    new_prompts.append(  prompts[ i % len(prompts)] )
                    new_image_start.append(image_start[i // len(prompts)] )
                    if image_end != None:
                        new_image_end.append(image_end[i // len(prompts)] )
                prompts = new_prompts
                image_start = new_image_start 
                if image_end != None:
                    image_end = new_image_end 
            else:
                if len(prompts) >= len(image_start):
                    if len(prompts) % len(image_start) != 0:
                        gr.Info("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                        return ret()
                    rep = len(prompts) // len(image_start)
                    new_image_start = []
                    new_image_end = []
                    for i, _ in enumerate(prompts):
                        new_image_start.append(image_start[i//rep] )
                        if image_end != None:
                            new_image_end.append(image_end[i//rep] )
                    image_start = new_image_start 
                    if image_end != None:
                        image_end = new_image_end 
                else: 
                    if len(image_start) % len(prompts)  !=0:
                        gr.Info("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                        return ret()
                    rep = len(image_start) // len(prompts)  
                    new_prompts = []
                    for i, _ in enumerate(image_start):
                        new_prompts.append(  prompts[ i//rep] )
                    prompts = new_prompts
            if image_end == None or len(image_end) == 0:
                image_end = [None] * len(prompts)

            for single_prompt, start, end in zip(prompts, image_start, image_end) :
                override_inputs.update({
                    "prompt" : single_prompt,
                    "image_start": start,
                    "image_end" : end,
                })
                inputs.update(override_inputs) 
                add_video_task(**inputs)
        else:
            for single_prompt in prompts :
                override_inputs["prompt"] = single_prompt 
                inputs.update(override_inputs) 
                add_video_task(**inputs)
        new_prompts_count = len(prompts)
    else:
        new_prompts_count = 1
        override_inputs["prompt"] = "\n".join(prompts)
        inputs.update(override_inputs) 
        add_video_task(**inputs)
    new_prompts_count += gen.get("prompts_max",0)
    gen["prompts_max"] = new_prompts_count
    state["validate_success"] = 1
    queue= gen.get("queue", [])
    first_time_in_queue = state.get("first_time_in_queue", True)
    state["first_time_in_queue"] = True
    return update_queue_data(queue, first_time_in_queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()
