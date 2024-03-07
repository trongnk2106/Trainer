import subprocess
import os

import toml
from tqdm import tqdm
import argparse

from config import *
import shutil
# from train_network import train_network

def remove_files(dir_path):
    if os.path.isfile(dir_path):
        os.remove(dir_path)
    elif os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    else:
        print(f"Item '{dir_path}' is neither a file nor a directory.")


def run(data_dir, user, task, sdxl, is_male):
    
    project_name = user + task

    train_data_dir = data_dir
    
    reg_data_dir = train_data_dir
    os.chdir(root_dir)
    test = os.listdir(train_data_dir)
    
    
    os.chdir(finetune_dir)
    
    for item in test:
        file_ext = os.path.splitext(item)[1]
        if file_ext not in supported_types:
            remove_files(os.path.join(train_data_dir, item))
            
            
    
    subprocess.run([
    "python", "tag_images_by_wd14_tagger.py",
    train_data_dir,
    "--batch_size", "4",
    "--repo_id", model,
    "--thresh", str(threshold),
    "--caption_extension", ".txt",
    "--max_data_loader_n_workers", str(max_data_loader_n_workers)
    ])
    
    if extension != "" :
        def add_tag(filename, tag, append):
            with open(filename, "r") as f:
                contents = f.read()
        
            tag = ", ".join(tag.split())
            tag = tag.replace("_", " ")
        
            if tag in contents:
                return
        
            if append:
                contents = contents.rstrip() + ", " + tag
            else:
                contents = tag + ", " + contents
        
            with open(filename, "w") as f:
                f.write(contents)
        
            if not any([filename.endswith("." + extension) for filename in os.listdir(train_data_dir)]):
                for filename in os.listdir(train_data_dir):
                    if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                        open(os.path.join(train_data_dir, filename.split(".")[0] + "." + extension), "w").close()
        
        
    print("Project Name: ", project_name)
    print("Model Version: Stable Diffusion V1.x") if not v2 else ""
    print("Model Version: Stable Diffusion V2.x") if v2 and not v_parameterization else ""
    print("Model Version: Stable Diffusion V2.x 768v") if v2 and v_parameterization else ""
    # print("Pretrained Model Path: ", pretrained_model_name_or_path) if pretrained_model_name_or_path else print("No Pretrained Model path specified.")
    # print("VAE Path: ", vae) if vae else print("No VAE path specified.")
    print("Output Path: ", output_dir)
    
    
    config = {
        "general": {
            "enable_bucket": True,
            "caption_extension": caption_extension,
            "shuffle_caption": True,
            "keep_tokens": keep_tokens,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "resolution": resolution,
                "min_bucket_reso": 320 if resolution > 640 else 256,
                "max_bucket_reso": 1280 if resolution > 640 else 1024,
                "caption_dropout_rate": caption_dropout_rate if caption_extension == ".caption" else 0,
                "caption_tag_dropout_rate": caption_dropout_rate if caption_extension == ".txt" else 0,
                "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
                "flip_aug": flip_aug,
                "color_aug": False,
                "face_crop_aug_range": None,
                "subsets": [
                    {
                        "image_dir": train_data_dir,
                        "class_tokens": f"{instance_token} {class_token}",
                        "num_repeats": train_repeats,
                    },
                    {
                        "is_reg": True,
                        "image_dir": reg_data_dir,
                        "class_tokens": class_token,
                        "num_repeats": reg_repeats,
                    }
                ]
            }
        ]
    }
    
    
    config_str = toml.dumps(config)
    
    dataset_config = os.path.join(config_dir, "dataset_config.toml")
    #check file exist   
    
    
    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None
    
    config_str = toml.dumps(config)
    
    with open(dataset_config, "w") as f:
        f.write(config_str)

    sample_every_n_type_value = 2
    if not enable_sample:
        sample_every_n_type_value = 999999
        
    if is_male == "1": 
        sample_str = f"""1man, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps} {'--d ' + str(seed) if seed > 0 else ''}"""
    elif is_male == "0": 
        sample_str = f"""1girl, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps} {'--d ' + str(seed) if seed > 0 else ''}"""

    
    prompt_path = os.path.join(config_dir, "sample_prompt.txt")
    
    with open(prompt_path, "w") as f:
        f.write(sample_str)


    config = {
        "model_arguments": {
            "v2": v2,
            "v_parameterization": v_parameterization if v2 and v_parameterization else False,
            "pretrained_model_name_or_path": pretrained_model_name_or_path if sdxl == "0" else None,
            "vae": vae,
        },
        "additional_network_arguments": {
            "no_metadata": False,
            "unet_lr": float(unet_lr) if train_unet else None,
            "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
            "network_weights": network_weight,
            "network_module": network_module,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": network_args,
            "network_train_unet_only": True if train_unet and not train_text_encoder else False,
            "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
            "training_comment": None,
        },
        "optimizer_arguments": {
            "optimizer_type": optimizer_type,
            "learning_rate": unet_lr,
            "max_grad_norm": 1.0,
            "optimizer_args": eval(optimizer_args) if optimizer_args else None,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        },
        "dataset_arguments": {
            "cache_latents": True,
            "debug_dataset": False,
        },
        "training_arguments": {
            "output_dir": output_dir,
            "output_name": project_name,
            "save_precision": save_precision,
            "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
            "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
            "save_last_n_epochs": None,
            "save_state": None,
            "save_last_n_epochs_state": None,
            "resume": None,
            "train_batch_size": train_batch_size,
            "max_token_length": 225,
            "mem_eff_attn": False,
            "xformers": True,
            "max_train_epochs": num_epochs,
            "max_data_loader_n_workers": 8,
            "persistent_data_loader_workers": True,
            "seed": seed if seed > 0 else None,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "clip_skip": clip_skip if not v2 else None,
            "logging_dir": logging_dir,
            "log_prefix": project_name,
            "noise_offset": noise_offset if noise_offset > 0 else None,
            "lowram": lowram,
            "no_half_vae": True if sdxl == "1" else False
        },
        "sample_prompt_arguments":{
            "sample_every_n_steps": sample_every_n_type_value if sample_every_n_type == "sample_every_n_steps" else None,
            "sample_every_n_epochs": sample_every_n_type_value if sample_every_n_type == "sample_every_n_epochs" else None,
            "sample_sampler": sampler,
        },
        "dreambooth_arguments":{
            "prior_loss_weight": 1.0,
        },
        "saving_arguments":{
            "save_model_as": save_model_as
        },
    }
    
    config_path = os.path.join(config_dir, "config_file.toml")
    
    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None
    
    config_str = toml.dumps(config)
    
    with open(config_path, "w") as f:
        f.write(config_str)
    
    print(config_str)

    
    # Navigate to the specified directory
    os.chdir(repo_dir)
    
    if sdxl == "1":
        subprocess.run([
        "accelerate", "launch",
        "--config_file", accelerate_config,
        "--num_cpu_threads_per_process", "1",
        "sdxl_train_network.py",
        "--sample_prompts", os.path.join(root_dir, "LoRA/config/sample_prompt.txt"),
        "--dataset_config", os.path.join(root_dir, "LoRA/config/dataset_config.toml"),
        "--config_file", os.path.join(root_dir, "LoRA/config/config_file.toml")
    ])
    else:
        subprocess.run([
        "accelerate", "launch",
        "--config_file", accelerate_config,
        "--num_cpu_threads_per_process", "1",
        "train_network.py",
        "--sample_prompts", os.path.join(root_dir, "LoRA/config/sample_prompt.txt"),
        "--dataset_config", os.path.join(root_dir, "LoRA/config/dataset_config.toml"),
        "--config_file", os.path.join(root_dir, "LoRA/config/config_file.toml")
    ])


#     subprocess.run([
#     "accelerate", "launch",
#     "--num_cpu_threads_per_process", "1",
#     "train_network.py",
#     "--config_file" , 
# ])

def trainner_lora(feedback_status : str, data_dir : str, user : str, task : str, sdxl : str, is_male : str):
    '''
    
        Args:
            feedback_status : str : The feedback status of the user
            data_dir : str : The directory of the data
            user : str : The user's name
            task : str : The task of the user
            sdxl : str : 1 train lora sdxl, 0 train lora sd 1.5
            is_male : str : 1 user is male else female
    
    '''
    if feedback_status == "bad":
        run(data_dir , user , task , sdxl , is_male )
    else :
        return "Good"

# if __name__ ==  '__main__':
    
#     parser = argparse.ArgumentParser()
    # implement the react_feedback if react_feedback is True => run train else not

    # parser.add_argument("--data_dir", required=True, default='', type=str)
    # parser.add_argument("--user", required=True, default='', type=str)
    # parser.add_argument("--task", required=True, default='', type=str)
    # parser.add_argument("--sdxl", required=True, default='0', type=str)
    # parser.add_argument("--is_male", required=True, default=True, type=bool)
    # # parser.add_argument("--store_weight", action='store_true')
    
    # args = parser.parse_args()
    
    # feedback_react = False
    
    # if feedback_react:
    #     main(args)
    
    