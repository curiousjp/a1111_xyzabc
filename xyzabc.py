from collections import namedtuple
from copy import copy
from itertools import permutations, chain, product
from functools import reduce
import random
import argparse
import shlex
import csv
import os.path
from io import StringIO
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_samplers_kdiffusion, errors
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import re

from modules.ui_components import ToolButton

fill_values_symbol = "\U0001f4d2"  # 📒

AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])

parser = argparse.ArgumentParser(description = 'Parse prompt replacement strings')
parser.add_argument('--prompt', required=True)
parser.add_argument('--negative_prompt', default=None)

def prepare_sr_tuple_prompt(xs):
    valslist = csv_string_to_list_strip(xs)
    return [(valslist[0], x) for x in valslist[1:]]

def prepare_prompt_replacement(xs):
    # the namespace objects travel a bit awkwardly
    results = []
    for item in xs.split('\n'):
        pa, unknown = parser.parse_known_args(shlex.split(item)) 
        results.append((pa.prompt, pa.negative_prompt))
    return results

def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)
    return fun

def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")
    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)

def apply_tuple_prompt(p, x, xs):
    k, v = x
    if k not in p.prompt and k not in p.negative_prompt:
        raise RuntimeError(f'Prompt S/R (tuple) did not find {k} in prompt or negative prompt.')
    p.prompt = p.prompt.replace(k, v)
    p.negative_prompt = p.negative_prompt.replace(k, v)

def apply_prompt_replacement(p, x, xs):
    p.prompt = x[0]
    if x[1]:
        p.negative_prompt = x[1]

def apply_order(p, x, xs):
    token_order = []
    # Initally grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))
    token_order.sort(key=lambda t: t[0])
    prompt_parts = []
    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]
    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt

def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")

def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    p.override_settings['sd_model_checkpoint'] = info.name

def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")

def confirm_checkpoints_or_none(p, xs):
    for x in xs:
        if x in (None, "", "None", "none"):
            continue
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")

def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x

def apply_upscale_latent_space(p, x, xs):
    if x.lower().strip() != '0':
        opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        opts.data["use_scale_latent_for_hires_fix"] = False

def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return modules.sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(modules.sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            print(f"No VAE found for {name}; using automatic")
            return modules.sd_vae.unspecified
        else:
            return modules.sd_vae.vae_dict[choices[0]]

def apply_vae(p, x, xs):
    modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))

def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))

def apply_uni_pc_order(p, x, xs):
    opts.data["uni_pc_order"] = min(x, p.steps - 1)

def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')
    p.restore_faces = is_active

def apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        p.override_settings[field] = x
    return fun

def boolean_choice(reverse: bool = False):
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]
    return choice

def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return f"{opt.label}: {x}"

def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x

def format_value_join_list(p, opt, x):
    return ", ".join(x)

def do_nothing(p, x, xs):
    pass

def format_nothing(p, opt, x):
    return ""

def format_remove_path(p, opt, x):
    return os.path.basename(x)

def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x

def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()

def csv_string_to_list_strip(data_str):
    return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str)))))

class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True


class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption("Prompt S/R (skip first)", tuple, apply_tuple_prompt, prepare=prepare_sr_tuple_prompt, format_value=format_value),
    AxisOption("Prompt Replacement", tuple, apply_prompt_replacement, prepare=prepare_prompt_replacement, format_value=format_value),
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),
    AxisOptionTxt2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers]),
    AxisOptionTxt2Img("Hires sampler", str, apply_field("hr_sampler_name"), confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOptionImg2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_remove_path, confirm=confirm_checkpoints, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption("Schedule type", str, apply_override("k_sched_type"), choices=lambda: list(sd_samplers_kdiffusion.k_diffusion_scheduler)),
    AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
    AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
    AxisOption("Schedule rho", float, apply_override("rho")),
    AxisOption("Eta", float, apply_field("eta")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOption("Initial noise multiplier", float, apply_field("initial_noise_multiplier")),
    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['None'] + list(sd_vae.vae_dict)),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
    AxisOption("Always discard next-to-last sigma", str, apply_override('always_discard_next_to_last_sigma', boolean=True), choices=boolean_choice(reverse=True)),
    AxisOption("SGM noise multiplier", str, apply_override('sgm_noise_multiplier', boolean=True), choices=boolean_choice(reverse=True)),
    AxisOption("Refiner checkpoint", str, apply_field('refiner_checkpoint'), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("Refiner switch at", float, apply_field('refiner_switch_at')),
    AxisOption("RNG source", str, apply_override("randn_source"), choices=lambda: ["GPU", "CPU", "NV"]),
]

class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.vae = opts.sd_vae
        self.uni_pc_order = opts.uni_pc_order

    def __exit__(self, exc_type, exc_value, tb):
        opts.data["sd_vae"] = self.vae
        opts.data["uni_pc_order"] = self.uni_pc_order
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()
        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers

re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")

class Script(scripts.Script):
    def title(self):
        return "X/Y/Z/A plot"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    x_type = gr.Dropdown(label="X type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[1].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))
                    fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)

                with gr.Row():
                    y_type = gr.Dropdown(label="Y type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))
                    fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)

                with gr.Row():
                    z_type = gr.Dropdown(label="Z type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                    z_values = gr.Textbox(label="Z values", lines=1, elem_id=self.elem_id("z_values"))
                    fill_z_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)

                with gr.Row():
                    a_type = gr.Dropdown(label="A type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("a_type"))
                    a_values = gr.Textbox(label="A values", lines=1, elem_id=self.elem_id("a_values"))
                    fill_a_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_a_tool_button", visible=False)

                with gr.Row():
                    b_type = gr.Dropdown(label="B type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("b_type"))
                    b_values = gr.Textbox(label="B values", lines=1, elem_id=self.elem_id("b_values"))
                    fill_b_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_b_tool_button", visible=False)

                with gr.Row():
                    c_type = gr.Dropdown(label="C type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("c_type"))
                    c_values = gr.Textbox(label="C values", lines=1, elem_id=self.elem_id("c_values"))
                    fill_c_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_c_tool_button", visible=False)

                with gr.Row():
                    d_type = gr.Dropdown(label="D type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("d_type"))
                    d_values = gr.Textbox(label="D values", lines=1, elem_id=self.elem_id("d_values"))
                    fill_d_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_d_tool_button", visible=False)

                with gr.Row():
                    e_type = gr.Dropdown(label="E type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("e_type"))
                    e_values = gr.Textbox(label="E values", lines=1, elem_id=self.elem_id("e_values"))
                    fill_e_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_e_tool_button", visible=False)

                with gr.Row():
                    f_type = gr.Dropdown(label="F type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("f_type"))
                    f_values = gr.Textbox(label="F values", lines=1, elem_id=self.elem_id("f_values"))
                    fill_f_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_f_tool_button", visible=False)

        with gr.Row(variant="compact", elem_id="axis_options"):
            no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))

        def fill(axis_type):
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                return list_to_csv_string(axis.choices())
            return gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type], outputs=[x_values])
        fill_y_button.click(fn=fill, inputs=[y_type], outputs=[y_values])
        fill_z_button.click(fn=fill, inputs=[z_type], outputs=[z_values])
        fill_a_button.click(fn=fill, inputs=[a_type], outputs=[a_values])
        fill_b_button.click(fn=fill, inputs=[b_type], outputs=[b_values])
        fill_c_button.click(fn=fill, inputs=[c_type], outputs=[c_values])
        fill_d_button.click(fn=fill, inputs=[d_type], outputs=[d_values])
        fill_e_button.click(fn=fill, inputs=[e_type], outputs=[e_values])
        fill_f_button.click(fn=fill, inputs=[f_type], outputs=[f_values])

        def select_axis(axis_type, axis_values):
            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None
            return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=True, value=axis_values))

        x_type.change(fn=select_axis, inputs=[x_type, x_values], outputs=[fill_x_button, x_values])
        y_type.change(fn=select_axis, inputs=[y_type, y_values], outputs=[fill_y_button, y_values])
        z_type.change(fn=select_axis, inputs=[z_type, z_values], outputs=[fill_z_button, z_values])
        a_type.change(fn=select_axis, inputs=[a_type, a_values], outputs=[fill_a_button, a_values])
        b_type.change(fn=select_axis, inputs=[b_type, b_values], outputs=[fill_b_button, b_values])
        c_type.change(fn=select_axis, inputs=[c_type, c_values], outputs=[fill_c_button, c_values])
        d_type.change(fn=select_axis, inputs=[d_type, d_values], outputs=[fill_d_button, d_values])
        e_type.change(fn=select_axis, inputs=[e_type, e_values], outputs=[fill_e_button, e_values])
        f_type.change(fn=select_axis, inputs=[f_type, f_values], outputs=[fill_f_button, f_values])

        self.infotext_fields = (
            (x_type, "X Type"), (x_values, "X Values"),
            (y_type, "Y Type"), (y_values, "Y Values"),
            (z_type, "Z Type"), (z_values, "Z Values"),
            (a_type, "A Type"), (a_values, "A Values"),
            (b_type, "B Type"), (b_values, "B Values"),
            (c_type, "C Type"), (c_values, "C Values"),
            (d_type, "D Type"), (d_values, "D Values"),
            (e_type, "E Type"), (e_values, "E Values"),
            (f_type, "F Type"), (f_values, "F Values")
        )

        # it's a crime they don't let me pack these into some kind of list
        return [
            x_type, x_values, 
            y_type, y_values, 
            z_type, z_values, 
            a_type, a_values, 
            b_type, b_values, 
            c_type, c_values, 
            d_type, d_values,
            e_type, e_values,
            f_type, f_values,
            no_fixed_seeds]

    def run(self, p, x_t, x_v, y_t, y_v, z_t, z_v, a_t, a_v, b_t, b_v, c_t, c_v, d_t, d_v, e_t, e_v, f_t, f_v, no_fixed_seeds):
        axis_setup = [(x_t, x_v), (y_t, y_v), (z_t, z_v), (a_t, a_v), (b_t, b_v), (c_t, c_v), (d_t, d_v), (e_t, e_v), (f_t, f_v)]
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)
        p.batch_size = 1

        def process_axis(opt, vals):
            if opt.label == 'Nothing':
                return [0]
            if opt.prepare is not None:
                valslist = opt.prepare(vals)
            else:
                valslist = csv_string_to_list_strip(vals)
            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end = int(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end = float(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))
            valslist = [opt.type(x) for x in valslist]
            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)
            return valslist

        # axis_setup is a list of tuples of axis type numbers and user inputs
        # we will retrieve the true axis types using the numbers

        axis_setup = [(self.current_axis_options[atn], atn, av) for atn, av in axis_setup]

        # remove any non-activated items
        axis_setup = [(at, atn, av) for at, atn, av in axis_setup if at.label != 'Nothing']

        # sort the list by expected cost, largest to smallest

        axis_setup.sort(key = lambda x: x[0].cost, reverse = True)
        axis_labels = [x[0].label for x in axis_setup]

        # process the arguments away from their user input forms into 'real' forms

        axis_setup = [(at, atn, process_axis(at, av)) for at, atn, av in axis_setup]
         
        # process any 'Seed' or 'Var. seed' items if allowed
        def fix_axis_seeds(axis_opt, axis_list): 
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list
        if not no_fixed_seeds:
            axis_setup = [(at, atn, fix_axis_seeds(at, av)) for at, atn, av in axis_setup]

        def expectedStepsForAxis(at, atn, av):
            if at.label == 'Steps':
                return sum(av)
            if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr and at.label == 'Hires steps':
                return sum(av)
            return len(av)

        expected_images = reduce(lambda x,y: x*y, [len(av) for at, atn, av in axis_setup]) 

        expected_steps = reduce(lambda x,y: x*y, [expectedStepsForAxis(at, atn, av) for at, atn, av in axis_setup])
        if 'Steps' not in axis_labels:
            expected_steps *= p.steps
        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr and 'Hires steps' not in axis_labels:
            if p.hr_second_pass_steps:
               expected_steps *= p.hr_second_pass_steps
            else:
               expected_steps *= 2

        expected_steps *= p.n_iter

        print(f"X/Y/Z/A plot will create {expected_images} images. (Total steps to process: {expected_steps})")
        shared.total_tqdm.updateTotal(expected_steps)

        state.plot_state = [AxisInfo(at, av) for at, atn, av in axis_setup]

        def draw_single(axis_setup, axis_values):
            if shared.state.interrupted:
                return Processed(p, [], p.seed, "")
            pc = copy(p)
            pc.styles = pc.styles[:]
            print()
            print('setting x/y/z/a axis values:')
            for i in range(len(axis_values)):
                axis_type = axis_setup[i][0]
                axis_value = axis_values[i]
                axis_all_values = axis_setup[i][2]
                print(f' ** {axis_type.label}: {axis_value}')
                axis_type.apply(pc, axis_value, axis_all_values)
            try:
                res = process_images(pc)
            except Exception as e:
                errors.display(e, "generating image for xyza plot")
                res = Processed(p, [], p.seed, "")
            return res

        with SharedSettingsStackHelper():
            results_container = None
            for value_combination in product(*[av for at, atn, av in axis_setup]):
                processed: Processed = draw_single(axis_setup, value_combination)
                if results_container is None:
                    results_container = copy(processed)
                    results_container.images = []
                    results_container.all_prompts = []
                    results_container.all_seeds = []
                    results_container.infotexts = []
                    results_container.index_of_first_image = 1
                if processed.images:
                    results_container.images.append(processed.images[0])
                    results_container.all_prompts.append(processed.prompt)
                    results_container.all_seeds.append(processed.seed)
                    results_container.infotexts.append(processed.infotexts[0])
            if not results_container:
                # Should never happen, I've only seen it on one of four open tabs and it needed to refresh.
                print("Unexpected error: Processing could not begin, you may need to refresh the tab or restart the service.")
                results_container = Processed(p, [])
            elif not any(results_container.images):
                print("Unexpected error: draw_xyza_grid failed to return even a single processed image")
                results_container = Processed(p, [])

        return results_container
