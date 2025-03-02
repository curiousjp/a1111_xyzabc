# a1111_xyzabc
`xyzabc.py` is a fork of the [xyz_grid.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/xyz_grid.py) script that ships with the A1111 stable diffusion webui, and can be dropped alongside it in your installation. It is tailored to my particular interests when generating variations, and adds some features to the original `xyz_grid` while removing others.

The generous licensing of the webui under the GNU Affero General Public License makes this derivative work possible, and it is released under the same license.

## features removed
This script removes the grid drawing and labelling functionality of the original script, which I seldom used. As such, it also removes the ceiling that the maximum grid image dimensions imposed on the number of variants you could generate in one run. The removal of the grid allowed me to make some changes to the internal working of the script that make it a bit easier to expand on.

## features added
The original three axes (x, y, and z) have been increased to nine (x, y, z, a, b, c, d, e, and f). Some additional axes types have been added.

## new axes types
### Checkpoint name and matching style
Works like "Checkpoint name", but tries to add a style by the same name to the prompt. If the checkpoint file ends in ".safetensors", this is removed first, so a checkpoint "my_cool_model.safetensors" will try to add a "my_cool_model" style to the style list, and cause an error if it cannot find one. Useful if you have standard styles that must be expressed differently in e.g. SD1/SDXL models.

### Prompt S/R (skip first)
The original Prompt S/R includes the base case - applying an argument of `ANIMAL, dog, cat` to a prompt of `a coffee shop logo featuring an ANIMAL` would generate three images:
* `a coffee shop logo featuring an ANIMAL`
* `a coffee shop logo featuring an dog`
* `a coffee shop logo featuring an cat`

This variant omits the first example, making it safe to use meaninglessly named variable names in your search and replace operations.

### Prompt S/R (by dictionary)
Takes a single string representing a JSON object - either a dictionary or list of dictionaries. During axis evaluation, each key in the dictionary is replaced into the prompt string - failure to find a key will raise an exception. If the key leads to a list of strings instead of a single string, one will be chosen at random. Example: `[{"SETTING": "outer space", "SKY": "there is a starry sky overhead"}, {"SETTING": "campsite", "SKY": ["the moon is hanging in the night sky", "there is a bright blue sky with clouds overhead"]}]`.

### Global Prompt Reweight and Global Prompt Reweight (Positive Only)
Wraps the entire prompt (either the positive prompt only, or both) in a grouping operator, and then weights it according to the provided values, using the normal syntax for floating point arguments. For example, a prompt `beautiful, masterpiece, best quality, perfect lighting, night, landscape, (no humans),  advntr` and argument of `1.0-1.5 [3]` would create the following prompts:
* `(beautiful, masterpiece, best quality, perfect lighting, night, landscape, (no humans), advntr:1.0)`
* `(beautiful, masterpiece, best quality, perfect lighting, night, landscape, (no humans), advntr:1.25)`
* `(beautiful, masterpiece, best quality, perfect lighting, night, landscape, (no humans), advntr:1.5)`

Useful for probing the burn-out points of combinations of LORA or embeddings.

### Prompt Replacement
Accepts an attenuated form of the 'prompts from file' style format, replacing the positive and negative prompts in their entirety (styles are still preserved unless otherwise displaced). For example,
```
--prompt='a coffee shop logo featuring a cat' --negative_prompt='cartoon'
--prompt='a design for a coffee cup featuring a hedgehog' --negative_prompt='cartoon'
--prompt='a colourful t-shirt'
```

`--prompt` is required here, but `--negative_prompt` is optional. If it is not set, the default negative prompt will apply. For compatibility reasons with bulk prompt generators, other tags will be read but ignored.

### Use of '-2' for seeds
I have noticed some issues in forge where successive -1 prompts would resolve to the same seed unexpectedly. This may be user error, but the Seed axis type now recognises -2 as a random integer calculated _as the axis is being set_.