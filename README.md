# a1111_xyzabc
`xyzabc.py` is a fork of the [xyz_grid.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/xyz_grid.py) script that ships with the A1111 stable diffusion webui, and can be dropped alongside it in your installation. It is tailored to my particular interests when generating variations, and adds some features to the original `xyz_grid` while removing others.

The generous licensing of the webui under the GNU Affero General Public License makes this derivative work possible, and it is released under the same license.

## features removed
This script removes the grid drawing and labelling functionality of the original script, which I seldom used. As such, it also removes the ceiling that the maximum grid image dimensions imposed on the number of variants you could generate in one run. The removal of the grid allowed me to make some changes to the internal working of the script that make it a bit easier to expand on.

## features added
The original three axes (x, y, and z) have been increased to nine (x, y, z, a, b, c, d, e, and f). Some additional axes types have been added.

## new axes types
### Prompt S/R (skip first)
The original Prompt S/R includes the base case - applying an argument of `ANIMAL, dog, cat` to a prompt of `a coffee shop logo featuring an ANIMAL` would generate three images:
* `a coffee shop logo featuring an ANIMAL`
* `a coffee shop logo featuring an dog`
* `a coffee shop logo featuring an cat`

This variant omits the first example, making it safe to use meaninglessly named variable names in your search and replace operations.

### Prompt Replacement
Accepts an attenuated form of the 'prompts from file' style format, replacing the positive and negative prompts in their entirety (styles are still preserved unless otherwise displaced). For example,
```
--prompt='a coffee shop logo featuring a cat' --negative_prompt='cartoon'
--prompt='a design for a coffee cup featuring a hedgehog' --negative_prompt='cartoon'
--prompt='a colourful t-shirt'
```

`--prompt` is required here, but `--negative_prompt` is optional. If it is not set, the default negative prompt will apply. For compatibility reasons with bulk prompt generators, other tags will be read but ignored.
