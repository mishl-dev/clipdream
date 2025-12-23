# CLIPDream

CLIP-guided image optimization. Feed it a prompt, watch it hallucinate shapes into your image.

Think Google's DeepDream but with text prompts instead of dog-eyes everywhere.

## Example

| Input | Output | Process |
|-------|--------|---------|
| <img src="assets/input.jpg" width="250"> | <img src="assets/output.jpg" width="250"> | <img src="assets/output.gif" width="250"> |

prompt: `"space"`

## What it does

- takes an input image
- you give it a text prompt like "skull" or "ocean waves"
- it optimizes the image pixels to match your prompt using CLIP
- outputs a trippy image and animated gif of the process

## How it works

1. coarse pass at 50% resolution (big shapes, heavy blur)
2. fine pass at 100% resolution (details, lighter blur)
3. color decorrelation to avoid rainbow noise
4. gradient blurring to keep it smooth
5. early stopping when improvement plateaus

## Usage

```bash
python clipdream.py "skull"
```

### Options

```
positional arguments:
  prompt                text prompt to optimize towards

options:
  -i, --input           input image path (default: assets/input.jpg)
  -o, --output          output image path (default: assets/output.jpg)
  -s, --steps           total optimization steps (default: 800)
  --lr                  learning rate (default: 0.08)
  --blur                gradient blur sigma (default: 3.0)
  --smooth              TV smoothness weight (default: 2.0)
  --cutouts             number of cutouts per step (default: 8)
  --no-gif              skip gif generation
  --gif-interval        capture frame every N steps (default: 4)
  --patience            early stop if no improvement for N steps (default: 20)
  --min-delta           minimum improvement to reset patience (default: 0.001)
```

### Examples

```bash
# basic (uses assets/input.jpg by default)
python clipdream.py "ocean waves"

# custom input/output
python clipdream.py "dragon" -i photo.jpg -o result.jpg

# more steps, no early stopping
python clipdream.py "face" --patience 9999

# skip gif for faster runs
python clipdream.py "skull" --no-gif
```

## Requirements

- pytorch
- torchvision
- open_clip
- pillow
- tqdm

```bash
pip install torch torchvision open-clip-torch pillow tqdm
```

## Outputs

- `assets/output.jpg` - final image
- `assets/output.gif` - animated process

## License

[MIT](LICENSE)
