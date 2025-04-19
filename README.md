# CLIP It and Rank It

A tiny script that uses [OpenAI’s CLIP](https://openai.com/research/clip) to score how well an image matches a set of text descriptions. It’s like your AI friend that says, *“Yeah, that caption works.”*

### What It Does

You give it:
- An image (`clip.jpg`)
- A list of descriptions

And it returns a probability for each caption — higher = better match!

### How to Run

1. **Install dependencies**:
```bash
pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install Pillow
```

2. **Add your image**  
Make sure `clip.jpg` (or whatever image you want) is in the same folder.

3. **Run the script**:
```bash
python your_script.py
```

It'll print something like:
```python
{
  "A blue couch with black table": 0.48,
  "A couch and a table": 0.27,
  ...
}
```

### Under the Hood

- Uses **ViT-B/32** CLIP model
- Extracts embeddings for the image and text
- Computes **cosine similarity**, then softmaxes the result
