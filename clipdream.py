import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import open_clip
from tqdm import tqdm
import argparse
import tempfile
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIPDream - CLIP-guided image optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", type=str, help="text prompt to optimize towards")
    parser.add_argument("-i", "--input", type=str, default="assets/input.jpg", help="input image path")
    parser.add_argument("-o", "--output", type=str, default="assets/output.jpg", help="output image path")
    parser.add_argument("-s", "--steps", type=int, default=300, help="total optimization steps")
    parser.add_argument("--lr", type=float, default=0.08, help="learning rate")
    parser.add_argument("--blur", type=float, default=3.0, help="gradient blur sigma")
    parser.add_argument("--smooth", type=float, default=2.0, help="TV smoothness weight")
    parser.add_argument("--cutouts", type=int, default=16, help="number of cutouts per step")
    parser.add_argument("--no-gif", action="store_true", help="skip gif generation")
    parser.add_argument("--gif-interval", type=int, default=4, help="capture frame every N steps")
    parser.add_argument("--patience", type=int, default=30, help="early stop if no improvement for N steps")
    parser.add_argument("--min-delta", type=float, default=0.001, help="minimum improvement to reset patience")
    return parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# color correlation matrix from imagenet stats
# prevents rgb channels from being optimized independently which causes rainbow noise
color_correlation_svd = torch.tensor([
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
], dtype=torch.float32).to(DEVICE)
color_correlation_svd_inv = torch.linalg.inv(color_correlation_svd.cpu()).to(DEVICE)

# wraps image tensor for optimization
# converts to decorrelated color space so optimizer cant make rainbow artifacts
class DirectOptim(nn.Module):
    def __init__(self, img_tensor):
        super().__init__()
        # convert to logit space so we can optimize unbounded values
        logits = torch.logit(img_tensor.clamp(0.01, 0.99))
        b, c, h, w = logits.shape
        logits_flat = logits.view(b, c, -1).permute(0, 2, 1)
        # project into decorrelated color space
        params_flat = torch.matmul(logits_flat, color_correlation_svd_inv.T)
        self.params = nn.Parameter(params_flat.permute(0, 2, 1).view(b, c, h, w))
        
    def forward(self):
        b, c, h, w = self.params.shape
        params_flat = self.params.view(b, c, -1).permute(0, 2, 1)
        # project back to rgb
        rgb_logits_flat = torch.matmul(params_flat, color_correlation_svd.T)
        rgb_logits = rgb_logits_flat.permute(0, 2, 1).view(b, c, h, w)
        # sigmoid keeps output in 0-1 range
        return torch.sigmoid(rgb_logits)

# random crops of the image at different scales
# clip sees these instead of full image for better gradients
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        # augmentations help clip see varied perspectives
        self.augs = nn.Sequential(
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.1, p=0.7),
        )

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            # bias towards larger crops
            size = int(torch.rand([])**0.5 * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.interpolate(cutout, size=(self.cut_size, self.cut_size), mode='bilinear', align_corners=False)
            cutouts.append(self.augs(cutout))
        return torch.cat(cutouts)

# main optimization loop for one stage coarse or fine
def optimize_stage(init_tensor, steps, size, blur_sigma, tv_weight, lr, cutouts, prompt, 
                   model, tokenizer, desc="Stage", frame_dir=None, gif_interval=4, 
                   frame_counter=0, patience=50, min_delta=0.001):
    # resize if needed
    if init_tensor.shape[-1] != size[1] or init_tensor.shape[-2] != size[0]:
        init_tensor = F.interpolate(init_tensor, size=size, mode='bilinear', align_corners=False)
    
    optim_net = DirectOptim(init_tensor).to(DEVICE)
    # momentum helps converge faster
    optimizer = torch.optim.Adam(optim_net.parameters(), lr=lr, betas=(0.5, 0.99))
    scaler = torch.amp.GradScaler('cuda')
    
    make_cutouts = MakeCutouts(224, cutouts).to(DEVICE)
    # clip normalization values
    normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    # encode text prompt once
    with torch.no_grad():
        text_tokens = tokenizer([prompt]).to(DEVICE)
        target_features = model.encode_text(text_tokens)
        target_features /= target_features.norm(dim=-1, keepdim=True)
    
    # early stopping state
    best_loss = float('inf')
    patience_counter = 0
    
    pbar = tqdm(range(steps), desc=desc)
    for i in pbar:
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            img_tensor = optim_net()
            cuts = make_cutouts(img_tensor)
            cuts = normalize(cuts)
            image_features = model.encode_image(cuts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # maximize similarity between image and text
            clip_loss = -torch.cosine_similarity(target_features, image_features).mean()
            
            # total variation loss for smoothness
            tv_loss = (torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]).mean() +
                       torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]).mean())
            total_loss = clip_loss + (tv_weight * tv_loss)
        
        loss_val = clip_loss.item()
        
        # early stopping check
        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            pbar.set_description(f"{desc} (early stop)")
            break
        
        scaler.scale(total_loss).backward()
        
        # blur gradients to prevent pixel noise
        if blur_sigma > 0:
            with torch.no_grad():
                g = optim_net.params.grad
                k_size = int(blur_sigma * 4) + 1
                if k_size % 2 == 0: k_size += 1
                g = TF.gaussian_blur(g, kernel_size=k_size, sigma=blur_sigma)
                # normalize gradient magnitude
                g = g / (g.std() + 1e-8) * 0.1
                optim_net.params.grad = g
        
        scaler.step(optimizer)
        scaler.update()

        # save frame for gif
        if frame_dir is not None and i % gif_interval == 0:
            with torch.no_grad():
                current = optim_net()
                _, _, fh, fw = current.shape
                max_dim = 512
                scale = min(max_dim / max(fh, fw), 1.0)
                new_h, new_w = int(fh * scale), int(fw * scale)
                frame_small = F.interpolate(current, size=(new_h, new_w), mode='bilinear', align_corners=False)
                frame_pil = T.ToPILImage()(frame_small.squeeze(0).cpu())
                frame_pil.save(os.path.join(frame_dir, f"{frame_counter:05d}.jpg"), quality=85)
                frame_counter += 1
        
    return optim_net().detach(), frame_counter

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[!] error: {args.input} not found")
        return
    
    print(f"[*] device: {DEVICE}")
    print(f"[*] loading CLIP...")
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model.to(DEVICE).eval()
    print(f"[+] loaded")
    
    print(f"[*] prompt: '{args.prompt}'")
    print(f"[*] input: {args.input}")
    print(f"[*] steps: {args.steps} (patience: {args.patience})")
    
    img_pil = Image.open(args.input).convert("RGB")
    base_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)
    _, _, h, w = base_tensor.shape
    
    # temp folder for gif frames
    frame_dir = None
    if not args.no_gif:
        frame_dir = tempfile.mkdtemp(prefix="clipdream_")
    
    frame_counter = 0
    
    # 70/30 split between coarse and fine
    coarse_steps = int(args.steps * 0.7)
    fine_steps = args.steps - coarse_steps

    # coarse pass at half resolution with heavy blur
    print(f"[1/2] coarse pass (70% = {coarse_steps} steps)")
    coarse_tensor, frame_counter = optimize_stage(
        base_tensor, 
        steps=coarse_steps, 
        size=(h//2, w//2), 
        blur_sigma=args.blur, 
        tv_weight=args.smooth,
        lr=args.lr,
        cutouts=args.cutouts,
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        desc="coarse",
        frame_dir=frame_dir,
        gif_interval=args.gif_interval,
        frame_counter=frame_counter,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # fine pass at full resolution with lighter blur
    print(f"[2/2] fine pass (30% = {fine_steps} steps)")
    fine_tensor, frame_counter = optimize_stage(
        coarse_tensor, 
        steps=fine_steps, 
        size=(h, w), 
        blur_sigma=args.blur * 0.5,
        tv_weight=args.smooth,
        lr=args.lr,
        cutouts=args.cutouts,
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        desc="fine",
        frame_dir=frame_dir,
        gif_interval=args.gif_interval,
        frame_counter=frame_counter,
        patience=args.patience,
        min_delta=args.min_delta
    )

    with torch.no_grad():
        final_pil = T.ToPILImage()(fine_tensor.squeeze(0).cpu())
        final_pil.save(args.output)
    print(f"\n[+] saved: {args.output}")

    # stitch frames into gif
    if frame_dir and frame_counter > 0:
        print(f"[*] creating gif from {frame_counter} frames...")
        
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        
        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(frame_dir, fname))
            # convert to palette mode for smaller gif
            frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        
        # calculate duration per frame to fit within 30 seconds
        max_duration_ms = 30000
        duration_per_frame = max(20, max_duration_ms // len(frames))
        
        gif_path = args.output.rsplit(".", 1)[0] + ".gif"
        frames[0].save(
            gif_path, 
            save_all=True, 
            append_images=frames[1:], 
            optimize=True, 
            duration=duration_per_frame, 
            loop=0
        )
        print(f"[+] gif: {gif_path}")
        
        # cleanup temp files
        shutil.rmtree(frame_dir)

if __name__ == "__main__":
    main()
