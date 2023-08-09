import os

import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import numpy as np
from torchvision import transforms, datasets

from DDPMnet import Unet  # DDPM模型


# 定义4种生成β的方法，均需传入总步长T，返回β序列
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# 从序列a中取t时刻的值a[t](batch_size个)，维度与x_shape相同，第一维为batch_size
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# 扩散过程采样，即通过x0和t计算xt
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumpord_t * noise


# 损失函数loss，共3种计算方式，原文使用l2
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# 逆扩散过程采样，即通过xt和t计算xt-1，此过程需要通过网络
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumpord_t)
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# 逆扩散过程T次采样，即通过xT和T计算xi，获得每一个时刻的图像列表[xi]，此过程需要通过网络
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []
    for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs


# 逆扩散过程T次采样，允许传入batch_size指定生成图片的个数，用于生成结果的可视化
@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


if __name__ == "__main__":
    timesteps = 300  # 总步长T
    # 以下参数均为序列(List)，需要传入t获得对应t时刻的值 xt = X[t]
    betas = linear_beta_schedule(timesteps=timesteps)  # 选择一种方式，生成β(t)
    alphas = 1. - betas  # α(t)
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # α的连乘序列，对应α_bar(t)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0),
                                value=1.0)  # 将α_bar的最后一个值删除，在最开始添加1，对应前一个时刻的α_bar，即α_bar(t-1)
    sqrt_recip_alphas = torch.sqrt(1. / alphas)  # 1/根号下α(t)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 根号下α_bar(t)
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1. - alphas_cumprod)  # 根号下(1-α_bar(t))
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod)  # β(t)x(1-α_bar(t-1))/(1-α_bar(t))，即β^~(t)

    total_epochs = 10
    image_size = 28
    channels = 1
    batch_size = 64    #256
    lr = 1e-3

    os.makedirs("../dataset/mnist", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # 此处将输入数据从(0,1)区间转换到(-1,1)区间
    ])
    dataset = datasets.MNIST(root="../dataset/mnist", train=True, transform=transform, download=True)

    reverse_transform = transforms.Compose([  # tensor转换为PIL图片
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(total_epochs):
        total_loss = 0
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{total_epochs}", postfix=dict,
                    miniters=0.3)
        for iter, (img, _) in enumerate(dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            batch_size = img.shape[0]

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, img, t, loss_type="huber")  # 选择loss计算方式，计算loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix(**{"Loss": loss.item()})
            pbar.update(1)
        pbar.close()
        print("total_loss:%.4f" %
              (total_loss / len(dataloader)))

        # 展示一张图片的生成过程(去噪过程)，每3步生成一张图片，共100张图片(在一幅图中展示)
        val_images = sample(model, image_size, batch_size=1, channels=channels)
        fig, axs = plt.subplots(10, 10, figsize=(20, 20))
        plt.rc("text", color="blue")
        for t in range(100):
            i = t // 10
            j = t % 10
            image = val_images[t * 3 + 2].squeeze(0)
            image = reverse_transform(image)
            axs[i, j].imshow(image, cmap="gray")
            axs[i, j].set_axis_off()
            axs[i, j].set_title("$q(\mathbf{x}_{" + str(300 - 3 * t - 3) + "})$")
        plt.savefig(f"images/{epoch + 1}.png", bbox_inches='tight')
        plt.close()

