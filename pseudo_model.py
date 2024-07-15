import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tools.pseudo_face_data import faces_data
from models.rcan import make_cleaning_net, make_SR_net
from models.generators import TransferNet
from models.discriminators import NLayerDiscriminator
from models.losses import GANLoss
from models.geo_loss import geometry_ensemble



class Pseudo_Model():
    def __init__(self, device, cfg, use_ddp=False):
        # Initialize parameters and set device
        self.device = device  # Set the device (CPU or GPU)
        self.idt_input_clean = cfg.OPT_CYC.IDT_INPUT == "clean"  # Check if input is clean based on the config
        rgb_range = cfg.DATA.IMG_RANGE # Set the RGB image range from the configuration
        rgb_mean_point = (0.5, 0.5, 0.5) if cfg.DATA.IMG_MEAN_SHIFT else (0, 0, 0)  # Determine mean value adjustment
        # Initialize generator and discriminator networks
        self.G_xy = make_cleaning_net(rgb_range=rgb_range, rgb_mean=rgb_mean_point).to(device)   # Cleaning network
        self.G_yx = TransferNet(rgb_range=rgb_range, rgb_mean=rgb_mean_point).to(device)  # Transfer network
        self.U = make_SR_net(rgb_range=rgb_range, rgb_mean=rgb_mean_point, scale_factor=2).to(device)  # Super-resolution network
        self.D_x = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity).to(device)  # Discriminator for low-res images
        self.D_y = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity).to(device)  # Discriminator for high-res images
        self.D_sr = NLayerDiscriminator(3, scale_factor=cfg.SR.SCALE, norm_layer=nn.Identity).to(device) # Discriminator for super-res images

        # Wrap networks with Distributed Data Parallel if specified
        if use_ddp:
            self.G_xy = DDP(self.G_xy, device_ids=[device])
            self.G_yx = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_yx), device_ids=[device])
            self.U = DDP(self.U, device_ids=[device])
            self.D_x = DDP(self.D_x, device_ids=[device])
            self.D_y = DDP(self.D_y, device_ids=[device])
            self.D_sr = DDP(self.D_sr, device_ids=[device])

        # Initialize optimizers for each network
        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_U = optim.Adam(self.U.parameters(), lr=cfg.OPT_SR.LR_G, betas=cfg.OPT_SR.BETAS_G)

        # Initialize learning rate schedulers for each network
        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)

       # Organize networks, optimizers, and schedulers into dictionaries for easy management
        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr}
        self.discs = ["D_x", "D_y", "D_sr"]
        self.gens = ["G_xy", "G_yx", "U"]

        self.n_iter = 0  # Initialize iteration count for training
        self.gan_loss = GANLoss("lsgan")  # Initialize the GAN loss function with "lsgan" mode
        self.l1_loss = nn.L1Loss()   # Initialize the L1 loss function

        # Loss weights from configuration
        self.d_sr_weight = cfg.OPT_CYC.LOSS.D_SR_WEIGHT
        self.cyc_weight = cfg.OPT_CYC.LOSS.CYC_WEIGHT
        self.idt_weight = cfg.OPT_CYC.LOSS.IDT_WEIGHT
        self.geo_weight = cfg.OPT_CYC.LOSS.GEO_WEIGHT

    def net_save(self, folder, shout=False):
        """Save network, optimizer, and learning rate scheduler states to a file."""
        file_name = os.path.join(folder, f"nets_{self.n_iter}.pth")
        nets = {k:v.state_dict() for k, v in self.nets.items()}
        optims = {k:v.state_dict() for k, v in self.optims.items()}
        lr_decays = {k:v.state_dict() for k, v in self.lr_decays.items()}
        alls = {"nets":nets, "optims":optims, "lr_decays":lr_decays}
        torch.save(alls, file_name)
        if shout: print("Saved: ", file_name)
        return file_name

    def net_load(self, file_name, strict=True):
        """Load network, optimizer, and scheduler states from a file."""
        map_loc = {"cuda:0": f"cuda:{self.device}"}
        loaded = torch.load(file_name, map_location=map_loc)
        for n in self.nets:
            self.nets[n].load_state_dict(loaded["nets"][n], strict=strict)
        for o in self.optims:
            self.optims[o].load_state_dict(loaded["optims"][o])
        for l in self.lr_decays:
            self.lr_decays[l].load_state_dict(loaded["lr_decays"][l])

    def net_grad_toggle(self, nets, need_grad):
        """Toggle gradient calculation for selected networks."""
        for n in nets:
            for p in self.nets[n].parameters():
                p.requires_grad = need_grad

    def mode_selector(self, mode="train"):
        """Set networks to train or evaluation mode based on the input mode."""
        if mode == "train":
            for n in self.nets:
                self.nets[n].train()
        elif mode in ["eval", "test"]:
            for n in self.nets:
                self.nets[n].eval()

    def lr_decay_step(self, shout=False):
        """Adjust learning rates based on predefined milestones."""
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += f", {n}={self.lr_decays[n].get_last_lr()[0]}" if i > 0 else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
        if shout and changed: print(lrs)

    def test_sample(self, Xs, Yds=None, Zs=None):
        """Generate test samples from input data using the networks."""
        x = None
        with torch.no_grad():
            y = self.nets["G_xy"](Xs)
            sr = self.nets["U"](y)
        if Yds is not None and Zs is not None:
            x = self.nets["G_yx"](Yds, Zs)
        return y, sr, x

    def train_step(self, Ys, Xs, Yds, Zs):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        self.n_iter += 1
        loss_dict = dict()

        # forward
        # Generate fake low-resolution images using downsampled HR and noise
        fake_Xs = self.G_yx(Yds, Zs)
        # Reconstruct downsampled HR images from the fake low-res images
        rec_Yds = self.G_xy(fake_Xs)
        # Generate fake downsampled HR images from low-res images
        fake_Yds = self.G_xy(Xs)
        # Apply geometric ensemble for downsampled HR images
        geo_Yds = geometry_ensemble(self.G_xy, Xs)
        # Generate identity outputs if needed
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        # Super-resolve the reconstructed and fake images
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        # Enable gradient calculation for discriminators
        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        # Train discriminator `D_x`
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        # Calculate GAN loss for `D_x`
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        # Optimize `D_x` using backpropagation
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()
        # Train discriminator `D_y`
        # D_y
        # Compare real downsampled HR images with fake ones
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        # Calculate GAN loss for `D_y`
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        # Optimize `D_y` using backpropagation
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()
        # Train discriminator `D_sr`
        # Compare super-resolved images with fake images
        # D_sr
        pred_sr_x = self.D_sr(sr_x.detach())
        pred_sr_y = self.D_sr(sr_y.detach())
        # Calculate GAN loss for `D_sr`
        loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        # Optimize `D_sr` using backpropagation
        self.opt_Dsr.zero_grad()
        loss_D_sr.backward()
        self.opt_Dsr.step()
        loss_dict["D_sr"] = loss_D_sr.item()

        # Disable gradients for discriminators
        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)

        # ===== Train Generators =====
        # G_yx
        # Zero the gradients of the generator optimizers
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        # Calculate GAN loss for generator `G_yx` using discriminator `D_x`
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # Calculate losses for generator `G_xy`
        # GAN loss from discriminator `D_y`
        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        # Identity loss if enabled, between output and the target (clean or noisy)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        # Cycle consistency loss (reconstruction)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        # Geometric ensemble loss
        loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        # GAN loss for super-resolution
        loss_d_sr = self.gan_loss(pred_sr_y, True, False)
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["D_sr"] = loss_d_sr.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # Backpropagate the total generator loss and update weights
        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        # ===== Train Upscaler =====
        # Zero the gradients of the upscaler optimizer
        # U
        self.opt_U.zero_grad()
        # Calculate L1 loss between the super-resolved images and the original high-res images
        loss_U = self.l1_loss(self.U(rec_Yds.detach()), Ys)
        # Backpropagate and optimize the upscaler
        loss_U.backward()
        self.opt_U.step()
        loss_dict["U_pix"] = loss_U.item()
        return loss_dict

if __name__ == "__main__":
    from yacs.config import CfgNode
    with open("/Users/noymachluf/Desktop/pseudo-sr2/configs/faces.yaml", "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()
    device = "cpu"
    x = torch.randn(8, 3, 32, 32, dtype=torch.float32, device=device)
    y = torch.randn(8, 3, 64, 64, dtype=torch.float32, device=device)
    yd = torch.randn(8, 3, 32, 32, dtype=torch.float32, device=device)
    z = torch.randn(8, 1, 8, 8, dtype=torch.float32, device=device)
    model = Pseudo_Model(device, CFG)
    losses = model.train_step(y, x, yd, z)
    file_name = model.net_save(".", True)
    model.net_load(file_name)
    for i in range(110000):
        model.lr_decay_step(True)
    info = f"  1/(1):"
    for i, itm in enumerate(losses.items()):
        info += f", {itm[0]}={itm[1]:.3f}" if i > 0 else f" {itm[0]}={itm[1]:.3f}"
    print(info)
    print("fin")


from yacs.config import CfgNode
from torch.utils.data import DataLoader

# Assuming faces_data is already imported and available


import os


# Now you can safely use directory_path in your application

def main():
    # Load configuration
    with open("/Users/noymachluf/Desktop/pseudo-sr2/configs/faces.yaml", "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Pseudo_Model(device, CFG)

    # Set data paths
    high_res_folder = "/Users/noymachluf/Desktop/pseudo-sr2/dataset/train/HIGH"
    low_res_folder = "/Users/noymachluf/Desktop/pseudo-sr2/dataset/train/LOW/wider_lnew"

    # Initialize dataset and dataloader
    dataset = faces_data(low_res_folder, high_res_folder, img_range=CFG.DATA.IMG_RANGE)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Process one batch from the dataloader
    for data in dataloader:
        lr = data['lr'].to(device)            # Low-resolution images
        hr = data.get('hr', None)             # High-resolution images (if available)
        if hr is not None:
            hr = hr.to(device)
        hr_down = data.get('hr_down', None)   # Down-sampled high-resolution images
        if hr_down is not None:
            hr_down = hr_down.to(device)
        z = data.get('z', None)               # Noise (if present in the dataset)
        if z is not None:
            z = z.to(device)

        # Check the shapes for debugging
        print(f'LR shape: {lr.shape}')  # Expected shape: [8, 3, 32, 32] or similar, based on your dataset resizing
        if hr is not None:
            print(f'HR shape: {hr.shape}')
        if hr_down is not None:
            print(f'HR downsampled shape: {hr_down.shape}')
        if z is not None:
            print(f'Noise shape: {z.shape}')

        # Run training step
        losses = model.train_step(hr, lr, hr_down, z)
        break  # Process only one batch for demonstration

    # Optionally, save the model
    file_name = model.net_save("/Users/noymachluf/Desktop/pseudo-sr2", True)
    model.net_load(file_name)

    # Print losses
    info = "Losses:"
    for key, value in losses.items():
        info += f" {key}={value:.3f},"
    print(info[:-1])  # Remove the last comma

if __name__ == "__main__":
    main()
