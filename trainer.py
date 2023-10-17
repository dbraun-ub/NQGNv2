import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks

from networks import *


# to manage CPU usage
torch.set_num_threads(8)

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.max_pool = {}
        for s in self.opt.scales:
            self.max_pool[s] = nn.MaxPool2d(2**s)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"


        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        builder = ModelBuilder()

        if self.opt.load_weights_EPCDepth_encoder is not None:
            self.models["encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
            checkpoint = torch.load(self.opt.load_weights_EPCDepth_encoder, map_location=self.device)
            self.models["encoder"].load_state_dict(checkpoint["encoder"])
        elif self.opt.arch_encoder == "resnet18":
            self.models["encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        else:
            self.models["encoder"] = builder.build_encoder(arch=self.opt.arch_encoder)
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())


        if self.opt.arch_encoder == "mobilenetv3":
            ch = [160,112,40,24,16,16]
        elif self.opt.arch_encoder == "mobilenetv2":
            ch = [320,96,32,24,16,16]
        else:
            ch = [512,256,128,64,64,64]

        if self.opt.arch_decoder == "QGN":
            self.models["depth"] = QGNDepthDecoder(ch)
        elif self.opt.arch_decoder == "dense":
            self.models["depth"] = DepthDecoder(np.array([16, 24, 32, 96, 320]))
        elif self.opt.arch_decoder == "QuadtreeSpConv":
            self.models["depth"] = QuadtreeDepthDecoderSpConv(ch)
        elif self.opt.arch_decoder == "QuadtreeSpConv2":
            self.models["depth"] = QuadtreeDepthDecoderSpConv2(ch, True)
        elif self.opt.arch_decoder == "QuadtreeSpConv3":
            self.models["depth"] = QuadtreeDepthDecoderSpConv3(ch)
        else:
            self.models["depth"] = QuadtreeDepthDecoder(ch)

        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        stereoNet_path = self.opt.stereoNet_path

        # Load stereoNet teacher network
        with torch.no_grad():
            self.models["stereoNet_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, False, 2)
            self.models["stereoNet_encoder"].to(self.device)
            encoder_path = os.path.join(stereoNet_path, "encoder.pth")
            encoder_dict = torch.load(encoder_path, map_location=torch.device(self.device))
            model_dict = self.models["stereoNet_encoder"].state_dict()
            self.models["stereoNet_encoder"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            self.models["stereoNet_encoder"].eval()

            self.models["stereoNet_decoder"] = networks.DepthDecoder(
                self.models["stereoNet_encoder"].num_ch_enc)
            self.models["stereoNet_decoder"].to(self.device)
            decoder_path = os.path.join(stereoNet_path, "depth.pth")
            self.models["stereoNet_decoder"].load_state_dict(torch.load(decoder_path, map_location=torch.device(self.device)))
            self.models["stereoNet_decoder"].eval()


        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # Load dataset
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext)
        if self.opt.val:
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.huberloss = torch.nn.SmoothL1Loss()
        self.bceloss = torch.nn.BCELoss()
        self.mseloss = torch.nn.MSELoss()
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                self.log("train", inputs, outputs, losses)
                if self.opt.val:
                    self.val()

            self.step += 1

    def process_batch(self, inputs, validation=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        with torch.no_grad():
            # ref = self.models["stereoNet_decoder"](self.models["stereoNet_encoder"](torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)))
            ref = self.models["stereoNet_decoder"](self.models["stereoNet_encoder"](torch.cat([inputs[("color_aug", 0, 0)],inputs[("color_aug", "s", 0)]], 1)))
            outputs[("disp_ref",0)] = ref[("disp", 0)]
            outputs = self.compute_quadtree_mask_ref(outputs)
            outputs[("quad_mask_ref", 5)] = torch.ones(self.opt.batch_size, 1, self.opt.height//2**6, self.opt.width//2**6).to(self.device)
            if validation:
                labels = None
            elif self.opt.use_labels:
                if "QuadtreeLight" in self.opt.arch_decoder:
                    labels = [outputs[("quad_mask_ref", 2)],
                              outputs[("quad_mask_ref", 1)],
                              outputs[("quad_mask_ref", 0)]]
                elif self.opt.crit < 0:
                    labels = [torch.ones(self.opt.batch_size, 1, self.opt.height//2**5, self.opt.width//2**5).to(self.device),
                              torch.ones(self.opt.batch_size, 1, self.opt.height//2**4, self.opt.width//2**4).to(self.device),
                              torch.ones(self.opt.batch_size, 1, self.opt.height//2**3, self.opt.width//2**3).to(self.device),
                              torch.ones(self.opt.batch_size, 1, self.opt.height//2**2, self.opt.width//2**2).to(self.device),
                              torch.ones(self.opt.batch_size, 1, self.opt.height//2**1, self.opt.width//2**1).to(self.device),]
                else:
                    labels = [outputs[("quad_mask_ref", 4)],
                              outputs[("quad_mask_ref", 3)],
                              outputs[("quad_mask_ref", 2)],
                              outputs[("quad_mask_ref", 1)],
                              outputs[("quad_mask_ref", 0)]]
            else:
                labels = None


        features = self.models["encoder"](inputs["color_aug", 0, 0])

        if self.opt.arch_decoder == "dense":
            outputs = self.models["depth"](features)
            outputs[("disp_ref",0)] = ref[("disp", 0)]
            for i in self.opt.scales:
                outputs[("disp_dense",i)] = outputs[("disp",i)]
        else:
            quad, mask = self.models["depth"](features, labels, crit=self.opt.crit)

            outputs[("disp", 0)] = quad[5]
            outputs[("disp", 1)] = quad[4]
            outputs[("disp", 2)] = quad[3]
            outputs[("disp", 3)] = quad[2]
            outputs[("disp", 4)] = quad[1]
            outputs[("disp", 5)] = quad[0]

            outputs[("quad_mask", 5)] = torch.ones(self.opt.batch_size, 1, 3, 10).to(self.device)
            outputs[("quad_mask", 4)] = mask[0]
            outputs[("quad_mask", 3)] = mask[1]
            outputs[("quad_mask", 2)] = mask[2]
            outputs[("quad_mask", 1)] = mask[3]
            outputs[("quad_mask", 0)] = mask[4]

            if validation:
                outputs = self.densify_disp_val(outputs)
            else:
                outputs = self.densify_disp(outputs, self.opt.quadtree_interp)

        self.generate_images_pred(inputs, outputs)
        if self.opt.arch_decoder == "dense":
            losses = self.compute_losses_dense(outputs, validation)
        else:
            losses = self.compute_losses(inputs, outputs, validation)

        return outputs, losses


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp_dense", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs, validation=True)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def densify_disp(self, outputs, mode="nearest"):
        # densify disparity maps
        outputs[("disp_dense", 5)] = outputs[("disp", 5)].clone()

        for i in range(4,-1,-1):
            disp = outputs[("disp", i)].clone()
            if self.opt.use_labels:
                mask = self.up(outputs[("quad_mask_ref", i)].clone())
            else:
                mask = torch.round(self.up(outputs[("quad_mask", i)].clone()))
            outputs[("disp_dense", i)] = F.interpolate(outputs[("disp_dense", i+1)], scale_factor=2, mode=mode) * (1-mask) + disp * mask

        return outputs

    def densify_disp_light(self, outputs, mode="nearest"):
        # densify disparity maps
        outputs[("disp_dense", 3)] = outputs[("disp", 3)].clone()

        for i in range(2,-1,-1):
            disp = outputs[("disp", i)].clone()
            if self.opt.use_labels:
                mask = self.up(outputs[("quad_mask_ref", i)].clone())
            else:
                mask = torch.round(self.up(outputs[("quad_mask", i)].clone()))
            outputs[("disp_dense", i)] = F.interpolate(outputs[("disp_dense", i+1)], scale_factor=2, mode=mode) * (1-mask) + disp * mask

        return outputs

    def densify_disp_val(self, outputs):
        # densify disparity maps
        outputs[("disp_dense", 5)] = outputs[("disp", 5)].clone()

        for i in range(4,-1,-1):
            disp = outputs[("disp", i)].clone()
            mask = torch.round(self.up(outputs[("quad_mask", i)].clone()))
            outputs[("disp_dense", i)] = self.up(outputs[("disp_dense", i+1)]) * (1-mask) + disp * mask

        return outputs

    def densify_disp_val_light(self, outputs):
        # densify disparity maps
        outputs[("disp_dense", 3)] = outputs[("disp", 3)].clone()

        for i in range(2,-1,-1):
            disp = outputs[("disp", i)].clone()
            mask = torch.round(self.up(outputs[("quad_mask", i)].clone()))
            outputs[("disp_dense", i)] = self.up(outputs[("disp_dense", i+1)]) * (1-mask) + disp * mask

        return outputs

    def compute_quadtree_mask_ref(self, outputs):
        _, depth = disp_to_depth(outputs[("disp_ref",0)].clone(), self.opt.min_depth, self.opt.max_depth)
        depth *= 5.4
        depth[depth>50] = 50.0
        disp = 1 / depth

        for i in range(5):
            s = 2**(i+1)
            a = torch.zeros(self.opt.batch_size, s**2, self.opt.height//s, self.opt.width//s).to(self.device)

            for j in range(s**2):
                a[:,j,:,:] = disp[:,0,(j//s)::s,(j%s)::s]

            a = torch.std(a, dim=1, keepdim=True) > self.opt.crit
            outputs[("quad_mask_ref", i)] = a.type(torch.FloatTensor).to(self.device)

        return outputs


    def depth_quadtree_mask_ref(self, outputs):
        _, depth = disp_to_depth(outputs[("disp_ref",0)], self.opt.min_depth, self.opt.max_depth)
        depth *= 5.4 #apply stereo scale
        depth[depth>80] = 80.0


        for i in range(5):
            s = 2**(i+1)
            a = torch.zeros(self.opt.batch_size, s**2, self.opt.height//s, self.opt.width//s).to(self.device)

            for j in range(s**2):
                a[:,j,:,:] = depth[:,0,(j//s)::s,(j%s)::s]

            a = 20 / np.log(20) * torch.log(a)
            a = torch.std(a, dim=1, keepdim=True) > self.opt.crit
            outputs[("quad_mask_ref", i)] = a.type(torch.FloatTensor).to(self.device)

        return outputs

    def get_densified_disp(self, outputs, scale):
        disp = outputs[("disp", 5)].clone()

        for i in range(4,scale-1,-1):
            d = outputs[("disp", i)].clone()
            if self.opt.use_labels:
                mask = self.up(outputs[("quad_mask_ref", i)].clone())
            else:
                mask = torch.round(self.up(outputs[("quad_mask", i)].clone()))

            disp = F.interpolate(disp, scale_factor=2, mode="bilinear") * (1-mask) + d * mask

        return disp

    def log_depth_error(self, disp_pred, disp_ref):
        # Based on the loss from chiu2020icpr
        m = self.opt.min_depth
        M = self.opt.max_depth

        depth_pred = 1 / disp_pred
        depth_ref = 1 / disp_ref

        G_pred = (torch.log(depth_pred+1) - np.log(m+1)) * M / (np.log(M+1) - np.log(m+1))
        G_ref = (torch.log(depth_ref+1) - np.log(m+1)) * M / (np.log(M+1) - np.log(m+1))

        return self.huberloss(G_pred, G_ref)



    def compute_losses(self, inputs, outputs, validation=False):
        losses = {}
        total_loss = 0


        disp_ref = outputs[("disp_ref", 0)]
        disp_ref, depth_ref = disp_to_depth(disp_ref, self.opt.min_depth, self.opt.max_depth)
        _,_,h,w = disp_ref.size()
        for scale in self.opt.scales:
            loss = 0

            if validation:
                quad_mask = outputs[("quad_mask", scale)]
            elif self.opt.use_labels:
                quad_mask = outputs[("quad_mask_ref", scale)]
            else:
                quad_mask = outputs[("quad_mask", scale)]


            mask = F.interpolate(quad_mask, size=(h,w), mode="nearest") > 0.5

            # Quadtree loss
            if scale != self.opt.scales[-1] and torch.any(quad_mask != 0.5):
                quad_mask = outputs[("quad_mask", scale)]
                quad_mask_ref = outputs[("quad_mask_ref", scale)]
                loss_quadtree = self.bceloss(quad_mask[quad_mask != 0.5], quad_mask_ref[quad_mask != 0.5])
                losses["loss/quadtree/{}".format(scale)] = loss_quadtree
                loss += self.opt.coef_quadtree * loss_quadtree

            # loss v1
            if torch.any(mask):
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, size=(h,w), mode="nearest")
                disp, _ = disp_to_depth(disp[mask], self.opt.min_depth, self.opt.max_depth)
                loss_huber = torch.log(torch.abs(disp - disp_ref[mask]) + 1).mean()
                losses["loss/logL1/{}".format(scale)] = loss_huber
                loss += self.opt.coef_l1 * loss_huber

            loss_monodepth = self.compute_monodepth_losses(inputs, outputs, scale)
            losses["loss/self/{}".format(scale)] = self.opt.coef_rep * loss_monodepth
            loss += loss_monodepth

            total_loss += loss

        losses["loss"] = total_loss

        return losses

    def compute_monodepth_losses(self, inputs, outputs, scale):
        loss = 0
        reprojection_losses = []

        if self.opt.v1_multiscale:
            source_scale = scale
        else:
            source_scale = 0

        disp = outputs[("disp_dense", scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in self.opt.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not self.opt.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not self.opt.disable_automasking:
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

        return loss

    def compute_losses_dense(self, outputs, validation=False):
        losses = {}
        total_loss = 0


        disp_ref = outputs[("disp_ref", 0)]
        disp_ref, depth_ref = disp_to_depth(disp_ref, self.opt.min_depth, self.opt.max_depth)
        _,_,h,w = disp_ref.size()
        for scale in self.opt.scales:
            loss = 0

            losses["loss/quadtree/{}".format(scale)] = 0

            # loss v1
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, size=(h,w), mode="nearest")
            disp, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            loss_huber = torch.log(torch.abs(disp - disp_ref) + 1).mean()
            losses["loss/logL1/{}".format(scale)] = loss_huber
            loss += self.opt.coef_l1 * loss_huber

            total_loss += loss

        losses["loss"] = total_loss

        return losses


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_quadtree_loss(self, outputs, rep_error):
        a = torch.argmin(rep_error, 1, keepdims=True)
        loss = 0

        for scale in self.opt.scales[:-1]:
            quad_mask = F.interpolate(outputs[("quad_mask", scale)].clone(), scale_factor=2**(scale+1), mode="nearest")
            # Values at 0.5 are : A(0) = sigmoid(0) = 0.5
            mask_active_nodes = quad_mask != 0.5
            r = (torch.rand(a.size()).to(self.device) > (0.5/(self.epoch+1))).type(quad_mask.dtype)
            target = torch.clamp((scale >= a).type(quad_mask.dtype) + r,0,1).to(self.device)

            loss += self.bceloss(quad_mask[mask_active_nodes], target[mask_active_nodes]) / 2**scale

        return loss

    def get_quadtree_loss(self, outputs, rep_error):
        b = self.opt.batch_size
        h = self.opt.height
        w = self.opt.width
        disp, _ = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        disp = disp.detach()
        rep_error = rep_error.detach()
        rep_error = rep_error * disp.squeeze(1)**2

        quadtree_loss_1 = 0
        quadtree_loss_2 = 0

        for i in range(5):
            s = 2**(i+1)
            e = torch.zeros(b, s**2, h//s, w//s).to(self.device)
            a = torch.zeros(b, s**2, h//s, w//s).to(self.device)
            mask = outputs[("quad_mask", i)] != 0.5

            for j in range(s**2):
                e[:,j,:,:] = rep_error[:, (j//s)::s, (j%s)::s]
                a[:,j,:,:] = disp[:, 0, (j//s)::s, (j%s)::s]

            e_max = torch.max(e, dim=1, keepdims=True).values
            e_min = torch.min(e, dim=1, keepdims=True).values
            _,d = disp_to_depth(outputs[("disp", i+1)], self.opt.min_depth, self.opt.max_depth)
            e = (((e_max - e_min) * d**2) > self.opt.crit).type(torch.FloatTensor).to(self.device)

            m_max = torch.max(a, dim=1, keepdims=True).values
            m_min = torch.min(a, dim=1, keepdims=True).values
            m = ((m_max**2 - m_min**2) > self.opt.crit).type(torch.FloatTensor).to(self.device)

            # if torch.all(mask == False):
            quadtree_loss_1 += self.bceloss(outputs[("quad_mask", i)][mask], e[mask])
            quadtree_loss_2 += self.bceloss(outputs[("quad_mask", i)][mask], m[mask])

        return quadtree_loss_1 / 5, quadtree_loss_2 / 5

    def compute_mask_quadtree(self, disp, scale, criterion):
        # scale disp
        disp, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        b,_,h,w = disp.size()

        s = 2**scale
        a = torch.zeros(b, s**2, h//s, w//s)

        for j in range(s**2):
            a[:,j,:,:] = disp[:, 0, (j//s)::s, (j%s)::s]


        mask = (torch.max(a, axis=1, keepdims=True).values**2 - torch.min(a, axis=1).values**2) > criterion
        mask = mask.type(torch.FloatTensor).to(self.device)

        return mask

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """

        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image(
                "disp_ref/{}".format(j),
                normalize_image(outputs[("disp_ref", 0)][j]), self.step)

            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}/{}".format(frame_id, j),
                    inputs[("color", frame_id, 0)][j].data, self.step)

            for s in self.opt.scales:

                writer.add_image(
                    "quad_disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.arch_decoder != "dense":
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp_dense", s)][j]), self.step)

                    if s != self.opt.scales[-1]:
                        writer.add_image(
                            "quad_mask_{}/{}".format(s, j),
                            outputs[("quad_mask", s)][j], self.step)
                        writer.add_image(
                            "quad_mask_ref_{}/{}".format(s, j),
                            outputs[("quad_mask_ref", s)][j], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
