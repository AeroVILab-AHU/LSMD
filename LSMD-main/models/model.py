import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2
from .CAIM import TransformerFusionBlock as CAIM

    
class NeighborContextEnhancement(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NeighborContextEnhancement, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2   # e.g. 32
        self.out_d = out_d        # e.g. 64
        self.alpha_s3 = nn.Parameter(torch.tensor(0.1))
        self.alpha_s4 = nn.Parameter(torch.tensor(0.1))
        self.alpha_s5 = nn.Parameter(torch.tensor(0.1))

        # --- scale2 (unchanged) ---
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, self.in_d[1], self.out_d)

        # ---------------------------
        # scale3: keep conv_scale3_c2 producing mid_d,
        # but add adapter to map s2 (out_d) -> mid_d before addition
        # ---------------------------
        self.conv_scale3_c2 = nn.Sequential(
            # operate on original c2 (size HxW), then maxpool in forward to match s2_down resolution
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)

        # adapter convs: map s2_down/s3_down/s4_down (out_d channels) -> mid_d channels
        self.adapter_s2_to_mid = nn.Sequential(
            nn.Conv2d(self.out_d, self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.adapter_s3_to_mid = nn.Sequential(
            nn.Conv2d(self.out_d, self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.adapter_s4_to_mid = nn.Sequential(
            nn.Conv2d(self.out_d, self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        # ---------------------------
        # scale4 (keep producing mid_d)
        # ---------------------------
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)

        # ---------------------------
        # scale5
        # ---------------------------
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')
        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)  # s2: [B, out_d, H, W]

        # scale 3
        # process c2 path -> mid_d and downsample (inside conv)
        c2_s3 = self.conv_scale3_c2(c2)  # [B, mid_d, H/2, W/2]
        # downsample s2 and map out_d -> mid_d before addition
        s2_down = F.max_pool2d(s2, kernel_size=2, stride=2)  # [B, out_d, H/2, W/2]
        s2_down_mapped = self.adapter_s2_to_mid(s2_down)     # [B, mid_d, H/2, W/2]
        c2_s3 = (1 - self.alpha_s3) * c2_s3 + self.alpha_s3 * s2_down_mapped

        c3_s3 = self.conv_scale3_c3(c3)                       # [B, mid_d, H/2, W/2]
        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')  # up to H/2
        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)  # s3: [B, out_d, H/2, W/2]

        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)  # [B, mid_d, H/4, W/4]
        s3_down = F.max_pool2d(s3, kernel_size=2, stride=2)  # [B, out_d, H/4, W/4]
        s3_down_mapped = self.adapter_s3_to_mid(s3_down)     # [B, mid_d, H/4, W/4]
        c3_s4 = (1 - self.alpha_s4) * c3_s4 + self.alpha_s4 * s3_down_mapped

        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')
        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)  # s4: [B, out_d, H/4, W/4]

        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)  # [B, mid_d, H/8, W/8]
        s4_down = F.max_pool2d(s4, kernel_size=2, stride=2)  # [B, out_d, H/8, W/8]
        s4_down_mapped = self.adapter_s4_to_mid(s4_down)     # [B, mid_d, H/8, W/8]
        c4_s5 = (1 - self.alpha_s5) * c4_s5 + self.alpha_s5 * s4_down_mapped

        c5_s5 = self.conv_scale5_c5(c5)
        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        # self.conv_identity = nn.Sequential(
        #     nn.Conv2d(self.id_d, self.out_d, kernel_size=1),
        #     nn.BatchNorm2d(self.out_d),
        #     nn.ReLU(inplace=True)
        # )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x):
        # temporal fusion
        # x = torch.abs(x1 - x2)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out

class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(TemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.stfm_x2 = SpatioTemporalFusionModule(self.in_d, self.out_d)
        self.stfm_x3 = SpatioTemporalFusionModule(self.in_d, self.out_d)
        self.stfm_x4 = SpatioTemporalFusionModule(self.in_d, self.out_d)
        self.stfm_x5 = SpatioTemporalFusionModule(self.in_d, self.out_d)

    def forward(self, fused_2, fused_3, fused_4, fused_5):
        c2 = self.stfm_x2(fused_2)
        c3 = self.stfm_x3(fused_3)
        c4 = self.stfm_x4(fused_4)
        c5 = self.stfm_x5(fused_5)

        return c2, c3, c4, c5

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask
    
class SpatioTemporalFusionModule(torch.nn.Module):
    def __init__(self, in_d,out_d):
        super(SpatioTemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.tffm = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.query_proj = nn.Conv2d(out_d, out_d, kernel_size=1)
        self.key_proj = nn.Conv2d(out_d, out_d, kernel_size=1)
        self.value_proj = nn.Conv2d(out_d, out_d, kernel_size=1)
        self.out_proj = nn.Conv2d(out_d, out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv

    def forward(self, x):
        _, _, h, w = x.size()
        identity = x  # ---- 保存残差
        fused = self.tffm(x)

        # q = x.mean(dim=[2, 3], keepdim=True)
        # k = self.proj(x)
        # k = x
        q = self.query_proj(fused).mean(dim=[2, 3], keepdim=True)
        k = self.key_proj(fused)
        v = self.value_proj(fused)
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        out = v * att_weight
        out = self.out_proj(out)
        out = self.relu(out + identity)
        # print(sigma)

        return out

class SE_Block(nn.Module):
    def __init__(self, channel, is_dis=False):
        super(SE_Block, self).__init__()

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.is_dis = is_dis

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.AvgPool(x).view(b, c)
        y = self.FC(y).view(b, c, 1, 1)
        out = x * y

        if self.is_dis is True:
            out = y

        return out
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.PReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Prelu is True:
            out_F = self.PReLU(out)
            return out_F

class SMRM(nn.Module):
    def __init__(self, channel, is_last=False):
        super(SMRM, self).__init__()

        self.is_last = is_last

        self.SE_Block = SE_Block(channel, is_dis=False)
        self.Conv1_1 = ConvLayer(channel, channel, 3, 1, True)
        self.Conv1_2 = ConvLayer(channel, channel, 3, 1, True)

    def forward(self, Fused_Image, diff_nir, diff_rgb, d):
    # compute diff fusion & Fused_Image projection
        add_diff = self.Conv1_1(diff_nir + diff_rgb)
        add_d = self.Conv1_1(d)
        Fused_Image = self.Conv1_1(Fused_Image)

    # ensure spatial sizes consistent (target = add_diff)
        target_h, target_w = add_diff.shape[2], add_diff.shape[3]

        if add_d.shape[2] != target_h or add_d.shape[3] != target_w:
            add_d = F.interpolate(add_d, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)

        if Fused_Image.shape[2] != target_h or Fused_Image.shape[3] != target_w:
            Fused_Image = F.interpolate(Fused_Image, size=(target_h, target_w),
                                   mode='bilinear', align_corners=False)
        w = self.SE_Block(Fused_Image)
        if w.dim() == 2 or (w.dim() == 3 and w.shape[2] == 1):
        # handle unexpected shapes robustly
            w = w.view(w.size(0), w.size(1), 1, 1)
            w = w.expand(-1, -1, target_h, target_w)

    # safety: ensure w matches target
        if w.shape[2] != target_h or w.shape[3] != target_w:
            w = F.interpolate(w, size=(target_h, target_w),
                          mode='bilinear', align_corners=False)

    # now shapes: add_diff, add_d, w all have same spatial size
        add_all = w * add_diff + (1.0 - w) * add_d + Fused_Image

        out = self.Conv1_2(add_all)

        # if self.is_last is True:
        #     out = self.Conv4_2(self.Up(add_all))

        return out


#Saliency-Guided Multi-source Enhancement
class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d

        # p5阶段加入显著性mask信息
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.mid_d + 2, self.mid_d, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.sgmde_p4 = SMRM(self.mid_d)
        self.sgmde_p3 = SMRM(self.mid_d)
        self.sgmde_p2 = SMRM(self.mid_d, is_last=True)  # 最后输出预测

        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5, mask_A, mask_B, diff_rgb_2, diff_rgb_3, diff_rgb_4, diff_rgb_5, diff_nir_2, diff_nir_3, diff_nir_4, diff_nir_5):

        """
        Args:
            d2, d3, d4, d5: 主干特征
            diff_nir_list: [diff_nir_2, diff_nir_3, diff_nir_4, diff_nir_5]
            diff_rgb_list: [diff_rgb_2, diff_rgb_3, diff_rgb_4, diff_rgb_5]
            mask_A, mask_B: 输入掩码
        """

        # 1️⃣ 高层特征显著性融合
        mask_A_resize = F.interpolate(mask_A, size=d5.shape[2:], mode="bilinear", align_corners=False)
        mask_B_resize = F.interpolate(mask_B, size=d5.shape[2:], mode="bilinear", align_corners=False)
        p5_aug = torch.cat([d5, mask_A_resize, mask_B_resize], dim=1)
        p5 = self.conv1x1(p5_aug)


        mask_p5 = self.cls(p5)

        # 2️⃣ 多源差异增强解码
        p4 = self.sgmde_p4(d4, diff_rgb_4, diff_nir_4, p5)
        mask_p4 = self.cls(p4)

        p3 = self.sgmde_p3(d3, diff_rgb_3, diff_nir_3, p4)
        mask_p3 = self.cls(p3)

        p2 = self.sgmde_p2(d2, diff_rgb_2, diff_nir_2, p3)
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5



class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.rgb_backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        self.nir_backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2

        # NCEM
        self.rgb_nfe = NeighborContextEnhancement(channles, self.mid_d)
        self.nir_nfe = NeighborContextEnhancement(channles, self.mid_d)

        #CAIM
        self.mmfm_2 = CAIM(d_model=64, vert_anchors=16, horz_anchors=16, h=8, n_layer=1)
        self.mmfm_3 = CAIM(d_model=64, vert_anchors=8, horz_anchors=8, h=8, n_layer=1)
        self.mmfm_4 = CAIM(d_model=64, vert_anchors=4, horz_anchors=4, h=8, n_layer=1)
        self.mmfm_5 = CAIM(d_model=64, vert_anchors=2, horz_anchors=2, h=8, n_layer=1)
        # SMRM
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, rgb_pre, rgb_post, nir_pre, nir_post, mask_A, mask_B):
        # forward backbone resnet
        rgb_pre_1, rgb_pre_2, rgb_pre_3, rgb_pre_4, rgb_pre_5 = self.rgb_backbone(rgb_pre)
        rgb_post_1, rgb_post_2, rgb_post_3, rgb_post_4, rgb_post_5 = self.rgb_backbone(rgb_post)
        nir_pre_1, nir_pre_2, nir_pre_3, nir_pre_4, nir_pre_5 = self.nir_backbone(nir_pre)
        nir_post_1, nir_post_2, nir_post_3, nir_post_4, nir_post_5 = self.nir_backbone(nir_post)

        # # NCEM
        rgb_pre_2, rgb_pre_3, rgb_pre_4, rgb_pre_5 = self.rgb_nfe(rgb_pre_2, rgb_pre_3, rgb_pre_4, rgb_pre_5)
        rgb_post_2, rgb_post_3, rgb_post_4, rgb_post_5 = self.rgb_nfe(rgb_post_2, rgb_post_3, rgb_post_4, rgb_post_5)
        nir_pre_2, nir_pre_3, nir_pre_4, nir_pre_5 = self.nir_nfe(nir_pre_2, nir_pre_3, nir_pre_4, nir_pre_5)
        nir_post_2, nir_post_3, nir_post_4, nir_post_5 = self.nir_nfe(nir_post_2, nir_post_3, nir_post_4, nir_post_5)

        # 3. 计算每个模态时相差分（多个尺度）
        diff_rgb_2 = torch.abs(rgb_pre_2 - rgb_post_2)
        diff_rgb_3 = torch.abs(rgb_pre_3 - rgb_post_3)
        diff_rgb_4 = torch.abs(rgb_pre_4 - rgb_post_4)
        diff_rgb_5 = torch.abs(rgb_pre_5 - rgb_post_5)
        #
        diff_nir_2 = torch.abs(nir_pre_2 - nir_post_2)
        diff_nir_3 = torch.abs(nir_pre_3 - nir_post_3)
        diff_nir_4 = torch.abs(nir_pre_4 - nir_post_4)
        diff_nir_5 = torch.abs(nir_pre_5 - nir_post_5)

        # CAIM
        c2 = self.mmfm_2([diff_rgb_2, diff_nir_2])
        c3 = self.mmfm_3([diff_rgb_3, diff_nir_3])
        c4 = self.mmfm_4([diff_rgb_4, diff_nir_4])
        c5 = self.mmfm_5([diff_rgb_5, diff_nir_5])

        # SMRM
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5, mask_A, mask_B,diff_rgb_2, diff_rgb_3, diff_rgb_4, diff_rgb_5, diff_nir_2, diff_nir_3, diff_nir_4, diff_nir_5)


        # change map
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)
        # return mask_p1, mask_p2, mask_p3, mask_p4, mask_p5
        return mask_p2, mask_p3, mask_p4, mask_p5
