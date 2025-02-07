"""
Original work Copyright 2019 Davy Neven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler (MIT License)
Modifications: changed IOU calculation; extended with tracking loss
"""





import torch
import torch.nn as nn
from embedtrack.criterions.lovasz_losses import lovasz_hinge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbedTrackLoss(nn.Module):
    def __init__(
        self,
        cluster,
        grid_y=1024,
        grid_x=1024,
        pixel_y=1,
        pixel_x=1,
        n_sigma=2,
        foreground_weight=1,
    ):
        super().__init__()
        print(
            "Created spatial emb loss function with: n_sigma: {}, foreground_weight: {}".format(
                n_sigma, foreground_weight
            )
        )
        self.cluster = cluster
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # Coordinate map
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        yxm = torch.cat((ym, xm), 0)

        self.register_buffer("yxm", yxm)
        self.register_buffer("yx_shape", torch.tensor(self.yxm.size()[1:]).view(-1, 1))

    def forward(
        self,
        predictions,
        instances,
        labels,
        center_images,
        offsets,
        w_inst=1,
        w_var=10,
        w_seed=1,
        iou=False,
        iou_meter=None,
    ):
        segmentation_predictions, offset_predictions = predictions
        batch_size, height, width = (
            segmentation_predictions.size(0),
            segmentation_predictions.size(2),
            segmentation_predictions.size(3),
        )

        yxm_s = self.yxm[:, 0:height, 0:width]

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        track_loss = torch.tensor(0.0, device=device, requires_grad=True)
        track_count = torch.tensor(0.0, device=device)

        loss_values = {
            "instance": torch.tensor(0.0, device=device, requires_grad=True),
            "variance": torch.tensor(0.0, device=device, requires_grad=True),
            "seed": torch.tensor(0.0, device=device, requires_grad=True),
            "track": torch.tensor(0.0, device=device, requires_grad=True),
        }

        for b in range(0, min(batch_size, offset_predictions.size(0))):
            seed_loss_it = torch.tensor(0.0, device=device, requires_grad=True)
            seed_loss_count = torch.tensor(0.0, device=device)

            segm_offsets = torch.tanh(segmentation_predictions[b, 0:2])
            spatial_emb = segm_offsets + yxm_s

            if b < batch_size // 2:
                track_offsets = torch.tanh(offset_predictions[b, ...])
                tracking_emb = yxm_s - track_offsets

            sigma = torch.sigmoid(segmentation_predictions[b, 2 : 2 + self.n_sigma])
            seed_map = torch.sigmoid(
                segmentation_predictions[b, 2 + self.n_sigma : 2 + self.n_sigma + 1]
            )

            var_loss = torch.tensor(0.0, device=device, requires_grad=True)
            instance_loss = torch.tensor(0.0, device=device, requires_grad=True)
            seed_loss = torch.tensor(0.0, device=device, requires_grad=True)

            instance = instances[b].unsqueeze(0)
            label = labels[b].unsqueeze(0)
            center_image = center_images[b].unsqueeze(0)

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss_it = seed_loss_it + torch.mean(torch.pow(seed_map[bg_mask], 2))
                seed_loss_count = seed_loss_count + bg_mask.sum()

            if len(instance_ids) == 0:
                continue

            all_sigmas = torch.stack(
                [
                    sigma[:, instance.eq(inst_id).squeeze()].mean(dim=1)
                    for inst_id in instance_ids
                ]
            ).T

            for i, inst_id in enumerate(instance_ids):
                in_mask = instance.eq(inst_id)
                center_mask = in_mask & center_image.bool()

                if center_mask.sum().eq(0):
                    continue
                if center_mask.sum().eq(1):
                    center = yxm_s[center_mask.expand_as(yxm_s)].view(2, 1, 1)
                else:
                    xy_in = yxm_s[in_mask.expand_as(yxm_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)

                sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)
                s = all_sigmas[:, i].view(self.n_sigma, 1, 1)

                var_loss = var_loss + torch.mean(torch.pow(sigma_in - s.detach(), 2))
                s = torch.exp(s * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True)
                )

                instance_loss = instance_loss + lovasz_hinge(dist * 2 - 1, in_mask.to(device))
                seed_loss_it = seed_loss_it + self.foreground_weight * torch.mean(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                )
                seed_loss_count = seed_loss_count + in_mask.sum()

                if b < (batch_size // 2):
                    index_gt_center = (in_mask & center_image.bool()).squeeze()
                    if index_gt_center.sum().eq(0):
                        continue
                    gt_prev_center_yxms = (
                        (
                            yxm_s[:, index_gt_center]
                            - offsets[b, :, index_gt_center] / self.yx_shape
                        )
                        .view(-1, 1, 1)
                        .float()
                    )

                    # Adjust shapes to be compatible
                    tracking_emb = self.align_shapes(tracking_emb, gt_prev_center_yxms)

                    dist_tracking = torch.exp(
                        -1 * torch.sum(torch.pow(tracking_emb - gt_prev_center_yxms, 2) * s, 0, keepdim=True)
                    )
                    track_loss = track_loss + lovasz_hinge(dist_tracking * 2 - 1, in_mask.to(device))
                    track_count = track_count + 1

            seed_loss = seed_loss + seed_loss_it

            if iou:
                instance_pred = self.cluster.cluster_pixels(
                    segmentation_predictions[b], n_sigma=2, return_on_cpu=False
                )
                iou_scores = calc_iou_full_image(instances[b].detach(), instance_pred.detach())
                for score in iou_scores:
                    iou_meter.update(score)

            loss = loss + w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss
            loss_values["instance"] = loss_values["instance"] + w_inst * instance_loss
            loss_values["variance"] = loss_values["variance"] + w_var * var_loss
            loss_values["seed"] = loss_values["seed"] + w_seed * seed_loss

        if track_count > 0:
            track_loss = track_loss / track_count
        loss = loss + track_loss
        loss_values["track"] = loss_values["track"] + track_loss

        loss = loss / (b + 1)

        return loss, loss_values

    def align_shapes(self, tensor_a, tensor_b):
        # Adjust tensor_a dimensions to match tensor_b
        while tensor_a.dim() > tensor_b.dim():
            tensor_a = tensor_a.squeeze(0)
        while tensor_a.dim() < tensor_b.dim():
            tensor_a = tensor_a.unsqueeze(-1)
        # Ensure sizes align, especially along spatial dimensions
        min_dims = min(tensor_a.dim(), tensor_b.dim())
        size_a = tensor_a.size()
        size_b = tensor_b.size()
        new_size = [min(sa, sb) for sa, sb in zip(size_a[-min_dims:], size_b[-min_dims:])]
        if new_size != list(size_a[-min_dims:]):
            tensor_a = torch.nn.functional.interpolate(
                tensor_a.unsqueeze(0),
                size=new_size,
                mode='nearest'
            ).squeeze(0)
        return tensor_a

def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou

def calc_iou_full_image(gt, prediction):
    gt_labels = torch.unique(gt)
    gt_labels = gt_labels[gt_labels > 0]
    pred_labels = prediction[prediction > 0].unique()

    ious = []
    matched_pred_labels = []
    for gt_l in gt_labels:
        gt_mask = gt.eq(gt_l)
        overlapping_pred_labels = prediction[gt_mask].unique()
        overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels > 0]
        if not len(overlapping_pred_labels):
            ious.append(0)
            continue
        gt_ious = torch.tensor(
            [calculate_iou(gt_mask, prediction.eq(p_l)) for p_l in overlapping_pred_labels]
        )
        if len(gt_ious) > 0:
            idx_max_iou = torch.argmax(gt_ious)
            ious.append(gt_ious[idx_max_iou])
            matched_pred_labels.append(overlapping_pred_labels[idx_max_iou])

    if len(matched_pred_labels) > 0:
        matched_pred_labels = torch.stack(matched_pred_labels)
        num_non_matched = (~(pred_labels[..., None] == matched_pred_labels).any(-1)).sum()
    else:
        num_non_matched = len(pred_labels)
    ious.extend([0] * num_non_matched)
    return ious

