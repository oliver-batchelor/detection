import torch

def class_loss(heatmap, prediction, gamma=2.0):
    """
    Focal loss variant of BCE as used in CornerNet and CenterNet.
    """

    # As per RetinaNet focal loss - if heatmap == 1
    pos_loss = -prediction.log() * (1 - pred).pow(gamma)

    # Negative penalty very small around gaussian near centre
    neg_weights = (1 - heatmap).pow(4) 
    neg_loss = -(1 - pred).log() * pred.pow(gamma) * neg_weights

    return torch.where(target == 1, pos_loss, neg_loss)

# def batch_loss(self, target, prediction):
#         """

#         """
#         H, W = pred_hm.shape[2:]
#         pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
#         hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

#         mask = wh_weight.view(-1, H, W)
#         avg_factor = mask.sum() + 1e-4

#         if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
#             base_step = self.down_ratio
#             shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
#                                     dtype=torch.float32, device=heatmap.device)
#             shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
#                                     dtype=torch.float32, device=heatmap.device)
#             shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#             self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

#         # (batch, h, w, 4)
#         pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
#                                 self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
#         # (batch, h, w, 4)
#         boxes = box_target.permute(0, 2, 3, 1)
#         wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight

#         return hm_loss, wh_loss