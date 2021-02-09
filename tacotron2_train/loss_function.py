# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

# import torch
from torch import nn

# class GuidedAttentionLoss(torch.nn.Module):
#     def __init__(self, sigma=0.4):
#         super(GuidedAttentionLoss, self).__init__()
#         self.sigma = sigma

#     def _make_ga_masks(self, ilens, olens):
#         B = len(ilens)
#         max_ilen = max(ilens)
#         max_olen = max(olens)
#         ga_masks = torch.zeros((B, max_olen, max_ilen))
#         for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
#             ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
#         return ga_masks

#     def forward(self, att_ws, ilens, olens):
#         ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
#         seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
#         losses = ga_masks * att_ws
#         loss = torch.mean(losses.masked_select(seq_masks))
#         return loss

#     @staticmethod
#     def _make_ga_mask(ilen, olen, sigma):
#         grid_x, grid_y = torch.meshgrid(
#             torch.arange(olen).to(olen), torch.arange(ilen).to(ilen)
#         )
#         grid_x, grid_y = grid_x.float(), grid_y.float()
#         return 1.0 - torch.exp(
#             -(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2))
#         )

#     @staticmethod
#     def _make_masks(ilens, olens):
#         in_masks = sequence_mask(ilens)
#         out_masks = sequence_mask(olens)
#         return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


# -----------------------------------------------------------------------------


class Tacotron2Loss(nn.Module):
    def forward(self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(
            mel_out_postnet, mel_target
        )
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        final_loss = mel_loss + gate_loss

        # guided attention loss (if enabled)
        # if self.config.ga_alpha > 0:
        #     ga_loss = self.criterion_ga(alignments, input_lens, alignment_lens)
        #     final_loss += ga_loss * self.ga_alpha

        return final_loss


# -----------------------------------------------------------------------------

LossType = Tacotron2Loss
