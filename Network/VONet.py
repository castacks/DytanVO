# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .PWC import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet

class VONet(nn.Module):
    def __init__(self):
        super(VONet, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()

    def forward(self, x, only_flow=False, only_pose=False):
        '''
        x[0]: rgb frame t-1
        x[1]: rgb frame t
        x[2]: intrinsics
        x[3]: flow t-1 -> t (optional)
        x[4]: motion segmentation mask
        '''
        # import ipdb;ipdb.set_trace()
        if not only_pose:
            flow_out = self.flowNet(torch.cat((x[0], x[1]), dim=1))

            if only_flow:
                return flow_out, None
            
            flow = flow_out[0]

        else:
            assert(len(x) > 3)
            flow_out = None

        if len(x) > 3 and x[3] is not None:
            flow_input = x[3]
        else:
            flow_input = flow

        # Mask out input flow using the segmentation result
        assert(len(x) > 4)
        mask = torch.gt(x[4], 0)
        for i in range(flow_input.shape[0]):
            zeros = torch.cat([mask[i], ]*2, dim=0)
            flow_input[i][zeros] = 0

        flow_input = torch.cat((flow_input, 1 - x[4]), dim=1)  # segmentation layer
        flow_input = torch.cat((flow_input, x[2]), dim=1)  # intrinsics layer
    
        pose = self.flowPoseNet(flow_input)

        return flow_out, pose
