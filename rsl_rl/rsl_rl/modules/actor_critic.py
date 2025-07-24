# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .net_model import *
from rsl_rl.utils import unpad_trajectories

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 output_activation=None,
                 model_name="",
                 NetModel=None,
                 init_noise_std=1.0,
                 max_std = 1.0,
                 min_std = 0.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        self.is_recurrent = False
        self.max_std = max_std
        self.min_std = min_std
        self.model_name = model_name
        print("teacher_model_name: ",model_name)
        teacher_net_class = eval(self.model_name)
        self.actor = teacher_net_class(obs_dim=num_actor_obs,
                                       act_dim=num_actions,
                                       activation=activation,
                                       output_activation=output_activation,
                                       **(NetModel[self.model_name]))
        
        critic_layers = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic = nn.Sequential(*critic_layers)
        self.critic_obs_dim = num_critic_obs

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, **kwargs):
        mean = self.actor(observations, **kwargs)
        std = torch.clamp(self.std, min=self.min_std, max=self.max_std)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, masks=None, **kwargs):
        if masks is not None:
            observations = unpad_trajectories(observations, masks).flatten(0, 1)
        self.update_distribution(observations, **kwargs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, masks=None, **kwargs):
        if masks is not None:
            observations = unpad_trajectories(observations, masks).flatten(0, 1)
        actions_mean = self.actor(observations, **kwargs)
        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, **kwargs):
        if masks is not None:
            critic_observations = unpad_trajectories(critic_observations, masks).flatten(0, 1)
        if (hasattr(self.actor, "evaluate")):
            value = self.actor.evaluate(critic_observations, **kwargs)
        else:
            value = self.critic(critic_observations)
        return value
    
    def get_hidden_states(self):
        return self.memory.get_hidden_states(), None
    
    def memory_encoder(self, obs, masks=None, hidden_states=None):
        latent = self.memory(obs, masks, hidden_states)
        return latent.flatten(0, 1)

