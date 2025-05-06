from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers import DDIMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

from torch.distributions.beta import Beta
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

class TimePredictionModele(nn.Module):
    def __init__(self,
                 input_dim=1280,
                 time_embed_dim=256,
                 hidden_dim=512,
                 output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim,hidden_dim,kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim,hidden_dim//2,kernel_size=1)
        self.time_step = nn.Sequential(
            nn.Linear(time_embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim//2,output_dim)

    def forward(self,features,time_embed):
        #features:(B,1280,T),time_embed:(B,256)
        x=F.relu(self.conv1(features)) #(B,512,T)
        #融入时间步嵌入
        time_scale = self.time_step(time_embed).unsqueeze(-1) #(B,512,1)
        x=x*time_scale #广播到(B,512,T)
        x=F.relu(self.conv2(x)) #(B,256,T)
        x=torch.mean(x,dim=2) #(B,256)
        ab=self.fc(x) #(B,2)
        #计算beta分布
        a,b =ab[:,0],ab[:,1]
        alpha = 1+torch.exp(a)
        beta = 1+torch.exp(b)
        r_n = Beta(alpha,beta).sample() #(B,)
        return r_n,(alpha,beta)

class SimpleDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDIMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None, #10 #原始去噪推理步数
            max_inference_steps=10, #tpm最大推理步数
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            tpm_hidden_dim=512,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")


        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        #初始化tpm
        self.tpm = TimePredictionModele(
            input_dim=down_dims[0]+down_dims[-1],
            time_embed_dim=diffusion_step_embed_dim,
            hidden_dim=tpm_hidden_dim,
            output_dim=2
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.max_inference_steps = max_inference_steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        tpm=self.tpm

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        #使用tpm动态调整时间步
        t_current = torch.ones(trajectory.shape[0],device=trajectory.device) #t_0=1.0
        t_current = t_current*scheduler.config.num_train_timesteps-1

        t_min = 0.01 #terminal
        t_min = torch.tensor(t_min,dtype=torch.float32,device=t_current.device)
        t_min = t_min.unsqueeze(0)

        step = 0 
        while(t_current > t_min).any() and step < self.max_inference_steps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            with torch.no_grad():
                #获取时间步嵌入
                time_embed = model.time_mlp(model.timestep_embedding(t_current)) #(B,256)

                #获取中间特征
                features_before,features_after = model.get_intermediate_features(
                    sample=trajectory, 
                    timestep=t_current, 
                    local_cond=local_cond, 
                    global_cond=global_cond, 
                )
            
            #拼接特征
            features = torch.cat([features_before,features_after],dim=1)

            #tpm预测衰减率
            r_n,(alpha,beta)=tpm(features,time_embed)

            #更新时间步
            t_next=r_n*t_current
            #t_next=torch.clamp(t_next,min=t_min,max=torch.tensor(1.0,device=t_next.device))
            
            # print("step",step)
            # print("t_current:",t_current)
            # print("r_n:",r_n)
            # print("t_next:",t_next)

            #使用ddim计算前一步
            model_output = model(sample=trajectory,
                                timestep=t_current, 
                                local_cond=local_cond, 
                                global_cond=global_cond)

            #自定义实现ddim
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)
            alpha_t=scheduler.alphas_cumprod[t_current.long()]
            alpha_t_prev=scheduler.alphas_cumprod[t_next.long()]
            alpha_t_prev=alpha_t_prev.view(-1,1,1)

            trajectory = torch.sqrt(alpha_t_prev)*model_output+torch.sqrt(1-alpha_t_prev)*model_output
            t_current=t_next
            step+=1
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict

    #=====tpm-ppo实现====#
    def compute_tpm_reward(self,trajectory,t_current,r_n,ground_truth,t_min):
        """
        args:
            trajectory: (B,T,Da), 当前去噪动作
            t_current: (B,)
            r_n: (B,)
            ground_truth: (B,T,Da), 真实动作（专家演示）
            t_min: float
        returns:
            reward: float
        """
        #动作质量奖励
        mse = F.mse_loss(trajectory,ground_truth,reduction='mean')
        r_quality = -mse.item()
        #效率奖励
        r_efficiency = torch.log(r_n+1e-8).mean().item()
        #终止奖励
        r_terminal = 1.0 if (t_current < t_min).all() else 0.0
        #综合奖励
        w1,w2,w3 = 1.0,0.1,1.0
        reward = w1*r_quality + w2*r_efficiency +w3*r_terminal
        return reward
    
    def train_tpm_with_ppo(self,
                           dataset,
                           num_epochs=200,
                           batch_size=256,
                           learning_rate=1e-5,
                           gamma=0.99,
                           clip_eps=0.2,
                           value_loss_coef=0.5,
                           entropy_coef=0.01):
        """
        使用ppo训练tpm
        args:
            dataset: 包含专家演示的数据集
            num_epochs: int, 训练轮数
            batch_size: int, 批次大小
            learning_rate: float, 学习率
            gamma: float, 折扣因子
            clip_eps: float, ppo剪切范围
            value_loss_coef: float, 价值损失权重
            entrepy_coef: float, 熵正则化权重
        """
        #冻结ConditionalUnet1D和obs_encoder
        self.model.eval()
        self.obs_encoder.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.obs_encoder.parameters():
            param.requires_grad = False

        #训练 tpm
        self.tpm.train()
        optimizer = optim.Adam(self.tpm.parameters(),lr=learning_rate)

        #critic
        critic = nn.Sequential(
            nn.Conv1d(1280,512,kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512,256,kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256,1)
        ).to(self.device)

        critic_optimizer=optim.Adam(critic.parameters(),lr=learning_rate)
        #critic_optimizer=optim.AdamW(critic.parameters(),lr=learning_rate,betas=[0.9,0.99])

        ##如果dataset是dataloader，直接使用；否则列表切片
        if isinstance(dataset,DataLoader):
            data_iter = iter(dataset)
        else:
            data_iter = None

        #ppo训练循环
        for epoch in range(num_epochs):
            if isinstance(dataset,DataLoader):
                #使用dataloader迭代
                for batch in dataset:
                    #收集轨迹
                    #每个批次重新初始化列表
                    states=[]
                    actions=[]
                    log_probs=[]
                    rewards=[]
                    values=[]
                    dones=[]

                    batch = dict_apply(batch,lambda x: x.to(self.device,non_blocking=True))
                    nobs = self.normalizer.normalize(batch['obs']) #归一化观测
                    nactions = self.normalizer['action'].normalize(batch['action']) #归一化真实动作
                    
                    if not self.use_pc_color:
                        nobs['point_cloud'] = nobs['point_cloud'][...,:3]

                    print("nactions",nactions.shape)
                    batch_size = nactions.shape[0]
                    horizon = nactions.shape[1]

                    local_cond = None
                    global_cond = None
                    trajectory = nactions
                    cond_data = trajectory

                        #=== same with predict_action ===#
                    if self.obs_as_global_cond:
                        this_nobs = dict_apply(nobs,
                        lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                        nobs_features = self.obs_encoder(this_nobs)
                        if "cross_attention" in self.condition_type:
                            global_cond = nobs_features.reshape(batch_size,self.n_obs_steps,-1)
                        else:
                            global_cond = nobs_features.reshape(batch_size,-1)
                    #=== same with compute_loss ===#

                    cond_data = torch.zeros(size=(batch_size,self.horizon,self.action_dim),device=self.device,dtype=nactions.dtype)
                    cond_mask = torch.zeros_like(cond_data,dtype=torch.bool)

                    #去噪过程   #====be similar to conditional_sample===#
                    trajectory = torch.randn_like(cond_data)
                    t_current = torch.ones((batch_size,),device=self.device)
                    t_min = 0.01
                    step=0

                    while(t_current > t_min).any() and step < self.max_inference_steps:
                        trajectory[cond_mask] = cond_data[cond_mask]

                        with torch.no_grad():
                            time_embed = self.model.time_mlp(self.model.timestep_embedding(t_current)) #(B,256)
                            features_before,features_after = self.model.get_intermediate_features(
                                sample=trajectory, 
                                timestep=t_current, 
                                local_cond=local_cond, 
                                global_cond=global_cond, 
                            )
                        features = torch.cat([features_before,features_after],dim=1)

                        #tpm预测r_n
                        r_n,(alpha,beta)=self.tpm(features,time_embed)

                        ##计算动作的对数概率
                        beta_dist=Beta(alpha,beta)
                        log_prob=beta_dist.log_prob(r_n).sum()

                        ##状态=特征+时间嵌入
                        #print("features",features.shape) #(B,1280,4)
                        #print("time_embed",time_embed.shape) #(B,128)
                        state=torch.cat([features,time_embed.unsqueeze(-1).expand(-1,-1,features.shape[-1])],dim=1)

                        ##价值估计
                        with torch.no_grad():
                            value=critic(features)

                        ##存储
                        states.append(state)
                        actions.append(r_n)
                        log_probs.append(log_prob)
                        values.append(value)

                        #更新时间步
                        t_next=r_n*t_current
                        
                        #使用ddim计算前一步
                        model_output = self.model(sample=trajectory,
                                                timestep=t_current, 
                                                local_cond=local_cond,
                                                global_cond=global_cond)                                     

                        #自定义实现ddim
                        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)
                        alpha_t=self.noise_scheduler.alphas_cumprod[t_current.long()]
                        alpha_t_prev=self.noise_scheduler.alphas_cumprod[t_next.long()]
                        alpha_t_prev=alpha_t_prev.view(-1,1,1)

                        trajectory = torch.sqrt(alpha_t_prev)*model_output+torch.sqrt(1-alpha_t_prev)*model_output
                        
                        ##计算奖励
                        reward=self.compute_tpm_reward(trajectory,t_current,r_n,nactions,t_min)
                        rewards.append(reward)

                        ##终止标志
                        done=(t_current<t_min).all() or (step>=self.num_inference_steps-1)
                        dones.append(done)
                        
                        t_current=t_next
                        step+=1

            else:
                #按批次切片
                for batch_idx in range(0,len(dataset),batch_size):
                    #=== same with compute_loss ===#
                    batch = dataset[batch_idx:batch_idx+batch_size]
                    nobs = self.normalizer.normalize(batch['obs']) #归一化观测
                    nactions = self.normalizer['action'].normalize(batch['action']) #归一化真实动作
                    
                    if not self.use_pc_color:
                        nobs['point_cloud'] = nobs['point_cloud'][...,:3]

                    batch_size = nactions.shape[0]
                    horizon = nactions.shape[1]

                    local_cond = None
                    global_cond = None
                    trajectory = nactions
                    cond_data = trajectory

                        #=== same with predict_action ===#
                    if self.obs_as_global_cond:
                        this_nobs = dict_apply(nobs,
                        lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                        nobs_features = self.obs_encoder(this_nobs)
                        if "cross_attention" in self.condition_type:
                            global_cond = nobs_features.reshape(batch_size,self.n_obs_steps,-1)
                        else:
                            global_cond = nobs_features.reshape(batch_size,-1)
                    #=== same with compute_loss ===#

                    cond_data = torch.zeros(size=(batch_size,self.horizon,self.action_dim),device=self.device,dtype=nactions.dtype)
                    cond_mask = torch.zeros_like(cond_data,dtype=torch.bool)

                    #去噪过程   #====be similar to conditional_sample===#
                    trajectory = torch.randn_like(cond_data)
                    t_current = torch.ones((batch_size,),device=self.device)
                    t_min = 0.01
                    step=0

                    while(t_current > t_min).any() and step < self.max_inference_steps:
                        trajectory[cond_mask] = cond_data[cond_mask]

                        with torch.no_grad():
                            time_embed = self.model.time_mlp(self.model.timestep_embedding(t_current)) #(B,256)
                            features_before,features_after = self.model.get_intermediate_features(
                                sample=trajectory, 
                                timestep=t_current, 
                                local_cond=local_cond, 
                                global_cond=global_cond, 
                            )
                        features = torch.cat([features_before,features_after],dim=1)

                        #tpm预测r_n
                        r_n,(alpha,beta)=self.tpm(features,time_embed)

                        ##计算动作的对数概率
                        beta_dist=Beta(alpha,beta)
                        log_prob=beta_dist.log_prob(r_n).sum()

                        ##状态=特征+时间嵌入
                        state=torch.cat([features,time_embed.unsqueeze(-1)],dim=1)

                        ##价值估计
                        with torch.no_grad():
                            value=critic(features)

                        ##存储
                        states.append(state)
                        actions.append(r_n)
                        log_probs.append(log_prob)
                        values.append(value)

                        #更新时间步
                        t_next=r_n*t_current
                        
                        #使用ddim计算前一步
                        model_output = self.model(sample=trajectory,
                                                timestep=t_current, 
                                                local_cond=local_cond,
                                                global_cond=global_cond)                                     

                        #自定义实现ddim
                        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)
                        alpha_t=self.noise_scheduler.alphas_cumprod[t_current.long()]
                        alpha_t_prev=self.noise_scheduler.alphas_cumprod[t_next.long()]

                        trajectory = torch.sqrt(alpha_t_prev)*model_output+torch.sqrt(1-alpha_t_prev)*model_output
                        
                        ##计算奖励
                        reward=self.compute_tpm_reward(trajectory,t_current,r_n,nactions,t_min)
                        rewards.append(reward)

                        ##终止标志
                        done=(t_current<t_min).all() or (step>=self.num_inference_steps-1)
                        dones.append(done)
                        
                        t_current=t_next
                        step+=1
            
            ##计算回报和优势
            returns =[]
            advantages = []
            G=0
            for r,d in zip(reversed(rewards),reversed(dones)):
                if d:
                    G=0
                G=r+gamma*G
                returns.insert(0,G)

            returns=torch.tensor(returns,device=self.device)
            values=torch.cat(values).squeeze()  #(num_steps*batch_size)

            returns=returns.unsqueeze(1).expand(-1,batch_size).reshape(-1)
            #print("returns shape",returns.shape) #79 #9 #36
            #print("values shape",values.shape) #8996 #36 
            
            advantages=returns-values            

            ##标准化优势
            advantages=(advantages-advantages.mean()) / (advantages.std()+1e-8)

            ##ppo更新
            states=torch.stack(states)
            actions=torch.stack(actions)
            old_log_probs=torch.stack(log_probs)

            for _ in range(10): #ppo更新多次
                ##重新计算策略和价值
                features=states[:,:-256,:] #去掉time_embed部分
                time_embed=states[:,-256:,:] #time_embed部分 time_embed:(B,256)
                time_embed=time_embed.squeeze(-1)
                r_n,(alpha,beta)=self.tpm(features,time_embed)
                beta_dist=Beta(alpha,beta)
                new_log_probs=beta_dist.log_prob(r_n).sum()

                value_pred=critic(features).squeeze()

                ##计算ppo损失
                ratio = torch.exp(new_log_probs-old_log_probs)
                surr1=ratio*advantages
                surr2=torch.clamp(ratio,1-clip_eps,1+clip_eps)*advantages
                policy_loss=-torch.min(surr1,surr2).mean()

                value_loss=F.mse_loss(value_pred,returns)

                entropy=beta_dist.entropy().mean()
                loss=policy_loss + value_loss_coef*value_loss - entropy_coef*entropy

                ##更新参数
                optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                critic_optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
