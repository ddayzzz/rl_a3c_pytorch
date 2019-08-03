from __future__ import division
# from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environments.utils import make_environment
from torch.nn.utils import clip_grad_norm_
from utils import ensure_shared_grads
from models.unreal import UNREAL
from experience import Experience, ExperienceFrame
import numpy as np
import time
import torch.nn.functional as F
from torch.autograd import Variable


LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Trainer(object):

    def __init__(self, rank, args, shared_model, optimizer, lr):
        # CUDA 相关
        self.gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
        torch.manual_seed(args.seed + rank)
        if self.gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)


        self.replay_buffer = Experience(history_size=2000)
        self.cx = None  # todo: 仍然是 一次 step 就前向传播
        self.hx = None
        self.episodic_score = 0
        self.rank = rank
        self.args = args
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.local_t = 0
        # 初始化
        # 初始化环境
        print('Training Agent: {}'.format(self.rank))
        # todo: 需要给 gym 环境加上 pc 等

        # agent 代理对象
        model = UNREAL(in_channels=3, action_size=6, enable_pixel_control=True)

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                model = model.cuda()

        model.train()

        # 学习率
        self.initial_learning_rate = lr
        self.max_global_time_step = 10 * 10 ** 7
        # 记录时间
        # For log output
        self.prev_local_t = 0

        self.model = model
        self.env = None
        self.reset()  # cx hx


    def train(self, global_t, summary_writer=None):
        t = self.local_t
        if not self.replay_buffer.is_full():
            self.fill_experience()
            return 0  # time_step = 0
        # sync
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(self.shared_model.state_dict())
        else:
            self.model.load_state_dict(self.shared_model.state_dict())

        loss_a3c, episode_score = self.process_a3c()
        # 获取 hx, cx
        h0, c0 = self.hx.detach(), self.cx.detach()
        loss_pc = self.process_pc(h0=h0, c0=c0)

        h0, c0 = self.hx.detach(), self.cx.detach()
        loss_vr = self.process_vr(h0, c0)

        loss_rp = self.process_rp()


        loss = loss_a3c + loss_pc + loss_vr + loss_rp


        self.model.zero_grad()
        loss.backward()

        clip_grad_norm_(self.model.parameters(), 40.0)
        ensure_shared_grads(self.model, self.shared_model, gpu=self.gpu_id >= 0)

        self.adjust_learning_rate(optimizer=self.optimizer, global_time_step=global_t)
        self.optimizer.step()
        if summary_writer is not None:
            with torch.no_grad():
                losses = list(map(lambda x: float(x.detach().cpu().numpy()), [loss_a3c, loss_pc, loss_vr, loss_rp, loss]))
                tags = dict(zip(['a3c', 'pc', 'vr', 'rp', 'total_loss'], losses))
                summary_writer.add_scalars('losses', tags, global_step=global_t)
                # 分数
                if episode_score:
                    summary_writer.add_scalars('score', {'score': episode_score}, global_step=global_t)
        self._print_log(global_t)
        return self.local_t - t  # offset

    def init_env(self):
        self.env = make_environment(env_type=self.args.env_type, env_name=self.args.env_name, channel_first=True)
        self.env.reset()  # 之后, 每一局结束都会自动 reset

    def adjust_learning_rate(self, optimizer, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @property
    def action_size(self):
        return 6  # self.env.action_space.n

    def set_start_time(self, start_time):
        self.start_time = start_time

    def _print_log(self, global_t):
        if (self.rank == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    def fill_experience(self):
        prev_state = self.env.last_state
        last_action = self.env.last_action
        last_reward = self.env.last_reward
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                      self.action_size,
                                                                      last_reward)
        with torch.no_grad():
            state = torch.from_numpy(self.env.last_state['rgb']).unsqueeze(0)
            lar = torch.from_numpy(last_action_reward).unsqueeze(0)
            # whether to gpu
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    state = state.cuda()
                    lar = lar.cuda()

            _, logits, (self.hx, self.cx) = self.model(task_type='a3c', states=state,
                                                       hx=self.hx, cx=self.cx,
                                                       last_action_reward=lar)

            action_index = self.choose_action(pi_values=F.softmax(logits, 1).cpu().numpy()[0])

            obs, reward, terminal, _ = self.env.step(action_index)  # 存储为数组

            frame = ExperienceFrame(prev_state['rgb'], reward, action_index, terminal, obs['pixel_change'], last_action, last_reward)
            self.replay_buffer.add_frame(frame)

            if terminal:
                self.env.reset()
            else:
                # 更新 LSTM 状态
                self.hx = self.hx.detach()
                self.cx = self.cx.detach()
            if self.replay_buffer.is_full():
                self.env.reset()
                print("Replay buffer filled")

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def l2_loss(self, x, y):
        return F.mse_loss(x, y, reduction='sum') * 0.5

    def process_a3c(self):
        rewards = []
        log_probs = []  # 指定的行为 概率
        entropies = []
        values = []
        action_one_hot = []
        # adv = []  # GAE, 采用 advantage 函数
        terminal_end = False  # 结束采样的时候是否是终止状态
        episode_score = None  # 决定是否显示 episodic score
        for t in range(20):
            state = torch.from_numpy(self.env.last_state['rgb']).unsqueeze(dim=0)  # batch = 1

            last_action = self.env.last_action
            last_reward = self.env.last_reward
            last_action_reward = torch.from_numpy(
                ExperienceFrame.concat_action_and_reward(last_action, self.action_size, last_reward)).unsqueeze(0)
            # whether to gpu
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    state = state.cuda()
                    last_action_reward = last_action_reward.cuda()

            value, logits, (self.hx, self.cx) = self.model('a3c', state, hx=self.hx, cx=self.cx, last_action_reward=last_action_reward)
            prob = F.softmax(logits, dim=1)  # batch, 6.
            log_prob = torch.log(prob.clamp(1e-20, 1.0))  #　F.log_softmax(logits, dim=1).clamp(1e-20, 1.0)  # batch, 6. NaN
            #
            entropy = -(log_prob * prob).sum(1)
            # 采取行为
            with torch.no_grad():
                action_index = self.choose_action(pi_values=F.softmax(logits, 1).cpu().numpy()[0])

            prev_state = self.env.last_state['rgb']

            observation, reward, terminal, _ = self.env.step(action_index)

            # 显示信息
            if self.rank == 0 and self.local_t % 100 == 0:
                print("pi={}".format(prob.detach().cpu().numpy()))
                print(" V={}".format(value.detach().cpu().numpy()))


            self.local_t += 1
            # 添加到 replay buffer
            frame = ExperienceFrame(prev_state, reward, action_index, terminal, observation['pixel_change'], last_action, last_reward)

            # Store to experience
            self.replay_buffer.add_frame(frame)


            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            log_probs.append(log_prob)

            a = torch.zeros(self.action_size, dtype=torch.float32)
            # a = np.zeros([self.action_size], dtype=np.float32)
            a[action_index] = 1.0
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    a = a.cuda()
            action_one_hot.append(a)


            self.episodic_score += reward
            if terminal:
                print('Score: {0}'.format(self.episodic_score))
                episode_score = self.episodic_score
                self.episodic_score = 0
                terminal_end = True
                self.env.reset()
                self.reset()
                break
            else:
                self.hx = self.hx.detach()
                self.cx = self.cx.detach()
        # 计算 R
        R = torch.zeros(1, 1)
        if not terminal_end:
            with torch.no_grad():
                # 这里进行 bootstrapping
                state = torch.from_numpy(observation['rgb']).unsqueeze(0)
                lar = torch.from_numpy(frame.get_action_reward(self.action_size)).unsqueeze(0)

                # whether to gpu
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        state = state.cuda()
                        lar = lar.cuda()

                value, _, (_, _) = self.model(task_type='a3c', states=state, hx=self.hx, cx=self.cx,
                                              last_action_reward=lar)
                R = value.detach()  # 这个值为 V(s_t,\theta_v^'), 在计算 actor 的梯度的时候可能会算入, 所以 detach
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                R = R.cuda()
        # values.append(R)  # 用于 bootstrapping
        # 准备计算 loss
        policy_loss = 0
        value_loss = 0
        # 反向计算 loss
        for i in reversed(range(len(rewards))):  # i=t - 1,...,t_start
            R = self.args.gamma * R + rewards[i]  # R <- r_t + \gamma * R
            adv = R - values[i]  # GAE = advantage, R - V(s_i;\theta^'_v), adv 在反向传播的时候是 \theta_v的梯度, 在 policy 的梯度需要 detach
            value_loss += 0.5 * self.l2_loss(R, values[i]) # 定义为 0.5 * MSELOSS(R - value), 学习率是 actor 一半
            log_prob_a = (log_probs[i] * action_one_hot[i]).sum(1)  # log(a_i|s_i;\theta^')
            policy_loss += -log_prob_a * adv.detach() + entropies[i] * 0.001  # entropy_beta
        return value_loss + policy_loss, episode_score

    def process_pc(self, h0, c0):
        """

        :param h0: PC 起始的 LSTM 的 hidden 状态
        :param c0:
        :return:
        """
        # [pixel change]
        # Sample 20+1 frame (+1 for last next state)
        pc_experience_frames = self.replay_buffer.sample_sequence(20 + 1)
        # Reverse sequence to calculate from the last
        pc_experience_frames.reverse()

        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []
        batch_pc_q = []
        batch_pc_last_action_reward = []

        pc_R = torch.zeros(20, 20)
        if not pc_experience_frames[1].terminal:
            with torch.no_grad():
                state = torch.from_numpy(pc_experience_frames[0].state).unsqueeze(dim=0)  # batch = 1
                last_action_reward = torch.from_numpy(
                    pc_experience_frames[0].get_last_action_reward(self.action_size)).unsqueeze(0)

                # whether to gpu
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        state = state.cuda()
                        last_action_reward = last_action_reward.cuda()

                # 每次forward传播的时候需要使用全零的 hx, cx
                (hx, cx) = self.get_zero_hx_cx()
                _, pc_q_max, (_, _) = self.model('pc', state, hx=hx, cx=cx,
                                         last_action_reward=last_action_reward)  # 不更新 lstm 的状态

                pc_R = pc_q_max.detach()  # 共享 LSTM 的参数

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                pc_R.cuda()

        for frame in pc_experience_frames[1:]:
            pc = torch.from_numpy(frame.pixel_change)
            a = torch.zeros(self.action_size, dtype=torch.float32)
            # a = np.zeros([self.action_size], dtype=np.float32)
            a[frame.action] = 1.0
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    a = a.cuda()
                    pc = pc.cuda()
            pc_R = pc + 0.9 * pc_R  # gamma_pc = 0.9

            last_action_reward = frame.get_last_action_reward(self.action_size)

            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)
            batch_pc_last_action_reward.append(last_action_reward)

        batch_pc_si.reverse()
        batch_pc_a.reverse()
        batch_pc_R.reverse()
        batch_pc_last_action_reward.reverse()

        # 计算输出的
        for (state, last_action_reward) in zip(batch_pc_si, batch_pc_last_action_reward):
            # 获取输出的 pc_q, 需要在网络中计算一遍
            state = torch.from_numpy(state).unsqueeze(dim=0)  # batch = 1
            last_action_reward = torch.from_numpy(last_action_reward).unsqueeze(0)
            # whether to gpu
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    state = state.cuda()
                    last_action_reward = last_action_reward.cuda()
            # 在反向传播的时候, LSTM 的Cell和 Hidden 与 A3C 最后一次的输出有关, 在 tf 的版本中, feed_dict设置的 初始LSTM 状态就是 A3C 运行之前的状态
            pc_q, _, (h0, c0) = self.model('pc', state, hx=h0, cx=c0,  # 使用 A3C 初始的状态
                                           last_action_reward=last_action_reward)  # 不更新 lstm 的状态
            # 修改 LSTM cell 和 hidden 的状态
            h0 = h0.detach()
            c0 = c0.detach()

            batch_pc_q.append(pc_q)
        # 损失, 使用 n-step Qlearning
        batch_pc_R = torch.cat(batch_pc_R, dim=0)
        batch_pc_q = torch.cat(batch_pc_q, dim=0)  # 按照第一个维度连接
        pc_a_reshape = torch.stack(batch_pc_a).view(-1, self.action_size, 1, 1)
        pc_qa_ = torch.mul(batch_pc_q, pc_a_reshape)
        pc_qa = pc_qa_.sum(dim=1, keepdim=False)  # -1, 20, 20
        pc_loss = 0.05 * self.l2_loss(pc_qa, batch_pc_R)  # pixel_change_lambda = 0.05
        return pc_loss

    def process_vr(self, h0, c0):
        # [Value replay]
        # Sample 20+1 frame (+1 for last next state)
        vr_experience_frames = self.replay_buffer.sample_sequence(20 + 1)
        # Reverse sequence to calculate from the last
        vr_experience_frames.reverse()

        batch_vr_si = []
        batch_vr_R = []
        batch_vr_last_action_reward = []
        batch_values = []

        vr_R = torch.zeros(1, 1)
        if not vr_experience_frames[1].terminal:
            with torch.no_grad():
                state = torch.from_numpy(vr_experience_frames[0].state).unsqueeze(dim=0)  # batch = 1
                last_action_reward = torch.from_numpy(
                    vr_experience_frames[0].get_last_action_reward(self.action_size)).unsqueeze(0)
                # whether to gpu
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        state = state.cuda()
                        last_action_reward = last_action_reward.cuda()
                #
                (hx, cx) = self.get_zero_hx_cx()
                vr_R, (_, _) = self.model('vr', state, hx=hx, cx=cx,
                                        last_action_reward=last_action_reward)  # 不更新 lstm 的状态
                vr_R = vr_R.detach()  # value 的参数, \theta^'_v

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                vr_R.cuda()

        # t_max times loop
        for frame in vr_experience_frames[1:]:
            vr_R = frame.reward + self.args.gamma * vr_R
            batch_vr_si.append(frame.state)
            batch_vr_R.append(vr_R)
            last_action_reward = frame.get_last_action_reward(self.action_size)
            batch_vr_last_action_reward.append(last_action_reward)

        batch_vr_si.reverse()
        batch_vr_R.reverse()
        batch_vr_last_action_reward.reverse()

        for (state, last_action_reward) in zip(batch_vr_si, batch_vr_last_action_reward):
            # 获取输出的 pc_q, 需要在网络中计算一遍
            state = torch.from_numpy(state).unsqueeze(dim=0)  # batch = 1
            last_action_reward = torch.from_numpy(last_action_reward).unsqueeze(0)
            # whether to gpu
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    state = state.cuda()
                    last_action_reward = last_action_reward.cuda()
            value, (h0, c0) = self.model('vr', state, hx=h0, cx=c0, last_action_reward=last_action_reward)  # 不更新 lstm 的状态
            h0 = h0.detach()
            c0 = c0.detach()
            batch_values.append(value)

        batch_vr_R = torch.cat(batch_vr_R, dim=0)
        batch_values = torch.cat(batch_values, dim=0)
        vr_loss = self.l2_loss(batch_vr_R, batch_values)
        return vr_loss

    def process_rp(self):
        # [Reward prediction]
        rp_experience_frames = self.replay_buffer.sample_rp_sequence()
        # 4 frames
        batch_rp_si = []
        # batch_rp_c = []

        for i in range(3):
            batch_rp_si.append(rp_experience_frames[i].state)
            # 求输出的 标签 0 + -
        states = torch.from_numpy(np.stack(batch_rp_si))  # batch = 3
        # whether to gpu
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                states = states.cuda()
        logits = self.model('rp', states, hx=None, cx=None, last_action_reward=None)  # 不更新 lstm 的状态, conv的参数

        # one hot vector for target reward
        r = rp_experience_frames[3].reward

        rp_c_target = torch.zeros(3, dtype=torch.float32)

        if r == 0:
            rp_c_target[0] = 1.0  # zero
        elif r > 0:
            rp_c_target[1] = 1.0  # positive
        else:
            rp_c_target[2] = 1.0  # negative

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                rp_c_target = rp_c_target.cuda()

        rp_c = F.softmax(logits, 1).clamp(1e-20, 1.0)
        rp_loss = -(rp_c_target * torch.log(rp_c)).sum()

        return rp_loss

    def reset(self):
        self.cx, self.hx = self.get_zero_hx_cx()

    def get_zero_hx_cx(self, batch_size=1):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                cx = Variable(torch.zeros(1, batch_size, 256).cuda())
                hx = Variable(torch.zeros(1, batch_size, 256).cuda())
        else:
            cx = Variable(torch.zeros(1, batch_size, 256))  # todo: 仍然是 一次 step 就前向传播
            hx = Variable(torch.zeros(1, batch_size, 256))
        return (cx, hx)

    def close(self):

        self.env.close()







