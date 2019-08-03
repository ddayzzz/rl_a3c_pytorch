from __future__ import division
from experience import Experience, ExperienceFrame
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.autograd import Variable


class Agent(object):

    def __init__(self, model, env, action_size, args, state):
        self.model = model
        self.env = env
        self.action_size = action_size
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True  # 初始化，可以设置一次新的状态值
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        # 参数

        self.memory = Experience(history_size=2000)

    def fill_experience(self):
        prev_state = self.env.last_state
        last_action = self.env.last_action
        last_reward = self.env.last_reward
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                      self.action_size,
                                                                      last_reward)
        with torch.no_grad():
            state = torch.from_numpy(self.env.last_state).unsqueeze(0)
            lar = torch.from_numpy(last_action_reward).unsqueeze(0)
            _, pi, (self.hx, self.cx) = self.model(task_type='a3c', states=state,
                                                   hx=self.hx, cx=self.cx,
                                                   last_action_rewards=lar)

            action_index = pi.max(1)[1].view(1, 1).item()

        new_state, reward, terminal, pixel_change = self.env.step(action_index)  # 存储为数组

        frame = ExperienceFrame(prev_state, reward, action_index, terminal, pixel_change, last_action, last_reward)
        self.memory.add_frame(frame)

        if terminal:
            self.env.reset()
        if self.memory.is_full():
            self.env.reset()
            print("Replay buffer filled")
        self.done = terminal

    def a3c_process(self):
        """
        在 on-policy 下运行程序
        :return:
        """
        states = []
        last_action_rewards = []
        actions = []  #
        rewards = []
        values = []  # V
        actions_prob = []

        terminal_end = False

        # t_max times loop
        for _ in range(self.args.num_steps):
            # Prepare last action reward
            last_action = self.env.last_action
            last_reward = self.env.last_reward
            last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                          self.action_size,
                                                                          last_reward)
            state = torch.from_numpy(self.env.last_state).unsqueeze(0)
            lar = torch.from_numpy(last_action_reward)

            v, pi, (self.hx, self.cx) = self.model(task_type='a3c', states=state,
                                                      hx=self.hx, cx=self.cx,
                                                      last_action_rewards=lar.unsqueeze(0))

            action_index = pi.max(1)[1].view(1, 1).item()

            states.append(torch.from_numpy(self.env.last_state))
            actions_prob.append(torch.squeeze(pi, dim=0))
            last_action_rewards.append(lar)
            actions.append(action_index)
            values.append(v)


            prev_state = self.env.last_state


            new_state, reward, terminal, pixel_change = self.env.step(action_index)
            frame = ExperienceFrame(prev_state, reward, action_index, terminal, pixel_change, last_action, last_reward)

            # Store to experience
            self.memory.add_frame(frame)

            # self.episode_reward += reward

            rewards.append(reward)

            self.update_lstm_state()
            if terminal:
                self.env.reset()
                break


        R = torch.zeros(1, 1)
        if not terminal_end:
            state = torch.from_numpy(new_state).unsqueeze(0)
            lar = torch.from_numpy(frame.get_action_reward(self.action_size)).unsqueeze(0)
            value, _, _ = self.model(task_type='a3c', states=state, hx=self.hx, cx=self.cx, last_action_rewards=lar)
            R = value.data
        # 构造误差项
        actions.reverse()
        rewards.reverse()
        values.reverse()

        batch_a = []
        batch_adv = []
        batch_R = []

        for (ai, ri, Vi) in zip(actions, rewards, values):
            R = ri + self.args.gamma * R
            adv = R - Vi
            a = np.zeros([self.action_size], dtype=np.float32)
            a[ai] = 1.0

            batch_a.append(torch.from_numpy(a))
            batch_adv.append(adv)
            batch_R.append(R)

        batch_a.reverse()
        batch_adv.reverse()
        batch_R.reverse()
        # 转换为张量

        return batch_a, batch_adv, batch_R, last_action_rewards, states, actions_prob, values

    def a3c_loss(self, batch_a, batch_adv, batch_R, last_action_rewards, states, actions_prob, values):
        batch_a = torch.stack(batch_a)  # batch, 6
        batch_adv = torch.stack(batch_adv)  # batch,1,1
        last_action_rewards = torch.stack(last_action_rewards)  # batch,7
        batch_R = torch.stack(batch_R)  # batch,1,1
        states = torch.stack(states)  # batch,3,84,84
        actions_prob = torch.stack(actions_prob)  # batch,6
        values = torch.stack(values)
        # 损失函数
        log_pi = torch.log(torch.clamp(actions_prob, min=1e-20, max=1.0))
        entropy = -torch.sum(log_pi * actions_prob, dim=1)
        # 对应的 a_i 的概率
        log_pi_a_i = torch.sum(torch.mul(log_pi, batch_a), dim=1)
        policy_loss = torch.sum(log_pi_a_i * batch_adv + entropy * 0.001)
        # value_loss
        value_loss = 0.5 * F.mse_loss(batch_R, values)
        return policy_loss + value_loss

    def action_train(self):
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            self.update_lstm_state()
            state = torch.from_numpy(self.env.last_state).unsqueeze(0)

            last_action = self.env.last_action
            last_reward = np.clip(self.env.last_reward, -1, 1)
            last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.action_size,
                                                                          last_reward)
            lar = torch.from_numpy(last_action_reward)

            v, pi, (self.hx, self.cx) = self.model(task_type='a3c', states=state,
                                                   hx=self.hx, cx=self.cx,
                                                   last_action_rewards=lar.unsqueeze(0))
        prob = F.softmax(pi, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, pixel_change = self.env.step(action[0])
        self.info = 5
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def update_lstm_state(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 256).cuda())
                    self.hx = Variable(torch.zeros(1, 256).cuda())
            else:
                self.cx = Variable(torch.zeros(1, 256))
                self.hx = Variable(torch.zeros(1, 256))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
