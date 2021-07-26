import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import os
import copy
from tqdm import tqdm
import numpy as np

from models.dve import DVE
from models.predictor import Predictor
from models.encoder import Encoder
from utils import one_hot


class DVRL(object):
    def __init__(self, train_dataset, valid_dataset, dict_dvrl={}):
        # data
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        assert len(dict_dvrl) > 0

        # params
        self.dve_lr = dict_dvrl['dve_lr']
        self.pred_lr = dict_dvrl['pred_lr']

        self.feat_dim = dict_dvrl['feat_dim']
        self.label_dim = dict_dvrl['category_num']

        self.epsilon = dict_dvrl['epsilon']
        self.threshold = dict_dvrl['threshold']

        self.outer_iterations = dict_dvrl['outer_iterations']
        self.inner_iterations = dict_dvrl['inner_iterations']

        self.batch_size = dict_dvrl['batch_size']
        self.mini_batch_size = dict_dvrl['batch_size_predictor']

        # models
        self.dve_model = DVE(self.feat_dim, self.label_dim)
        self.pred_model = Predictor(self.feat_dim, self.label_dim)
        self.encoder = Encoder()

        # optimizers
        self.dve_optimizer = optim.Adam(self.dve_model.parameters(), self.dve_lr)
        self.pred_optimizer = optim.Adam(self.pred_model.parameters(), self.pred_lr)

        # loss function
        self.pred_criterion = nn.CrossEntropyLoss()

        # checkpoint dir
        self.dve_checkpoint_dir = os.path.join(dict_dvrl['checkpoint_dir'], 'dve')
        self.pred_checkpoint_dir = os.path.join(dict_dvrl['checkpoint_dir'], 'predictor')
        self.assert_dir(self.dve_checkpoint_dir)
        self.assert_dir(self.pred_checkpoint_dir)

        # auxiliary checkpoint of predictor model
        self.init_checkpoint = os.path.join(self.pred_checkpoint_dir, 'ori_model.pth')
        self.ori_baseline_checkpoint = os.path.join(self.pred_checkpoint_dir, 'ori_model.pth')
        self.val_baseline_checkpoint = os.path.join(self.pred_checkpoint_dir, 'val_model.pth')

        # checkpoint of dve model
        self.dve_checkpoint = os.path.join(self.dve_checkpoint_dir, 'dve_model.pth')

        # init predictor model
        self.build_init_pred()
        # baseline predictor model
        self.ori_model = self.build_baseline(train_dataset, self.ori_baseline_checkpoint)
        self.val_model = self.build_baseline(valid_dataset, self.val_baseline_checkpoint)
    
    def assert_dir(self, file_dir):
        """
        确保指定的文件夹存在
        """
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    
    def build_init_pred(self):
        """
        保存predictor模型的初始参数，确保后面训练的predictor模型的初始条件相同
        """
        torch.save(self.pred_model.state_dict(), self.init_checkpoint)
        
    def build_baseline(self, dataset, baseline_checkpoint):
        """
        训练baseline predictor模型;
        在dvrl中需要训练两个baseline，一个是在训练集上训练的predictor模型，一个是在验证上训练的predictor模型;
        dataset: 训练集/验证集
        baseline_checkpoint: baseline模型保存的路径
        """
        model = copy.copy(self.pred_model)
        if os.path.exists(baseline_checkpoint):
            data = torch.load(baseline_checkpoint)
            model.load_state_dict(data)
        else:
            # init baseline model
            init_data = torch.load(self.init_checkpoint)
            model.load_state_dict(init_data)
            model.train()
            # train baseline model
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            pred_optimizer = optim.Adam(model.parameters(), self.pred_lr)
            criterion = nn.CrossEntropyLoss()
            for x_batch, y_batch in dataloader:
                """
                x_batch: (batch, 3, 32, 32)
                y_batch: (batch,)
                """
                y_pred_batch = model(x_batch)  # (batch, category_num)
        
                pred_optimizer.zero_grad()
                loss = criterion(y_pred_batch, y_batch)
                loss.backward()
                pred_optimizer.step()
            # save baseline model
            torch.save(model.state_dict(), baseline_checkpoint)

        return model.eval()
    
    def get_performance(self, model, dataset):
        """
        获取模型在指定数据集上的性能；
        此处的模型是在训练集上训练的predictor模型，此处的数据集是验证集；此处的性能是预测准确率；
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        model.eval()
        total_num, correct_num = 0, 0
        for x_batch, y_batch in dataloader:
            """
            x_batch: (batch, 3, 32, 32)
            y_batch: (batch,)
            """
            total_num += y_batch.shape[0]
            with torch.no_grad():
                y_pred_batch = model(x_batch)   # (batch, category_num)
            correct_num += y_pred_batch.eq(y_batch).cpu().float().sum().item()
        
        return correct_num / total_num

    def get_differences(self, model, dataset):
        """
        获取指定数据集上，模型的预测值与GT的差异；
        此处的模型是在验证集上训练的predictor模型，此处的dataset是训练集；
        """
        y_pred_diff = []
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        model.eval()
        for x_batch, y_batch in dataloader:
            with torch.no_grad():
                y_pred = model(x_batch)
            target = one_hot(y_batch)
            batch_pred_diff = torch.abs(target - y_pred)
            for i in batch_pred_diff.shape[0]:
                y_pred_diff.append(batch_pred_diff[i])
        return torch.tensor(y_pred_diff)
        
    def get_random_batch(self, dataset, y_pred_diff):
        """
        从训练集中获取batch_size个样本
        dataset: torch.utils.data.Dataset，数据集
        y_pred_diff: val模型在dataset上的预测值与gt label的差异值
        """
        assert len(dataset) == y_pred_diff.shape[0]
        x_batch, y_batch, y_hat_batch = [], [], []
        batch_idxs = np.random.permutation(len(dataset))[:self.batch_size]
        for idx in batch_idxs:
            x, y = dataset[idx]
            x_batch.append(x)
            y_batch.append(y)
            y_hat_batch.append(y_pred_diff[idx])
        return torch.tensor(x_batch), torch.tensor(y_batch), torch.tensor(y_hat_batch)
    
    # 需要check一下不使用某个样本时的优化情况
    def train_predictor(self, x_batch, y_batch, sample_weight):
        """
        使用batch_size个样本训练预测器模型，inner_iterations个epoch
        x_batch: (batch_size, 3, 32, 32)
        y_batch: (batch_size, )
        sample_weight: (batch_size, )
        """
        # init predictor model
        init_data = torch.load(self.init_checkpoint)
        self.pred_model.load_state_dict(init_data)
        # train predictor model
        for _ in range(self.inner_iterations):
            batch_idx_predictor = np.random.permutation(x_batch.shape[0])[:self.mini_batch_size]
            x_mini_batch, y_mini_batch, weight_mini_batch = x_batch[batch_idx_predictor], y_batch[batch_idx_predictor], sample_weight[batch_idx_predictor]
            """
            x_mini_batch: (mini_batch_size, feat_dim)
            y_mini_batch: (mini_batch_size, )
            weight_mini_batch: (mini_batch_size, )
            """
            x_masked_mini_batch = x_mini_batch * weight_mini_batch
            y_pred_mini_batch = self.pred_model(x_masked_mini_batch)
    
            self.pred_optimizer.zero_grad()
            loss = self.pred_criterion(y_pred_mini_batch, y_mini_batch)
            loss.backward()
            self.pred_optimizer.step()
    
    def get_maximum(self, input, other):
        """
        返回两个tensor中的较大值
        注：这两个tensor都是只有一个元素值
        """
        assert (len(input.shape) == len(other.shape) == 0) or (len(input.shape) == len(other.shape) == 1 and input.shape[0] == other.shape[0])
        ans = input if (input > other).item else other
        return ans

    def rl_loss(self, s_input, est_data_value, reward_input):
        """
        计算数据价值估计器的rl loss
        s_input:
        est_data_value:
        reward_input:
        """
        prob = torch.sum(
            s_input * torch.log(est_data_value + self.epsilon) + (1 - s_input) * torch.log(1 - est_data_value + self.epsilon)
        )
        return (-reward_input * prob) + \
            1e3 * (self.get_maximum(torch.mean(est_data_value)-self.threshold, 0) + self.get_maximum(1-torch.mean(est_data_value)-self.threshold, 0))

    def train_dve(self, x_input, y_input, y_hat_input, s_input, reward_input):
        """
        训练数据价值估计器
        x_input: (batch_size, feat_dim)
        y_input: (batch_size, )
        y_hat_input: (batch_size, category_num), the diff between val predictor and gt
        s_input: (batch_size, ), sample weight
        reward_input: 当前predictor与ori predictor在验证集上的性能差异，此处的两个preditor都是在训练集上训练的
        """
        est_data_value = self.dve_model(x_input, y_input, y_hat_input)  # current data value

        self.dve_optimizer.zero_grad()
        loss = self.rl_loss(s_input, est_data_value, reward_input)
        loss.backward()
        self.dve_optimizer.step()

    def train(self):
        """
        predictor和dve迭代训练
        """
        # baseline preformance
        valid_perf = self.get_performance(self.ori_model, self.valid_dataset)

        # prediction differences
        y_pred_diff = self.get_differences(self.val_model, self.train_dataset)

        # start training
        for _ in tqdm(range(self.outer_iterations)):
            # batch select
            x_batch, y_batch, y_hat_batch = self.get_random_batch(self.train_dataset, y_pred_diff)
            est_dv_curr = self.dve_model(x_batch, y_batch, y_hat_batch)
            # Samples the selection probability
            sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
            
            # train predictor with sampled batch data for inner_iterations
            self.train_predictor(x_batch, y_batch, sel_prob_curr)

            # train data value estimator with one iteration
            dvrl_perf = self.get_performance(self.pred_model, self.valid_dataset)
            reward_curr = dvrl_perf - valid_perf
            self.train_dve(x_batch, y_batch, y_hat_batch, sel_prob_curr, reward_curr)
        torch.save(self.dve_model.state_dict(), self.dve_checkpoint)

    def restore_dve(self):
        """
        加载训练好的数据价值估计器
        """
        data = torch.load(self.dve_checkpoint)
        self.dve_model.load_state_dict(data)

    def get_data_values(self, dataset, mode='train'):
        """
        估计指定数据集中各个样本的价值
        """
        if mode == 'infer':
            self.restore_dve()
        data_values = []
        self.val_model.eval()
        # y_pred_diff = self.get_differences(self.val_model, dataset)
        dataloader = DataLoader(dataset, batch_size=10)
        for x_batch, y_batch in dataloader:
            with torch.no_grad():
                y_pred_batch = self.val_model(x_batch)
            y_target_batch = one_hot(y_batch)
            y_hat_batch = torch.abs(y_target_batch - y_pred_batch)
            with torch.no_grad():
                batch_values = self.dve_model(x_batch, y_batch, y_hat_batch)
            data_values.append(batch_values.numpy())
        return np.array(data_values)
