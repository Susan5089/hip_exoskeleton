import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import copy
import collections
import numpy as np
from pymss import EnvManager
from IPython import embed
from Model import *
import action_filter
import time


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob','s_h','a_h','r_h', 'value_h', 'logprob_h'))
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()
Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()
class PPO(object):
	def __init__(self,meta_file, mode):
		np.random.seed(seed=int(time.time()))
		self.mode = mode
		self.num_slaves = 16  # origin: 16
		self.env = EnvManager(meta_file,self.num_slaves)
		self.use_muscle = self.env.UseMuscle()
		self.use_humannn = self.env.UseHumanNetwork()
		self.use_symmetry = self.env.UseSymmetry()
		# self.num_state = self.env.GetNumState()
		self.num_human_state = self.env.GetNumHumanObservation()
		self.num_state = self.env.GetNumFullObservation()-self.num_human_state
		print("-----------------num exo state--------", self.num_state)
		self.num_human_action = self.env.GetNumHumanAction()
		if(self.use_symmetry):
			self.num_action = int(self.env.GetNumAction()/2)
		else:
			self.num_action = int(self.env.GetNumAction())
		self.num_muscles = self.env.GetNumActiveMuscles()
		self.save_path = ''

		self.num_epochs = 10
		self.num_epochs_muscle = 3
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.99
		self.lb = 0.99

		
		self.batch_size = 128
		self.muscle_batch_size = 128
		self.buffer_size = 2048   # origin: 2048
		self.replay_buffer = ReplayBuffer(30000)
		self.replay_human_buffer = ReplayBuffer(30000)
		self.muscle_buffer = MuscleBuffer(30000)
		self.model = SimulationNN(self.num_state,self.num_action)
		self.human_model = SimulationHumanNN(self.num_human_state,self.num_human_action)
		print(self.model)

		if not self.use_muscle:
			self.num_muscles = 1  # original = 0, set it to 1 as temporary value
		self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),self.num_human_action,self.num_muscles)
		if use_cuda:
			self.model.cuda()
			self.muscle_model.cuda()
			self.human_model.cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
		self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate)
		self.optimizer_human = optim.Adam(self.human_model.parameters(),lr=self.learning_rate)
		self.max_iteration = 120000

		self.w_entropy = -0.001

		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.loss_muscle = 0.0
		self.rewards = []
		self.rewards_human=[]

		self.sum_return = 0.0
		self.sum_return_human = 0.0

		self.max_return = -1.0
		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()

		self._action_filter = [self._BuildActionFilter() for _i in range(self.num_slaves)]
		self._action_filter_human = [self._BuildHumanActionFilter() for _i in range(self.num_slaves)]
		self.env.Resets(True)

		# self._history_buffer_state = collections.deque(maxlen=3)
		# self._history_buffer_action = collections.deque(maxlen=3)
		# self._last_action = self.env.GetActions()
		#
		# while len(self._history_buffer_state) < 3:
		# 	self._history_buffer_state.appendleft(self.env.GetStates())
		# while len(self._history_buffer_action) < 3:
		# 	self._history_buffer_action.appendleft(self.env.GetActions())

	def _BuildActionFilter(self):
		sampling_rate = self.num_control_Hz #1 / (self.time_step * self._action_repeat)
		num_joints = self.num_action
		a_filter = action_filter.ActionFilterButter(
			sampling_rate=sampling_rate, num_joints=num_joints, filter_low_cut = 0, filter_high_cut = 8)
		return a_filter

	def _BuildHumanActionFilter(self):
		sampling_rate = self.num_control_Hz #1 / (self.time_step * self._action_repeat)
		num_joints = self.num_human_action
		a_filter = action_filter.ActionFilterButter(
			sampling_rate=sampling_rate, num_joints=num_joints, filter_low_cut = 0, filter_high_cut = 5)
		return a_filter


	def _ResetActionFilter(self):
		for filter in self._action_filter:
			filter.reset()
		return

	def _ResetHumanActionFilter(self):
		for filter in self._action_filter_human:
			filter.reset()
		return

	def _FilterAction(self, action):
		# initialize the filter history, since resetting the filter will fill
		# the history with zeros and this can cause sudden movements at the start
		# of each episode
		for i in range(action.shape[0]):
			if sum(self._action_filter[i].xhist[0])[0] == 0:
				self._action_filter[i].init_history(action[i])

		filtered_action = []
		for i in range(action.shape[0]):
			filtered_action.append(self._action_filter[i].filter(action[i]))
		return np.vstack(filtered_action)

	def _FilterHumanAction(self, action):
		# initialize the filter history, since resetting the filter will fill
		# the history with zeros and this can cause sudden movements at the start
		# of each episode
		for i in range(action.shape[0]):
			if sum(self._action_filter_human[i].xhist[0])[0] == 0:
				self._action_filter_human[i].init_history(action[i])

		filtered_action_human = []
		for i in range(action.shape[0]):
			filtered_action_human.append(self._action_filter_human[i].filter(action[i]))
		return np.vstack(filtered_action_human)


	def SetSaveModePath(self, path):
		self.save_path = path

	def SaveModel(self, path):
		if not os.path.exists('../nn/'+path):
			os.makedirs('../nn/'+path)
			print('create folder ../nn/'+path)

		self.model.save('../nn/'+path+'/current.pt')
		self.muscle_model.save('../nn/'+path+'/current_muscle.pt')
		self.human_model.save('../nn/'+path+'/current_human.pt')
		
		if self.max_return_epoch == self.num_evaluation:
			self.model.save('../nn/'+path+'/max.pt')
			self.muscle_model.save('../nn/'+path+'/max_muscle.pt')
			self.human_model.save('../nn/'+path+'/max_human.pt')
		if self.num_evaluation%100 == 0:
			self.model.save('../nn/'+ path + '/'+str(self.num_evaluation//100)+'.pt')
			self.muscle_model.save('../nn/'+ path + '/'+ str(self.num_evaluation//100)+'_muscle.pt')
			self.human_model.save('../nn/'+ path + '/'+ str(self.num_evaluation//100)+'_human.pt')

	def LoadModel(self,path):
		print('load')
		if self.mode == "train_all":
			self.model.load('../nn/'+path+'.pt')
			self.muscle_model.load('../nn/'+path+'_muscle.pt')
			self.human_model.load('../nn/'+path+'_human.pt')
		elif self.mode == "train_muscle_only":
			self.model.load('../nn/'+path+'.pt')
			self.human_model.load('../nn/'+path+'_human.pt')
		else:
			raise ValueError("Mode {} not implemented".format(mode))


	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
		self.replay_human_buffer.Clear()
		self.muscle_buffer.Clear()
		self.sum_return = 0.0
		self.sum_return_human = 0.0

		for epi in self.total_episodes:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			
			states, actions, rewards, values, logprobs, \
			states_human, actions_human, rewards_human, values_human, logprobs_human = zip(*data)

			# for skeleton buffer 
			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0
			if np.any(np.isnan(rewards)):
				print("nonrewards")
			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			self.sum_return += epi_return
			TD = values[:size] + advantages
			if np.any(np.isnan(TD)):
				print("nanTD")
			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

			# for human buffer 
			values_human = np.concatenate((values_human, np.zeros(1)), axis=0)
			advantages_human = np.zeros(size)
			ad_t_human = 0
			if np.any(np.isnan(rewards_human)):
				print("nonhumanreward")
			epi_return_human= 0.0
			for i in reversed(range(len(data))):
				epi_return_human += rewards_human[i]
				delta_human = rewards_human[i] + values_human[i+1] * self.gamma - values_human[i]
				ad_t_human = delta_human + self.gamma * self.lb * ad_t_human
				advantages_human[i] = ad_t_human
			self.sum_return_human += epi_return_human
			TD_human = values_human[:size] + advantages_human
			if np.any(np.isnan(TD_human)):
				print("nonhumantd")
			for i in range(size):
				self.replay_human_buffer.Push(states_human[i], actions_human[i], logprobs_human[i], TD_human[i], advantages_human[i])


		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

		muscle_tuples = self.env.GetMuscleTuples()
		for i in range(len(muscle_tuples)):
			self.muscle_buffer.Push(muscle_tuples[i][0],muscle_tuples[i][1],muscle_tuples[i][2],muscle_tuples[i][3])


	def GenerateTransitions(self):
		self.total_episodes = []
		# states = [None]*self.num_slaves
		# actions = [None]*self.num_slaves
		rewards = [None]*self.num_slaves
		rewards_human = [None] * self.num_slaves
		# states_next = [None]*self.num_slaves
		# states = self.env.GetStates()
		states = self.env.GetFullObservations()[:,0:-self.num_human_state]
		states_human = self.env.GetFullObservations()[:,-self.num_human_state:]
		local_step = 0
		terminated = [False]*self.num_slaves
		counter = 0
		counter_list = [0]*self.num_slaves
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')
			a_dist,v = self.model(Tensor(states))
			a_dist_human,v_human = self.human_model(Tensor(states_human))

			actions = a_dist.sample().cpu().detach().numpy()
			actions_human = a_dist_human.sample().cpu().detach().numpy()

			# actions = a_dist.loc.cpu().detach().numpy()
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			logprobs_human = a_dist_human.log_prob(Tensor(actions_human)).cpu().detach().numpy().reshape(-1)

			values = v.cpu().detach().numpy().reshape(-1)
			values_human = v_human.cpu().detach().numpy().reshape(-1)

			filterd_actions = self._FilterAction(actions).astype(np.float32)
			# filterd_actions_human = self._FilterHumanAction(actions_human).astype(np.float32)

			filterd_actions = actions
			filterd_actions_human = actions_human

			if(self.use_symmetry):
				actions_full = np.concatenate((filterd_actions, filterd_actions), axis=1)
			else:
				actions_full = filterd_actions

			self.env.SetActions(actions_full)
			self.env.SetHumanActions(filterd_actions_human)
			self.env.UpdateActionBuffers(actions_full)  # update action buffer
			# self.env.UpdateHumanActionBuffers(filterd_actions_human)
			if self.use_muscle:
				mt = Tensor(self.env.GetMuscleTorques())
				for i in range(self.num_simulation_per_control//2):
					dt = Tensor(self.env.GetDesiredTorques()[:,:self.num_human_action])
					activations = self.muscle_model(mt,dt).cpu().detach().numpy()
					self.env.SetActivationLevels(activations)
					self.env.Steps(2, i*2)
					# print(i)
					# Steps(int num, int doneStep)

				self.env.UpdateStateBuffers()  # update state buffer
			else:
				self.env.StepsAtOnce()  # 20 step with same action
				self.env.UpdateStateBuffers()  # update state buffer


			for j in range(self.num_slaves):
				nan_occur = False	
				terminated_state = True
				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])) or\
					np.any(np.isnan(states_human[j])) or np.any(np.isnan(actions_human[j])) or np.any(np.isnan(values_human[j])) or \
					np.any(np.isnan(logprobs_human[j])):
					print("State")
					print(states[j])
					print("Actions")
					print(actions[j])
					print("Values")
					print(values[j])
					print("logprobs")
					print(logprobs[j])
					nan_occur = True
				
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False
					rewards[j] = self.env.GetReward(j)
					rewards_human[j] = self.env.GetHumanReward(j)
					if (np.any(np.isnan(rewards[j]))) or (np.any(np.isnan(rewards_human[j]))):
						pass
					else:
						self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j], \
												states_human[j], actions_human[j], rewards_human[j], values_human[j], logprobs_human[j])
						local_step += 1
						counter_list[j] += 1

				if terminated_state or (nan_occur is True):
					# print("Terminated {}: has {} steps ".format(j, counter_list[j]))
					counter_list[j] = 0 

					if (nan_occur is True):
						self.episodes[j].Pop()
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()
					self.env.Reset(True, j) # state/action buffer is reset too.
					self._action_filter[j].init_history(self.env.GetAction(j)[:self.num_action])
					self._action_filter_human[j].init_history(self.env.GetHumanAction(j)[:self.num_human_action])
			if local_step >= self.buffer_size:
				break

			# states = self.env.GetFullObservations()
			states = self.env.GetFullObservations()[:,0:-self.num_human_state]
			states_human = self.env.GetFullObservations()[:,-self.num_human_state:]

	def OptimizeSimulationNN(self):
		all_transitions = np.array(self.replay_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)

				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.model(Tensor(stack_s))
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer.step()
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')


	def OptimizeSimulationHumanNN(self):
		all_transitions = np.array(self.replay_human_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.human_model(Tensor(stack_s))
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor_human = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic_human = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer_human.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.human_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer_human.step()
			print('Optimizing humansim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

	def OptimizeMuscleNN(self):
		muscle_transitions = np.array(self.muscle_buffer.buffer)
		for j in range(self.num_epochs_muscle):
			np.random.shuffle(muscle_transitions)
			for i in range(len(muscle_transitions)//self.muscle_batch_size):
				tuples = muscle_transitions[i*self.muscle_batch_size:(i+1)*self.muscle_batch_size]
				batch = MuscleTransition(*zip(*tuples))

				stack_JtA = np.vstack(batch.JtA).astype(np.float32)
				stack_tau_des = np.vstack(batch.tau_des).astype(np.float32)
				stack_L = np.vstack(batch.L).astype(np.float32)

				stack_L = stack_L.reshape(self.muscle_batch_size,self.num_human_action,self.num_muscles)
				stack_b = np.vstack(batch.b).astype(np.float32)

				stack_JtA = Tensor(stack_JtA)
				stack_tau_des = Tensor(stack_tau_des)
				stack_L = Tensor(stack_L)
				stack_b = Tensor(stack_b)

				activation = self.muscle_model(stack_JtA,stack_tau_des)
				tau = torch.einsum('ijk,ik->ij',(stack_L,activation)) + stack_b

				loss_reg = (activation).pow(2).mean()
				loss_target = (((tau-stack_tau_des)/10).pow(2)).mean()              #/100
				loss = 0.01*loss_reg + loss_target
				# loss = loss_target
				self.optimizer_muscle.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.muscle_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer_muscle.step()

			print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
		self.loss_muscle = loss.cpu().detach().numpy().tolist()
		print('')

		
	def OptimizeModel(self):
		if self.mode == "train_all":
			self.ComputeTDandGAE()
			self.OptimizeSimulationNN()
			if self.use_humannn:
				self.OptimizeSimulationHumanNN()
			if self.use_muscle:
				self.OptimizeMuscleNN()
		elif self.mode == "train_muscle_only":
			self.ComputeTDandGAE()
			if self.use_muscle:
				self.OptimizeMuscleNN()


	def Train(self):
		print("Start Generating Transitions.")
		start = time.process_time()
		self.GenerateTransitions()
		print("GenerateTransitions: {:.2f}s".format(time.process_time() - start))

		print("Start Optimizing Model.")
		start = time.process_time()
		self.OptimizeModel()
		print("OptimizeModel: {:.2f}s".format(time.process_time() - start))
	
	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode is 0:
			self.num_episode = 1
		if self.num_tuple is 0:
			self.num_tuple = 1
		if self.max_return < self.sum_return_human/self.num_episode:
			self.max_return = self.sum_return_human/self.num_episode
			self.max_return_epoch = self.num_evaluation
		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor_human))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic_human))
		print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
		# print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return_human/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return_human/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
		self.rewards.append(self.sum_return_human/self.num_episode)
		# self.rewards_human.append(self.sum_return_human/self.num_episode)
		self.SaveModel(self.save_path)
		
		print('=============================================')
		return np.array(self.rewards)

import matplotlib
import matplotlib.pyplot as plt

plt.ion()

def Plot(y,title, file_name, num_fig=1,ylim=True):
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'b')
	
	plt.plot(temp_y,'r')

	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	plt.savefig('{}.png'.format(file_name),  bbox_inches='tight') 

def Plot_Error(y,title, file_name, num_fig=1,ylim=True):

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'g')
	
	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	plt.savefig('{}.png'.format(file_name),  bbox_inches='tight') 


import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-d','--meta',help='meta file')
	parser.add_argument('-r','--dir', help='save path')
	parser.add_argument('--mode', default= "train_all", 
						help='specific mode: train_muscle_only: only train muscle; train_all: train three models')

	args =parser.parse_args()
	if args.meta is None:
		print('Provide meta file')
		exit()

	ppo = PPO(args.meta, args.mode)

	nn_dir = '../nn'
	
	if not os.path.exists(nn_dir):
		os.makedirs(nn_dir)
	
	if args.dir is not None:
		ppo.SetSaveModePath(args.dir)
		if not os.path.exists('../nn/'+args.dir):
			os.makedirs('../nn/'+args.dir)
			print('Created directory ../nn/' + args.dir)

	if args.model is not None:
		ppo.LoadModel(args.model)
	else:
		if args.dir is not None:
			ppo.SaveModel(args.dir)
		else:
			ppo.SaveModel('')

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))
	for i in range(ppo.max_iteration-5):
		ppo.Train()
		rewards = ppo.Evaluate()
		Plot(rewards,'reward', "reward", 0,False)
		# Plot(rewards_human,'reward_human', "reward_human", 0,False)