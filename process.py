import wave
import numpy as np
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


correspondence={'1.1': 0, '1.2': 1, '1.3': 2, '1.4': 3, '1.5': 4, '2.1': 5, '2.2': 6, '2.3': 7, '2.4': 8, '2.5': 9, '2.6': 10, '2.7': 11, '2.9': 12, '2.10': 13, '3.1': 14, '3.2': 15, '3.3': 16, '3.4': 17, '4.1': 18, '4.3': 19, '4.4': 20, '4.5': 21, '5.1': 22, '5.3': 23, '5.4': 24, '5.5': 25, '5.6': 26}
train=[]
test=[]
train_answer=[]
test_answer=[]

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1=nn.Linear(48000, 200)
		self.layer2=nn.Linear(200, 100)
		self.layer3=nn.Linear(100, 27)
	
	def forward(self, x):
		x=F.relu(self.layer1(x))
		x=F.relu(self.layer2(x))
		x=self.layer3(x)
		return x


for dir in os.listdir('./edited/'):
	flag=0
	for file in glob.glob('./edited/'+dir+'/*.wav'):
		w=wave.open(file, mode='rb')
		sound=np.frombuffer(w.readframes(w.getnframes()), dtype='int16')
		sound=(sound-sound.mean())/sound.std()
		if flag:
			index=random.randint(0, len(train))
			train.insert(index, sound)
			train_answer.insert(index, correspondence[dir])
		else:
			index=random.randint(0, len(test))
			test.insert(index, sound)
			test_answer.insert(index, correspondence[dir])
		flag+=1
		flag=flag%2
		w.close()

train=np.array(train, dtype='float32')
test=np.array(test, dtype='float32')
train_answer=np.array(train_answer, dtype='int64')
test_answer=np.array(test_answer, dtype='int64')

net=Net()
criterion=nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

data_start=0
batch_size=50
for epoch in range(1000):
	input=torch.from_numpy(train[data_start*batch_size:(data_start+1)*batch_size])
	target=torch.from_numpy(train_answer[data_start*batch_size:(data_start+1)*batch_size])
	data_start+=1
	if (data_start+1)*batch_size>len(train):
		data_start=0
	optimizer.zero_grad()
	output=net(input)
	loss=criterion(output, target)
	loss.backward()
	optimizer.step()
	print(loss.item())

infer_data=net(torch.from_numpy(test))
infer_label=torch.argmax(infer_data, dim=1)
accuracy=infer_label.numpy()-test_answer==0
print(np.sum(accuracy)/len(accuracy))
