from agent import SAC_Agent

train = True
load_from = 'model/26-May-19-04.45.07/e136r13867.955340322758.pth'


agent = SAC_Agent(load_from, train)
# agent = SAC_Agent()

if train:
    agent.train()
else:
    agent.race()
