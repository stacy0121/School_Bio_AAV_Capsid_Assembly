import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from cudaloader import CudaLoader

#from torchsummary import summary
#from dataloader import Loader
import os
import matplotlib.pyplot as plt

precision = []   # 정확도 리스트
avg_loss = []    # 손실값 리스트

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
cuda_loader = CudaLoader(dataset, world.TRAIN_epochs)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}") 
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
    
try:
    for epoch in range(world.TRAIN_epochs):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            procedure = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        # BPR Loss, average loss?
        output_information, loss = Procedure.BPR_train_original(cuda_loader, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        
        print(f'[saved][{output_information}]')
        torch.save(Recmodel.state_dict(), weight_file)
        print(f"[TOTAL TIME] {time.time() - start}")
        
        # 정확도, 손실값 리스트에 추가
        precision.append(procedure['precision'])
        avg_loss.append(loss)
finally:
    if world.tensorboard:
        w.close()

# model summary
# print(summary(Recmodel, (Loader.n_user, Loader.m_item)))

# 정확도, 손실 그래프 작성
plt.plot(precision)
plt.xlabel('epoch')
plt.ylabel('precision')
plt.show()
plt.plot(avg_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()