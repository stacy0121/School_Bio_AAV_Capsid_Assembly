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
from dataloader import Loader
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
            procedure, recModel = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])   # 예측
        # BPR Loss, average loss?
        output_information, loss = Procedure.BPR_train_original(cuda_loader, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        
        print(f'[saved][{output_information}]')
        torch.save(Recmodel.state_dict(), weight_file)   # 모델 저장
        print(f"[TOTAL TIME] {time.time() - start}")

        #----------------------------------------------------------
        # 정확도, 손실값 리스트에 추가
        precision.append(procedure['precision'])
        avg_loss.append(loss)

        # 검증 오차가 가장 적은 최적의 모델을 저장
        # if not best_val_loss or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model
finally:
    if world.tensorboard:
        w.close()

# model summary
# print(summary(Recmodel, (Loader.n_user, Loader.m_item)))

## 정확도, 손실 그래프 작성
plt.plot(precision)
plt.xlabel('epoch')
plt.ylabel('precision')
plt.show()
plt.plot(avg_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

## 테스트 세트 첫번째 데이터로 예측
test_data, user_index, item_index = dataset.get_test_dict()   # 테스트 데이터 딕셔너리
users = list(test_data.keys())        # 전체 항암제(user)
user = list(test_data.keys())[0]
item = test_data[user][0]             # 예측할 암세포(item)
pred = recModel(item, users)          # 예측
max_index = torch.argmax(pred)

for key, val in item_index.items():
    if val == item:
        print(f'\nCancer cell: "{key}"')

for key, val in user_index.items():
    if val == max_index.item():
        print(f'Predicted: "{key}"')