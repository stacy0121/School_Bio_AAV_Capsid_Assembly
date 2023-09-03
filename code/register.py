import world
import dataloader
# import model
import model_grand as model
import utils
from pprint import pprint

if world.dataset in ['cancer']:     # 변경
    dataset = dataloader.Loader("../data/"+world.dataset)
    # data_test = dataloader.Loader("../data/"+world.dataset, mode='test')

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'lgn': model.LightGCN
}