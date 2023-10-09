import world
import dataloader
# import model
import model_grand as model
import utils
from pprint import pprint

if world.dataset in ['cancer']:
    dataset = dataloader.Loader("../data/"+world.dataset)

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