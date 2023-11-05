import torch
from model import LightGCN
from dataloader import BasicDataset

# 1. 학습된 모델 불러오기
model_path = "code/checkpoints/lgn-cancer-3-16.pth.tar"
dataset = BasicDataset()  # 데이터셋 초기화
model = LightGCN(dataset)  # 모델 초기화
model.load_state_dict(torch.load(model_path))  # 학습된 모델 불러오기
model.eval()  # 모델을 평가 모드로 설정

# 2. 입력 데이터에 대한 예측 수행
users = [1, 2, 3]  # 예측할 사용자 ID 리스트
items = [4, 5, 6]  # 예측할 아이템 ID 리스트

with torch.no_grad():
    users_tensor = torch.LongTensor(users)  # 입력 데이터를 텐서로 변환
    items_tensor = torch.LongTensor(items)  # 입력 데이터를 텐서로 변환
    ratings = model.getUsersRating(users_tensor)  # 예측 점수 계산

    for user, item, rating in zip(users, items, ratings):
        print(f"User {user}의 아이템 {item}에 대한 예측 점수: {rating.item()}")

# import dataloader
# import torch
# from main import recModel

# # 테스트 세트 첫번째 데이터로 예측
# test_data, user_index, item_index = dataloader.get_test_dict()   # 테스트 데이터 딕셔너리
# users = list(test_data.keys())        # 전체 항암제(user)
# user = list(test_data.keys())[0]
# item = test_data[user][0]             # 예측할 암세포(item)
# pred = recModel(item, users)          # 예측
# max_index = torch.argmax(pred)

# for key, val in item_index.items():
#     if val == item:
#         print(f'\nCancer cell: "{key}"')

# for key, val in user_index.items():
#     if val == max_index.item():
#         print(f'Predicted: "{key}"')