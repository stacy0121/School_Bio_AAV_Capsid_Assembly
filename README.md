# AAV_Capsid_Assembly
아데노관련바이러스(AAV) capsid assembly 가능성 예측을 위한 모델 구축 연구 참여자 기록     
[참고]
https://github.com/gusye1234/LightGCN-PyTorch

### 1. User-based Collaborative Filtering, Content-based Filtering
<https://colab.research.google.com/drive/1Oge9cgDc3nmkXce_L9t9OAFPZ50TKdVa?usp=chrome_ntp#scrollTo=v9qghEP3L_x1>
      
### 2. Pytorch lightgcn - python scripts
  - log output
    
    ```
    ...
    ======================
    EPOCH[7/130]
    BPR[sample time][0.4=0.40+0.01]
    [saved][[BPR[aver loss 6.854e-01]]
    [TOTAL TIME] 1.6297967433929443
    ...
    ======================
    EPOCH[113/130]
    BPR[sample time][0.4=0.43+0.01]
    [saved][[BPR[aver loss 4.045e-01]]
    [TOTAL TIME] 1.6821250915527344
    ...
    ```
- results
  |       |recall|ndcg  |precision|
  |-------|------|------|---------|
  |Layer=1|0.0414|0.1352|0.1291   |
  |Layer=2|0.0651|0.1239|0.1141   |
  |Layer=3|0.0724|0.1552|0.1333   |
  |Layer=4|0.0706|0.1653|0.1388   |

- graph
<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/5773f16c-62da-4c20-885e-1371a8627fb6.png" width="500" height="400"/>

Testing every 10 epochs

<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/d92349c9-c3db-48df-9a5b-ba8c88ec4cdd.png" width="500" height="400"/>
<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/d50522c1-a652-46d8-bccf-fc6aa16942b6.png" width="500" height="400"/>
