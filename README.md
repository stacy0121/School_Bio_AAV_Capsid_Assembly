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
    EPOCH[7/155]
    BPR[sample time][0.6=0.64+0.00]
    [saved][[BPR[aver loss 6.336e-01]]
    [TOTAL TIME] 1.1895248889923096
    ...
    ======================
    EPOCH[132/155]
    BPR[sample time][0.6=0.60+0.00]
    [saved][[BPR[aver loss 2.141e-01]]
    [TOTAL TIME] 1.105276107788086
    ...
    ```
- results
  |       |recall|ndcg  |precision|
  |-------|------|------|---------|
  |Layer=1|0.0235|0.2843|0.2860   |
  |Layer=2|0.0536|0.3050|0.3023   |
  |Layer=3|0.0519|0.3042|0.2837   |
  |Layer=4|0.0555|0.3133|0.2895   |

- graph  
![Figure_1](https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/fd825234-2096-4ce5-a238-cafdb00aa94b)
![Figure_2](https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/6e8be682-d990-40f4-8fc3-2ed795958b1f)

- predict
  ```
  ...
  Cancer cell: "OACp4C"
  Predicted: "AZ960"
  ```
