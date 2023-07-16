# AAV_Capsid_Assembly
아데노관련바이러스(AAV) capsid assembly 가능성 예측을 위한 모델 구축 연구 참여자 기록     
[참고]
https://github.com/gusye1234/LightGCN-PyTorch

### 1. User-based Collaborative Filtering, Content-based Filtering
<https://colab.research.google.com/drive/1Oge9cgDc3nmkXce_L9t9OAFPZ50TKdVa?usp=chrome_ntp#scrollTo=v9qghEP3L_x1>
      
### 2. pytorch lightgcn - python scripts
  - log output
    
    ```
    ...
    ======================
    EPOCH[7/1000]
    BPR[sample time][0.6=0.63+0.01]
    [saved][[BPR[aver loss 6.336e-01]]
    [TOTAL TIME] 1.22074294090271
    ...
    ======================
    EPOCH[132/1000]
    BPR[sample time][0.6=0.58+0.00]
    [saved][[BPR[aver loss 2.141e-01]]
    [TOTAL TIME] 1.1073217391967773
    ...
    ```
- results
  |       |recall|ndcg  |precision|
  |-------|------|------|---------|
  |Layer=1|0.0235|0.2843|0.2860   |
  |Layer=2|0.0536|0.3050|0.3023   |
  |Layer=3|0.0519|0.3042|0.2837   |
  |Layer=4|0.0555|0.3133|0.2895   |
  
