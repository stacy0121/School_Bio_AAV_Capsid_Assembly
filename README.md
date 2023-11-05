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
    EPOCH[7/170]
    BPR[sample time][0.5=0.44+0.00]
    [saved][[BPR[aver loss 6.854e-01]]
    [TOTAL TIME] 1.7976963520050049
    ...
    ======================
    EPOCH[10/170]
    [TEST]
    {'precision': array([0.07751938]), 'recall': array([0.00042482]), 'ndcg': array([0.07751938])}
    BPR[sample time][0.4=0.40+0.00]
    [saved][[BPR[aver loss 6.736e-01]]
    [TOTAL TIME] 1.8793020248413086
    ...
    ======================
    EPOCH[129/170]
    BPR[sample time][0.4=0.39+0.00]
    [saved][[BPR[aver loss 3.944e-01]]
    [TOTAL TIME] 1.7875397205352783
    ...
    ```
- results
  |       |recall|ndcg  |precision|
  |-------|------|------|---------|
  |Layer=1|0.0017|0.1279|0.1279   |
  |Layer=2|0.0004|0.0775|0.0775   |
  |Layer=3|0.0011|0.1240|0.1240   |
  |Layer=4|0.0019|0.1628|0.1628   |

- graph
<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/2ee54a5c-62dd-4942-a2b4-f24d62dc0ce2.png" width="500" height="400"/>


Testing every 10 epochs

<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/43b762b7-1463-4675-8afc-01eb2a0d180b.png" width="500" height="400"/>
<img src="https://github.com/stacy0121/AAV_Capsid_Assembly/assets/72933504/245debb3-a368-4787-8b23-bcde4ae59316.png" width="500" height="400"/>
