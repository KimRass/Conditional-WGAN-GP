# 1. Pre-trained Model
- [cwgan_gp_mnist.pth](https://drive.google.com/file/d/1WsswjXwoe4h8eCL2vhIfNAMI97-Kr-3D/view?usp=sharing)
    ```python
    seed=888
    n_epochs=50
    batch_size=64
    lr=0.0002
    d_hidden_dim=32
    g_latent_dim=100
    g_hidden_dom=32
    gp_weight=10
    n_d_updates=3
    ```

# 2. Samples
- <img src="https://github.com/KimRass/KimRass/assets/67457712/5b50540e-d4fd-41da-a163-f5b39a257fa2" width="600">

# 3. Implementation Details
## 1) Architecture
- [1]에서 Architecture를 가져와서 몇 가지를 변경했습니다.
    - Discriminator:
        - 첫 번째 Convolutional layer 다음에 Batch normalization layer를 추가했습니다.
    - Generator:
        - 마지막 Transposed convolutional layer에서 `bias=True`로 변경했습니다.
        - ReLU activation을 Leaky ReLU activation으로 변경했습니다.
- 이렇게 변경함으로써 샘플의 퀄리티가 상승했습니다.

# 4. References
- [1] https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py
