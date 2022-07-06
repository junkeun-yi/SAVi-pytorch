# SAVi-pytorch

```
+-----------------------------------------------------------+-----------------+---------+-----------+---------+
| Name                                                      | Shape           | Size    | Mean      | Std     |
+-----------------------------------------------------------+-----------------+---------+-----------+---------+
| encoder.backbone.cnn_layers.conv_0.weight                 | (32, 3, 5, 5)   | 2,400   | -0.000433 | 0.117   |
| encoder.backbone.cnn_layers.conv_0.bias                   | (32,)           | 32      | 0.0       | 0.0     |
| encoder.backbone.cnn_layers.conv_1.weight                 | (32, 32, 5, 5)  | 25,600  | -3.37e-05 | 0.0356  |
| encoder.backbone.cnn_layers.conv_1.bias                   | (32,)           | 32      | 0.0       | 0.0     |
| encoder.backbone.cnn_layers.conv_2.weight                 | (32, 32, 5, 5)  | 25,600  | 7.71e-06  | 0.0354  |
| encoder.backbone.cnn_layers.conv_2.bias                   | (32,)           | 32      | 0.0       | 0.0     |
| encoder.backbone.cnn_layers.conv_3.weight                 | (32, 32, 5, 5)  | 25,600  | 9.38e-05  | 0.0356  |
| encoder.backbone.cnn_layers.conv_3.bias                   | (32,)           | 32      | 0.0       | 0.0     |
| encoder.pos_emb.pos_embedding                             | (1, 64, 64, 2)  | 8,192   | -1.49e-08 | 0.586   |
| encoder.pos_emb.output_transform.layernorm_module.weight  | (32,)           | 32      | 1.0       | 0.0     |
| encoder.pos_emb.output_transform.layernorm_module.bias    | (32,)           | 32      | 0.0       | 0.0     |
| encoder.pos_emb.output_transform.model.dense_mlp_0.weight | (64, 32)        | 2,048   | 0.00425   | 0.181   |
| encoder.pos_emb.output_transform.model.dense_mlp_0.bias   | (64,)           | 64      | 0.0       | 0.0     |
| encoder.pos_emb.output_transform.model.dense_mlp_1.weight | (32, 64)        | 2,048   | 0.00412   | 0.125   |
| encoder.pos_emb.output_transform.model.dense_mlp_1.bias   | (32,)           | 32      | 0.0       | 0.0     |
| encoder.pos_emb.project_add_dense.weight                  | (32, 2)         | 64      | -0.057    | 0.663   |
| encoder.pos_emb.project_add_dense.bias                    | (32,)           | 32      | 0.0       | 0.0     |
| decoder.backbone.cnn_layers.convtranspose_0.weight        | (128, 64, 5, 5) | 204,800 | -1.5e-05  | 0.025   |
| decoder.backbone.cnn_layers.convtranspose_0.bias          | (64,)           | 64      | 0.0       | 0.0     |
| decoder.backbone.cnn_layers.convtranspose_1.weight        | (64, 64, 5, 5)  | 102,400 | -1.8e-05  | 0.0251  |
| decoder.backbone.cnn_layers.convtranspose_1.bias          | (64,)           | 64      | 0.0       | 0.0     |
| decoder.backbone.cnn_layers.convtranspose_2.weight        | (64, 64, 5, 5)  | 102,400 | 7.25e-05  | 0.025   |
| decoder.backbone.cnn_layers.convtranspose_2.bias          | (64,)           | 64      | 0.0       | 0.0     |
| decoder.backbone.cnn_layers.conv_3.weight                 | (64, 64, 5, 5)  | 102,400 | -8.28e-05 | 0.025   |
| decoder.backbone.cnn_layers.conv_3.bias                   | (64,)           | 64      | 0.0       | 0.0     |
| decoder.pos_emb.pos_embedding                             | (1, 8, 8, 2)    | 128     | 0.0       | 0.657   |
| decoder.pos_emb.project_add_dense.weight                  | (128, 2)        | 256     | 0.0554    | 0.747   |
| decoder.pos_emb.project_add_dense.bias                    | (128,)          | 128     | 0.0       | 0.0     |
| decoder.target_readout.readout_modules.0.weight           | (3, 64)         | 192     | -0.00339  | 0.123   |
| decoder.target_readout.readout_modules.0.bias             | (3,)            | 3       | 0.0       | 0.0     |
| decoder.mask_pred.weight                                  | (1, 64)         | 64      | -0.00369  | 0.136   |
| decoder.mask_pred.bias                                    | (1,)            | 1       | 0.0       | nan     |
| corrector.w_q                                             | (1, 128, 128)   | 16,384  | -8.51e-05 | 0.00784 |
| corrector.w_k                                             | (1, 32, 128)    | 4,096   | 5.85e-05  | 0.0154  |
| corrector.w_v                                             | (1, 32, 128)    | 4,096   | -0.000318 | 0.0155  |
| corrector.layernorm_input.weight                          | (32,)           | 32      | 1.0       | 0.0     |
| corrector.layernorm_input.bias                            | (32,)           | 32      | 0.0       | 0.0     |
| corrector.layernorm_q.weight                              | (128,)          | 128     | 1.0       | 0.0     |
| corrector.layernorm_q.bias                                | (128,)          | 128     | 0.0       | 0.0     |
| corrector.gru.weight_ih                                   | (384, 128)      | 49,152  | -0.000535 | 0.0884  |
| corrector.gru.weight_hh                                   | (384, 128)      | 49,152  | -8.47e-05 | 0.051   |
| corrector.gru.bias_ih                                     | (384,)          | 384     | 0.0       | 0.0     |
| corrector.gru.bias_hh                                     | (384,)          | 384     | 0.0       | 0.0     |
| predictor.w_qkv.weight                                    | (384, 128)      | 49,152  | 0.000468  | 0.0881  |
| predictor.w_qkv.bias                                      | (384,)          | 384     | 0.0       | 0.0     |
| predictor.w_o.weight                                      | (128, 128)      | 16,384  | 5.43e-05  | 0.0892  |
| predictor.w_o.bias                                        | (128,)          | 128     | 0.0       | 0.0     |
| predictor.mlp.model.dense_mlp_0.weight                    | (256, 128)      | 32,768  | -0.000473 | 0.0881  |
| predictor.mlp.model.dense_mlp_0.bias                      | (256,)          | 256     | 0.0       | 0.0     |
| predictor.mlp.model.dense_mlp_1.weight                    | (128, 256)      | 32,768  | 0.000162  | 0.0624  |
| predictor.mlp.model.dense_mlp_1.bias                      | (128,)          | 128     | 0.0       | 0.0     |
| predictor.layernorm_query.weight                          | (128,)          | 128     | 1.0       | 0.0     |
| predictor.layernorm_query.bias                            | (128,)          | 128     | 0.0       | 0.0     |
| predictor.layernorm_mlp.weight                            | (128,)          | 128     | 1.0       | 0.0     |
| predictor.layernorm_mlp.bias                              | (128,)          | 128     | 0.0       | 0.0     |
| initializer.embedding_transform.model.dense_mlp_0.weight  | (256, 4)        | 1,024   | -0.00106  | 0.514   |
| initializer.embedding_transform.model.dense_mlp_0.bias    | (256,)          | 256     | 0.0       | 0.0     |
| initializer.embedding_transform.model.dense_mlp_1.weight  | (128, 256)      | 32,768  | 0.000597  | 0.0624  |
| initializer.embedding_transform.model.dense_mlp_1.bias    | (128,)          | 128     | 0.0       | 0.0     |
+-----------------------------------------------------------+-----------------+---------+-----------+---------+
Total: 895,524
SAVi(
  (encoder): FrameEncoder(
    (backbone): CNN(
      (cnn_layers): ModuleList(
        (conv_0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=same)
        (activ_0): ReLU()
        (conv_1): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=same)
        (activ_1): ReLU()
        (conv_2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=same)
        (activ_2): ReLU()
        (conv_3): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=same)
        (activ_3): ReLU()
      )
    )
    (pos_emb): PositionEmbedding(
      (pos_transform): Identity()
      (output_transform): MLP(
        (layernorm_module): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (model): ModuleList(
          (dense_mlp_0): Linear(in_features=32, out_features=64, bias=True)
          (dense_mlp_0_act): ReLU()
          (dense_mlp_1): Linear(in_features=64, out_features=32, bias=True)
        )
      )
      (project_add_dense): Linear(in_features=2, out_features=32, bias=True)
    )
    (output_transform): Identity()
  )
  (decoder): SpatialBroadcastDecoder(
    (backbone): CNN(
      (cnn_layers): ModuleList(
        (convtranspose_0): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (activ_0): ReLU()
        (convtranspose_1): ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (activ_1): ReLU()
        (convtranspose_2): ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        (activ_2): ReLU()
        (conv_3): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=same)
        (activ_3): ReLU()
      )
    )
    (pos_emb): PositionEmbedding(
      (pos_transform): Identity()
      (output_transform): Identity()
      (project_add_dense): Linear(in_features=2, out_features=128, bias=True)
    )
    (target_readout): Readout(
      (readout_modules): ModuleList(
        (0): Linear(in_features=64, out_features=3, bias=True)
      )
    )
    (mask_pred): Linear(in_features=64, out_features=1, bias=True)
  )
  (corrector): SlotAttention(
    (layernorm_input): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
    (layernorm_q): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (inverted_attention): InvertedDotProductAttention(
      (attn_fn): GeneralizedDotProductAttention()
    )
    (gru): GRUCell(128, 128)
  )
  (predictor): TransformerBlock(
    (w_qkv): Linear(in_features=128, out_features=384, bias=True)
    (w_o): Linear(in_features=128, out_features=128, bias=True)
    (attn): GeneralizedDotProductAttention()
    (mlp): MLP(
      (model): ModuleList(
        (dense_mlp_0): Linear(in_features=128, out_features=256, bias=True)
        (dense_mlp_0_act): ReLU()
        (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (layernorm_query): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (layernorm_mlp): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (initializer): CoordinateEncoderStateInit(
    (embedding_transform): MLP(
      (model): ModuleList(
        (dense_mlp_0): Linear(in_features=4, out_features=256, bias=True)
        (dense_mlp_0_act): ReLU()
        (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
      )
    )
  )
  (processor): Processor(
    (corrector): SlotAttention(
      (layernorm_input): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (layernorm_q): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (inverted_attention): InvertedDotProductAttention(
        (attn_fn): GeneralizedDotProductAttention()
      )
      (gru): GRUCell(128, 128)
    )
    (predictor): TransformerBlock(
      (w_qkv): Linear(in_features=128, out_features=384, bias=True)
      (w_o): Linear(in_features=128, out_features=128, bias=True)
      (attn): GeneralizedDotProductAttention()
      (mlp): MLP(
        (model): ModuleList(
          (dense_mlp_0): Linear(in_features=128, out_features=256, bias=True)
          (dense_mlp_0_act): ReLU()
          (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
        )
      )
      (layernorm_query): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (layernorm_mlp): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    )
  )
)
```