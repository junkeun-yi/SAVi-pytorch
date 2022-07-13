# SAVi-pytorch

```
+-----------------------------------------------------------+-----------------+---------+-----------+--------+
| Name                                                      | Shape           | Size    | Mean      | Std    |
+-----------------------------------------------------------+-----------------+---------+-----------+--------+
| initializer.embedding_transform.model.dense_mlp_0.weight  | (256, 4)        | 1,024   | -0.00731  | 0.506  |
| initializer.embedding_transform.model.dense_mlp_0.bias    | (256,)          | 256     | 0.0       | 0.0    |
| initializer.embedding_transform.model.dense_mlp_1.weight  | (128, 256)      | 32,768  | 0.000184  | 0.062  |
| initializer.embedding_transform.model.dense_mlp_1.bias    | (128,)          | 128     | 0.0       | 0.0    |
| encoder.backbone.cnn_layers.conv_0.weight                 | (32, 3, 5, 5)   | 2,400   | -0.00121  | 0.115  |
| encoder.backbone.cnn_layers.conv_0.bias                   | (32,)           | 32      | 0.0       | 0.0    |
| encoder.backbone.cnn_layers.conv_1.weight                 | (32, 32, 5, 5)  | 25,600  | 1.36e-05  | 0.0354 |
| encoder.backbone.cnn_layers.conv_1.bias                   | (32,)           | 32      | 0.0       | 0.0    |
| encoder.backbone.cnn_layers.conv_2.weight                 | (32, 32, 5, 5)  | 25,600  | 3.66e-05  | 0.0354 |
| encoder.backbone.cnn_layers.conv_2.bias                   | (32,)           | 32      | 0.0       | 0.0    |
| encoder.backbone.cnn_layers.conv_3.weight                 | (32, 32, 5, 5)  | 25,600  | 9.38e-05  | 0.0356 |
| encoder.backbone.cnn_layers.conv_3.bias                   | (32,)           | 32      | 0.0       | 0.0    |
| encoder.pos_emb.pos_embedding                             | (1, 64, 64, 2)  | 8,192   | -1.49e-08 | 0.586  |
| encoder.pos_emb.output_transform.layernorm_module.weight  | (32,)           | 32      | 1.0       | 0.0    |
| encoder.pos_emb.output_transform.layernorm_module.bias    | (32,)           | 32      | 0.0       | 0.0    |
| encoder.pos_emb.output_transform.model.dense_mlp_0.weight | (64, 32)        | 2,048   | 0.00425   | 0.181  |
| encoder.pos_emb.output_transform.model.dense_mlp_0.bias   | (64,)           | 64      | 0.0       | 0.0    |
| encoder.pos_emb.output_transform.model.dense_mlp_1.weight | (32, 64)        | 2,048   | 0.00412   | 0.125  |
| encoder.pos_emb.output_transform.model.dense_mlp_1.bias   | (32,)           | 32      | 0.0       | 0.0    |
| encoder.pos_emb.project_add_dense.weight                  | (32, 2)         | 64      | -0.057    | 0.663  |
| encoder.pos_emb.project_add_dense.bias                    | (32,)           | 32      | 0.0       | 0.0    |
| corrector.gru.dense_ir.weight                             | (128, 128)      | 16,384  | -0.00144  | 0.0886 |
| corrector.gru.dense_ir.bias                               | (128,)          | 128     | 0.0       | 0.0    |
| corrector.gru.dense_iz.weight                             | (128, 128)      | 16,384  | 3.85e-06  | 0.0891 |
| corrector.gru.dense_iz.bias                               | (128,)          | 128     | 0.0       | 0.0    |
| corrector.gru.dense_in.weight                             | (128, 128)      | 16,384  | -0.00101  | 0.0876 |
| corrector.gru.dense_in.bias                               | (128,)          | 128     | 0.0       | 0.0    |
| corrector.gru.dense_hr.weight                             | (128, 128)      | 16,384  | 0.000347  | 0.0884 |
| corrector.gru.dense_hz.weight                             | (128, 128)      | 16,384  | 0.000198  | 0.0884 |
| corrector.gru.dense_hn.weight                             | (128, 128)      | 16,384  | -0.000997 | 0.0884 |
| corrector.gru.dense_hn.bias                               | (128,)          | 128     | 0.0       | 0.0    |
| corrector.dense_q.weight                                  | (128, 128)      | 16,384  | -0.000674 | 0.0887 |
| corrector.dense_k.weight                                  | (128, 32)       | 4,096   | -0.00172  | 0.179  |
| corrector.dense_v.weight                                  | (128, 32)       | 4,096   | 0.0034    | 0.179  |
| corrector.layernorm_q.weight                              | (128,)          | 128     | 1.0       | 0.0    |
| corrector.layernorm_q.bias                                | (128,)          | 128     | 0.0       | 0.0    |
| corrector.layernorm_input.weight                          | (32,)           | 32      | 1.0       | 0.0    |
| corrector.layernorm_input.bias                            | (32,)           | 32      | 0.0       | 0.0    |
| decoder.backbone.cnn_layers.conv_0.weight                 | (128, 64, 5, 5) | 204,800 | 2.94e-05  | 0.0251 |
| decoder.backbone.cnn_layers.conv_0.bias                   | (64,)           | 64      | 0.0       | 0.0    |
| decoder.backbone.cnn_layers.conv_1.weight                 | (64, 64, 5, 5)  | 102,400 | 4.32e-05  | 0.025  |
| decoder.backbone.cnn_layers.conv_1.bias                   | (64,)           | 64      | 0.0       | 0.0    |
| decoder.backbone.cnn_layers.conv_2.weight                 | (64, 64, 5, 5)  | 102,400 | -2.19e-05 | 0.025  |
| decoder.backbone.cnn_layers.conv_2.bias                   | (64,)           | 64      | 0.0       | 0.0    |
| decoder.backbone.cnn_layers.conv_3.weight                 | (64, 64, 5, 5)  | 102,400 | 1.49e-05  | 0.025  |
| decoder.backbone.cnn_layers.conv_3.bias                   | (64,)           | 64      | 0.0       | 0.0    |
| decoder.pos_emb.pos_embedding                             | (1, 8, 8, 2)    | 128     | 0.0       | 0.657  |
| decoder.pos_emb.project_add_dense.weight                  | (128, 2)        | 256     | 0.104     | 0.707  |
| decoder.pos_emb.project_add_dense.bias                    | (128,)          | 128     | 0.0       | 0.0    |
| decoder.target_readout.readout_modules.0.weight           | (3, 64)         | 192     | 0.0142    | 0.128  |
| decoder.target_readout.readout_modules.0.bias             | (3,)            | 3       | 0.0       | 0.0    |
| decoder.mask_pred.weight                                  | (1, 64)         | 64      | -0.0126   | 0.114  |
| decoder.mask_pred.bias                                    | (1,)            | 1       | 0.0       | nan    |
| predictor.mlp.model.dense_mlp_0.weight                    | (256, 128)      | 32,768  | -0.000881 | 0.0889 |
| predictor.mlp.model.dense_mlp_0.bias                      | (256,)          | 256     | 0.0       | 0.0    |
| predictor.mlp.model.dense_mlp_1.weight                    | (128, 256)      | 32,768  | -0.000456 | 0.0624 |
| predictor.mlp.model.dense_mlp_1.bias                      | (128,)          | 128     | 0.0       | 0.0    |
| predictor.layernorm_query.weight                          | (128,)          | 128     | 1.0       | 0.0    |
| predictor.layernorm_query.bias                            | (128,)          | 128     | 0.0       | 0.0    |
| predictor.layernorm_mlp.weight                            | (128,)          | 128     | 1.0       | 0.0    |
| predictor.layernorm_mlp.bias                              | (128,)          | 128     | 0.0       | 0.0    |
| predictor.dense_q.weight                                  | (128, 128)      | 16,384  | -0.00041  | 0.0875 |
| predictor.dense_q.bias                                    | (128,)          | 128     | 0.0       | 0.0    |
| predictor.dense_k.weight                                  | (128, 128)      | 16,384  | 0.00023   | 0.089  |
| predictor.dense_k.bias                                    | (128,)          | 128     | 0.0       | 0.0    |
| predictor.dense_v.weight                                  | (128, 128)      | 16,384  | 0.00036   | 0.088  |
| predictor.dense_v.bias                                    | (128,)          | 128     | 0.0       | 0.0    |
| predictor.dense_o.weight                                  | (128, 128)      | 16,384  | -0.000598 | 0.0878 |
| predictor.dense_o.bias                                    | (128,)          | 128     | 0.0       | 0.0    |
+-----------------------------------------------------------+-----------------+---------+-----------+--------+
Total: 895,268
SAVi(
  (initializer): CoordinateEncoderStateInit(
    (embedding_transform): MLP(
      (model): ModuleList(
        (dense_mlp_0): Linear(in_features=4, out_features=256, bias=True)
        (dense_mlp_0_act): ReLU()
        (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
      )
    )
  )
  (encoder): FrameEncoder(
    (backbone): CNN2(
      (cnn_layers): ModuleList(
        (conv_0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_0): ReLU()
        (conv_1): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_1): ReLU()
        (conv_2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_2): ReLU()
        (conv_3): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_3): ReLU()
      )
    )
    (pos_emb): PositionEmbedding(
      (pos_transform): Identity()
      (output_transform): MLP(
        (layernorm_module): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
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
  (corrector): SlotAttention(
    (gru): myGRUCell(
      (dense_ir): Linear(in_features=128, out_features=128, bias=True)
      (dense_iz): Linear(in_features=128, out_features=128, bias=True)
      (dense_in): Linear(in_features=128, out_features=128, bias=True)
      (dense_hr): Linear(in_features=128, out_features=128, bias=False)
      (dense_hz): Linear(in_features=128, out_features=128, bias=False)
      (dense_hn): Linear(in_features=128, out_features=128, bias=True)
    )
    (dense_q): Linear(in_features=128, out_features=128, bias=False)
    (dense_k): Linear(in_features=32, out_features=128, bias=False)
    (dense_v): Linear(in_features=32, out_features=128, bias=False)
    (layernorm_q): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layernorm_input): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
    (inverted_attention): InvertedDotProductAttention(
      (attn_fn): GeneralizedDotProductAttention()
    )
  )
  (decoder): SpatialBroadcastDecoder(
    (backbone): CNN2(
      (cnn_layers): ModuleList(
        (conv_0): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        (act_0): ReLU()
        (conv_1): ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        (act_1): ReLU()
        (conv_2): ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        (act_2): ReLU()
        (conv_3): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (act_3): ReLU()
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
  (predictor): TransformerBlock(
    (attn): GeneralizedDotProductAttention()
    (mlp): MLP(
      (model): ModuleList(
        (dense_mlp_0): Linear(in_features=128, out_features=256, bias=True)
        (dense_mlp_0_act): ReLU()
        (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (layernorm_query): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layernorm_mlp): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (dense_q): Linear(in_features=128, out_features=128, bias=True)
    (dense_k): Linear(in_features=128, out_features=128, bias=True)
    (dense_v): Linear(in_features=128, out_features=128, bias=True)
    (dense_o): Linear(in_features=128, out_features=128, bias=True)
  )
  (processor): Processor(
    (corrector): SlotAttention(
      (gru): myGRUCell(
        (dense_ir): Linear(in_features=128, out_features=128, bias=True)
        (dense_iz): Linear(in_features=128, out_features=128, bias=True)
        (dense_in): Linear(in_features=128, out_features=128, bias=True)
        (dense_hr): Linear(in_features=128, out_features=128, bias=False)
        (dense_hz): Linear(in_features=128, out_features=128, bias=False)
        (dense_hn): Linear(in_features=128, out_features=128, bias=True)
      )
      (dense_q): Linear(in_features=128, out_features=128, bias=False)
      (dense_k): Linear(in_features=32, out_features=128, bias=False)
      (dense_v): Linear(in_features=32, out_features=128, bias=False)
      (layernorm_q): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (layernorm_input): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
      (inverted_attention): InvertedDotProductAttention(
        (attn_fn): GeneralizedDotProductAttention()
      )
    )
    (predictor): TransformerBlock(
      (attn): GeneralizedDotProductAttention()
      (mlp): MLP(
        (model): ModuleList(
          (dense_mlp_0): Linear(in_features=128, out_features=256, bias=True)
          (dense_mlp_0_act): ReLU()
          (dense_mlp_1): Linear(in_features=256, out_features=128, bias=True)
        )
      )
      (layernorm_query): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (layernorm_mlp): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (dense_q): Linear(in_features=128, out_features=128, bias=True)
      (dense_k): Linear(in_features=128, out_features=128, bias=True)
      (dense_v): Linear(in_features=128, out_features=128, bias=True)
      (dense_o): Linear(in_features=128, out_features=128, bias=True)
    )
  )
)
```