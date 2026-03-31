================================================================================
  α = 0.05
================================================================================
  Client 0: 4 cls, 9678 samp
  Client 1: 5 cls, 12963 samp
  Client 2: 5 cls, 15652 samp
  Client 3: 7 cls, 7483 samp
  Client 4: 10 cls, 4224 samp

  Training Client 0: 4 cls
      BB 200/600 loss=2.1889
      BB 400/600 loss=2.0939
      BB 600/600 loss=2.0703
      Exp 2... done (25.7s) MSE=0.000006
      Exp 4... done (2.8s) MSE=0.000096
      Exp 5... done (196.9s) MSE=0.000001
      Exp 6... done (160.4s) MSE=0.000001

  Training Client 1: 5 cls
      BB 200/600 loss=2.0521
      BB 400/600 loss=1.9859
      BB 600/600 loss=1.9495
      Exp 4... done (374.1s) MSE=0.000001
      Exp 5... done (5.4s) MSE=0.000026
      Exp 6... done (5.3s) MSE=0.000019
      Exp 7... done (230.0s) MSE=0.000001
      Exp 8... done (370.7s) MSE=0.000000

  Training Client 2: 5 cls
      BB 200/600 loss=2.0027
      BB 400/600 loss=1.8953
      BB 600/600 loss=1.8685
      Exp 1... done (370.0s) MSE=0.000000
      Exp 2... done (322.6s) MSE=0.000001
      Exp 3... done (329.5s) MSE=0.000001
      Exp 7... done (155.1s) MSE=0.000001
      Exp 8... done (5.3s) MSE=0.000090

  Training Client 3: 7 cls
      BB 200/600 loss=2.1371
      BB 400/600 loss=2.0122
      BB 600/600 loss=1.9658
      Exp 0... done (66.8s) MSE=0.000004
      Exp 1... done (10.0s) MSE=0.000060
      Exp 2... done (5.4s) MSE=0.000098
      Exp 3... done (48.5s) MSE=0.000006
      Exp 4... done (5.1s) MSE=0.000065
      Exp 6... done (72.3s) MSE=0.000003
      Exp 9... done (376.6s) MSE=0.000001

  Training Client 4: 10 cls
      BB 200/600 loss=2.4445
      BB 400/600 loss=2.4015
      BB 600/600 loss=2.3966
      Exp 0... done (311.6s) MSE=0.000001
      Exp 1... done (5.1s) MSE=0.000000
      Exp 2... done (5.2s) MSE=0.000057
      Exp 3... done (5.2s) MSE=0.000000
      Exp 4... done (5.2s) MSE=0.000000
      Exp 5... done (5.1s) MSE=0.000000
      Exp 6... done (5.3s) MSE=0.000002
      Exp 7... done (5.2s) MSE=0.000000
      Exp 8... done (5.2s) MSE=0.000030
      Exp 9... done (5.1s) MSE=0.000000

  ── α=0.05 结果 ──
  Union (discriminative):  78.90%
  Expert-only strategies:
    S1_min: 76.88%
    S2_weighted_log: 40.85%
    S3_top_expert: 76.74%
    S4_quality_min: 76.88%
  Best expert-only:        76.88% (S1_min)
  Fused (α=0.7):       79.68%
  Δ (Fused - Union):       +0.78%
  Error correlation:       0.0753
  Oracle (U∪E):            86.01%
  Fuse utilization:        10.97%
  Train time:              7436s

================================================================================
  α = 0.1
================================================================================
  Client 0: 6 cls, 12872 samp
  Client 1: 8 cls, 14319 samp
  Client 2: 6 cls, 4978 samp
  Client 3: 9 cls, 13803 samp
  Client 4: 10 cls, 4028 samp

  Training Client 0: 6 cls
      BB 200/600 loss=2.1776
      BB 400/600 loss=2.0206
      BB 600/600 loss=1.9895
      Exp 2... done (343.3s) MSE=0.000001
      Exp 3... done (5.2s) MSE=0.000053
      Exp 4... done (15.0s) MSE=0.000029
      Exp 5... done (368.1s) MSE=0.000001
      Exp 6... done (246.4s) MSE=0.000001
      Exp 9... done (5.4s) MSE=0.000067

  Training Client 1: 8 cls
      BB 200/600 loss=2.0586
      BB 400/600 loss=1.9547
      BB 600/600 loss=1.9306
      Exp 0... done (5.5s) MSE=0.000195
      Exp 1... done (5.3s) MSE=0.000030
      Exp 2... done (10.2s) MSE=0.000062
      Exp 4... done (361.8s) MSE=0.000001
      Exp 5... done (9.9s) MSE=0.000064
      Exp 6... done (10.3s) MSE=0.000043
      Exp 7... done (358.8s) MSE=0.000000
      Exp 8... done (344.2s) MSE=0.000000

  Training Client 2: 6 cls
      BB 200/600 loss=2.3006
      BB 400/600 loss=2.2438
      BB 600/600 loss=2.2305
      Exp 1... done (342.5s) MSE=0.000001
      Exp 2... done (5.5s) MSE=0.000000
      Exp 6... done (5.5s) MSE=0.000062
      Exp 7... done (15.0s) MSE=0.000026
      Exp 8... done (30.1s) MSE=0.000011
      Exp 9... done (6.3s) MSE=0.000127

  Training Client 3: 9 cls
      BB 200/600 loss=2.0374
      BB 400/600 loss=1.8832
      BB 600/600 loss=1.8189
      Exp 0... done (117.8s) MSE=0.000002
      Exp 1... done (44.2s) MSE=0.000004
      Exp 2... done (25.2s) MSE=0.000015
      Exp 3... done (357.2s) MSE=0.000001
      Exp 4... done (10.3s) MSE=0.000036
      Exp 5... done (5.2s) MSE=0.000108
      Exp 6... done (122.4s) MSE=0.000002
      Exp 7... done (15.0s) MSE=0.000052
      Exp 9... done (377.6s) MSE=0.000001

  Training Client 4: 10 cls
      BB 200/600 loss=2.3510
      BB 400/600 loss=2.2577
      BB 600/600 loss=2.2426
      Exp 0... done (268.0s) MSE=0.000002
      Exp 1... done (5.5s) MSE=0.000000
      Exp 2... done (5.2s) MSE=0.000160
      Exp 3... done (29.2s) MSE=0.000025
      Exp 4... done (5.2s) MSE=0.000000
      Exp 5... done (5.2s) MSE=0.000119
      Exp 6... done (5.2s) MSE=0.000000
      Exp 7... done (5.1s) MSE=0.000005
      Exp 8... done (15.1s) MSE=0.000037
      Exp 9... done (5.2s) MSE=0.000000

  ── α=0.1 结果 ──
  Union (discriminative):  82.09%
  Expert-only strategies:
    S1_min: 80.07%
    S2_weighted_log: 64.10%
    S3_top_expert: 79.46%
    S4_quality_min: 80.07%
  Best expert-only:        80.07% (S1_min)
  Fused (α=0.1):       81.89%
  Δ (Fused - Union):       -0.20%
  Error correlation:       0.3202
  Oracle (U∪E):            87.68%
  Fuse utilization:        -3.58%
  Train time:              8098s