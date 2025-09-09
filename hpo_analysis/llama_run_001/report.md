# HPO Log Report

- Completed trials: 10
- Pruned trials: 5
- Unknown: 0

## Top trials (by micro_f1)
- Trial 2: micro_f1=0.1977, epochs=6, lr=0.0002290681911216664, warmup=0.043458296778794626, wd=0.06716149236701771, grad_accum=16, lora_r=8, lora_alpha=32, dropout=0.14668849102924744, train_s=8502.5, gen_s=57.15
- Trial 12: micro_f1=0.1864, epochs=4, lr=0.0001576969043762452, warmup=0.1959773245637319, wd=0.07240825195266087, grad_accum=16, lora_r=8, lora_alpha=32, dropout=0.15227218193018327, train_s=5676.6, gen_s=61.01
- Trial 14: micro_f1=0.1798, epochs=5, lr=0.00017983147236818778, warmup=0.19709998692360398, wd=0.08184691219040692, grad_accum=16, lora_r=8, lora_alpha=32, dropout=0.16773003725403068, train_s=7105.4, gen_s=57.32
- Trial 4: micro_f1=0.1717, epochs=2, lr=0.00039341380953367144, warmup=0.01012979197880335, wd=0.06679351438883922, grad_accum=32, lora_r=8, lora_alpha=32, dropout=0.16085340460154451, train_s=2835.9, gen_s=59.17
- Trial 6: micro_f1=0.1593, epochs=5, lr=7.385972386716387e-05, warmup=0.021285105320063405, wd=0.09942069697434297, grad_accum=16, lora_r=8, lora_alpha=32, dropout=0.0173982461473454, train_s=7092.9, gen_s=60.31

## Pearson correlation with micro_f1 (completed trials)
- weight_decay: 0.606
- train_seconds: 0.548
- epochs: 0.544
- lora_dropout: 0.429
- learning_rate: 0.364
- gen_seconds: 0.262
- lora_alpha: 0.035
- warmup_ratio: -0.060
- disk_mb: -0.119
- lora_r: -0.119
- grad_accum: -0.684