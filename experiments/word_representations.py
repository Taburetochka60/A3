import os, sys, shutil, importlib.metadata, json, time
from pathlib import Path


BASELINE_COMMIT = '3da9ba8421c81ae886ccb111e39bb4d4428280a1'

CODE_HASH = '5989c8c717b47ee4ee7e42d78a4fa63daff768655555b6848fce4c8006523295'

TRACKING_HASH = 'a981a7fc28153b6d800fff9453d46dd7a1615de8087ab2ab99e2b451249e2625'

ASSET_HASHES = {'part2.py': '5cfd293f5d8c31e25647905711beea0c1472bdc2d07f571f0024990276475db8', 'part2_data.py': 'c27f931f7b3bab6093c9ce78c4b7b3a2fc4f368ab4e1933e3be424bc7d505879', 'part2_split.json': 'fb93675d35dd60a8aff2f9618ec88fc48f6cf521b1607ca2dce03a29a4ff5cfd', 'pilot_ner.py': '94c2d7b4b55fcadf4948beb4b3f6dafdd7cbe345e06be715f6ac3718107fc989', 'recurrent_model.py': '33f0ddfe9d722a044df6fd3546c917046c3aac83df49826a60af0a3628047ae1', 'ner_data.py': '9b3d13c4d2afd048bbf28771cab420b1f7bd217fa96b8c9dce959703cba53a60', 'validation.py': '8d1fde6af4d8cddadd6c8ad1b69d43277579ce2e238fe3542a0a002773161123', 'metrics.py': '694c5d62ae9a30c961e3dd12bdc5e07f76e33568a9a7e2ff5237e709e35dc35a'}

RUNTIME_HASHES = {'part2.py': '5cfd293f5d8c31e25647905711beea0c1472bdc2d07f571f0024990276475db8', 'part2_data.py': '18f0100bf6d05999063ecde4d0009aba9d8765eb71228ccfcf7d024ebe8b0f69', 'part2_split.json': 'fb93675d35dd60a8aff2f9618ec88fc48f6cf521b1607ca2dce03a29a4ff5cfd', 'pilot_ner.py': '94c2d7b4b55fcadf4948beb4b3f6dafdd7cbe345e06be715f6ac3718107fc989', 'recurrent_model.py': '33f0ddfe9d722a044df6fd3546c917046c3aac83df49826a60af0a3628047ae1', 'ner_data.py': '9b3d13c4d2afd048bbf28771cab420b1f7bd217fa96b8c9dce959703cba53a60', 'validation.py': '8d1fde6af4d8cddadd6c8ad1b69d43277579ce2e238fe3542a0a002773161123', 'metrics.py': '694c5d62ae9a30c961e3dd12bdc5e07f76e33568a9a7e2ff5237e709e35dc35a'}

REFERENCE = json.loads("{\"main\": {\"name\": \"CLEAN_LSTM2_100k\", \"key\": \"f362fa37dff209c08522e35cda3430ad373c3937482a892783683cc679f8ea30\", \"config\": {\"name\": \"CLEAN_LSTM2_100k\", \"profile\": \"clean_final\", \"seed\": 42, \"task\": \"ner\", \"encoder\": {\"rnn_type\": \"lstm\", \"num_layers\": 2, \"embedding_dim\": 64, \"hidden_size\": 64, \"dropout\": 0.2, \"bidirectional\": true}, \"tokenizer\": \"word\", \"optimizer\": \"AdamW\", \"lr\": 0.002, \"weight_decay\": 0.01, \"betas\": [0.9, 0.999], \"eps\": 1e-08, \"scheduler\": \"none\", \"epochs\": 15, \"patience\": 6, \"train_rows\": 100000, \"batch_size\": 128, \"max_tokens\": 128, \"overlap\": 32, \"gradient_clip\": 1.0, \"vectors\": null, \"selection\": \"f1\", \"prediction_filter\": \"original\", \"split_sha256\": \"c0a47f5245435f1d338ce0769affde2e9d98e9ac829db3d12866074a2291afe5\", \"code_sha256\": \"b1f1ad82ee465baa67037194db546714366a5b5a5fc0df28825272ca3b269c7a\", \"train_ids_sha256\": \"802624598cf6f1f3699d30c4e165d4619c1bdb5957bc8dfde1563ba113558e28\", \"validation_ids_sha256\": \"525eb21781b4c282b0749e580fc0c79edd247be614af8d701f443d7de897b605\"}, \"architecture\": \"lstm\", \"num_layers\": 2, \"seed\": 42, \"train_rows\": 100000, \"validation_rows\": 4000, \"epochs\": 14, \"best_epoch\": 8, \"f1\": 0.7638484, \"precision\": 0.7349228611500701, \"recall\": 0.795144157814871, \"runtime_seconds\": 144.32583434700007, \"checkpoint\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CLEAN_LSTM2_100k_seed42_f362fa37df/best.pt\", \"checkpoint_sha256\": \"e52a43075018519839b0213864ff0a057c5d04b826b4f652ca72d43bc5f11bd2\", \"run_dir\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CLEAN_LSTM2_100k_seed42_f362fa37df\", \"encoding_sha256\": \"521e1a0107f60eab798a4eb9146b0bb3687466d782ef4a4fbfb96321532c5426\", \"class_weights\": [1.0, 7.221465587615967, 8.0], \"word_tag_counts\": [1759787, 33745, 8296], \"pretrained_coverage\": null, \"precision_fallbacks\": [], \"public_lb_f1\": null, \"hardware\": [{\"epoch\": 1, \"next_batch\": 0, \"name\": \"Tesla T4\", \"vram_gb\": 14.56317138671875, \"free_vram_gb\": 14.44317626953125, \"bf16_supported\": false, \"torch\": \"2.14.1+cu130\", \"cuda\": \"13.0\", \"cudnn\": 92800, \"tf32\": false, \"capability\": [7, 5]}, {\"epoch\": 1, \"next_batch\": 0, \"name\": \"NVIDIA L4\", \"vram_gb\": 22.0343017578125, \"free_vram_gb\": 21.8135986328125, \"bf16_supported\": true, \"torch\": \"2.14.1+cu130\", \"cuda\": \"13.0\", \"cudnn\": 92800, \"tf32\": false, \"capability\": [8, 9]}], \"training_amp\": \"fp16\", \"batch_size\": 128, \"evaluation\": \"Full checkpoint development holdout\"}, \"main_folder\": \"CLEAN_LSTM2_100k_seed42_f362fa37df\", \"cv\": [{\"name\": \"CV_CLEAN_LSTM2_100k_fold0\", \"key\": \"798b606b6869c391e76ff88d55242dc2d6a6284a811348132a634ce186b6f45b\", \"config\": {\"name\": \"CV_CLEAN_LSTM2_100k_fold0\", \"profile\": \"clean_final\", \"seed\": 42, \"task\": \"ner\", \"encoder\": {\"rnn_type\": \"lstm\", \"num_layers\": 2, \"embedding_dim\": 64, \"hidden_size\": 64, \"dropout\": 0.2, \"bidirectional\": true}, \"tokenizer\": \"word\", \"optimizer\": \"AdamW\", \"lr\": 0.002, \"weight_decay\": 0.01, \"betas\": [0.9, 0.999], \"eps\": 1e-08, \"scheduler\": \"replay\", \"epochs\": 8, \"patience\": 6, \"train_rows\": 66667, \"batch_size\": 128, \"max_tokens\": 128, \"overlap\": 32, \"gradient_clip\": 1.0, \"vectors\": null, \"selection\": \"fixed_final_epoch\", \"prediction_filter\": \"original\", \"split_sha256\": \"c0a47f5245435f1d338ce0769affde2e9d98e9ac829db3d12866074a2291afe5\", \"code_sha256\": \"b1f1ad82ee465baa67037194db546714366a5b5a5fc0df28825272ca3b269c7a\", \"train_ids_sha256\": \"48a5998fb7cb07664dd1d5ec8e8fee74e73b0c1a175594498031d36b3adf04ab\", \"validation_ids_sha256\": \"407740b134f6b9bb1a652ddd2d769798ac19ad40c6a85b13520ef0628613b593\", \"epoch_lrs\": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002], \"scheduler_monitor\": null, \"parent_scheduler\": \"none\", \"parent_key\": \"f362fa37dff209c08522e35cda3430ad373c3937482a892783683cc679f8ea30\"}, \"architecture\": \"lstm\", \"num_layers\": 2, \"seed\": 42, \"train_rows\": 66667, \"validation_rows\": 33333, \"epochs\": 8, \"best_epoch\": 8, \"f1\": 0.71054806, \"precision\": 0.6343073897162124, \"recall\": 0.807620069761202, \"runtime_seconds\": 85.89098327900001, \"checkpoint\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold0_seed42_798b606b68/best.pt\", \"checkpoint_sha256\": \"2b9057951186981aba8f60db69f2c1a5a6a04ae477a78ec460dfde06ec599d9f\", \"run_dir\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold0_seed42_798b606b68\", \"encoding_sha256\": \"6b9f1fbedca4c505dfc7f5faa12e07e229ae70fa29816a2d0d9035630bce3156\", \"class_weights\": [1.0, 7.228503227233887, 8.0], \"word_tag_counts\": [1174765, 22483, 5543], \"pretrained_coverage\": null, \"precision_fallbacks\": [], \"public_lb_f1\": null, \"hardware\": [{\"epoch\": 1, \"next_batch\": 0, \"name\": \"NVIDIA L4\", \"vram_gb\": 22.0343017578125, \"free_vram_gb\": 21.7237548828125, \"bf16_supported\": true, \"torch\": \"2.14.1+cu130\", \"cuda\": \"13.0\", \"cudnn\": 92800, \"tf32\": false, \"capability\": [8, 9]}], \"training_amp\": \"bf16\", \"batch_size\": 128, \"evaluation\": \"NER outer CV fold; fixed epochs\", \"folder\": \"CV_CLEAN_LSTM2_100k_fold0_seed42_798b606b68\"}, {\"name\": \"CV_CLEAN_LSTM2_100k_fold1\", \"key\": \"a4d33e4d3276efb1f3d1ace69566ba505720c596fd078d5a924c49b4b584933e\", \"config\": {\"name\": \"CV_CLEAN_LSTM2_100k_fold1\", \"profile\": \"clean_final\", \"seed\": 43, \"task\": \"ner\", \"encoder\": {\"rnn_type\": \"lstm\", \"num_layers\": 2, \"embedding_dim\": 64, \"hidden_size\": 64, \"dropout\": 0.2, \"bidirectional\": true}, \"tokenizer\": \"word\", \"optimizer\": \"AdamW\", \"lr\": 0.002, \"weight_decay\": 0.01, \"betas\": [0.9, 0.999], \"eps\": 1e-08, \"scheduler\": \"replay\", \"epochs\": 8, \"patience\": 6, \"train_rows\": 66666, \"batch_size\": 128, \"max_tokens\": 128, \"overlap\": 32, \"gradient_clip\": 1.0, \"vectors\": null, \"selection\": \"fixed_final_epoch\", \"prediction_filter\": \"original\", \"split_sha256\": \"c0a47f5245435f1d338ce0769affde2e9d98e9ac829db3d12866074a2291afe5\", \"code_sha256\": \"b1f1ad82ee465baa67037194db546714366a5b5a5fc0df28825272ca3b269c7a\", \"train_ids_sha256\": \"2fce4c59e9547785940e9e2aa6a68025ab9e6cce5188b880d5fcc4c416ce054c\", \"validation_ids_sha256\": \"9e46006bb2852874774d2feb8eb46573387e99b5ae8e2ca84fd170cbb13fdfcd\", \"epoch_lrs\": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002], \"scheduler_monitor\": null, \"parent_scheduler\": \"none\", \"parent_key\": \"f362fa37dff209c08522e35cda3430ad373c3937482a892783683cc679f8ea30\"}, \"architecture\": \"lstm\", \"num_layers\": 2, \"seed\": 43, \"train_rows\": 66666, \"validation_rows\": 33334, \"epochs\": 8, \"best_epoch\": 8, \"f1\": 0.72017267, \"precision\": 0.6467800990738747, \"recall\": 0.8123534715960324, \"runtime_seconds\": 86.33647396600009, \"checkpoint\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold1_seed43_a4d33e4d32/best.pt\", \"checkpoint_sha256\": \"ce43018c8a4a47936a9e365cbf76cb2ae7b378153eada289a6dcb61195ea235e\", \"run_dir\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold1_seed43_a4d33e4d32\", \"encoding_sha256\": \"83761eeeb84ff72b8b520f6f6cda8cc3a73b56a657563612737200f234f86279\", \"class_weights\": [1.0, 7.210364818572998, 8.0], \"word_tag_counts\": [1172672, 22556, 5458], \"pretrained_coverage\": null, \"precision_fallbacks\": [], \"public_lb_f1\": null, \"hardware\": [{\"epoch\": 1, \"next_batch\": 0, \"name\": \"NVIDIA L4\", \"vram_gb\": 22.0343017578125, \"free_vram_gb\": 21.7178955078125, \"bf16_supported\": true, \"torch\": \"2.14.1+cu130\", \"cuda\": \"13.0\", \"cudnn\": 92800, \"tf32\": false, \"capability\": [8, 9]}], \"training_amp\": \"bf16\", \"batch_size\": 128, \"evaluation\": \"NER outer CV fold; fixed epochs\", \"folder\": \"CV_CLEAN_LSTM2_100k_fold1_seed43_a4d33e4d32\"}, {\"name\": \"CV_CLEAN_LSTM2_100k_fold2\", \"key\": \"634468c6ae4f96cb3853ae4d4092c2b906d64829168ec4f7e87486daf7754dd8\", \"config\": {\"name\": \"CV_CLEAN_LSTM2_100k_fold2\", \"profile\": \"clean_final\", \"seed\": 44, \"task\": \"ner\", \"encoder\": {\"rnn_type\": \"lstm\", \"num_layers\": 2, \"embedding_dim\": 64, \"hidden_size\": 64, \"dropout\": 0.2, \"bidirectional\": true}, \"tokenizer\": \"word\", \"optimizer\": \"AdamW\", \"lr\": 0.002, \"weight_decay\": 0.01, \"betas\": [0.9, 0.999], \"eps\": 1e-08, \"scheduler\": \"replay\", \"epochs\": 8, \"patience\": 6, \"train_rows\": 66667, \"batch_size\": 128, \"max_tokens\": 128, \"overlap\": 32, \"gradient_clip\": 1.0, \"vectors\": null, \"selection\": \"fixed_final_epoch\", \"prediction_filter\": \"original\", \"split_sha256\": \"c0a47f5245435f1d338ce0769affde2e9d98e9ac829db3d12866074a2291afe5\", \"code_sha256\": \"b1f1ad82ee465baa67037194db546714366a5b5a5fc0df28825272ca3b269c7a\", \"train_ids_sha256\": \"64f7c43eb51e0974191f6596b7747d3fea711f1a177ed74c48ddbf1e2580e09e\", \"validation_ids_sha256\": \"e0d03d9f940b5a87be540f8b92fa5c2384dbcf1f05333918eaeb31df8b9e0861\", \"epoch_lrs\": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002], \"scheduler_monitor\": null, \"parent_scheduler\": \"none\", \"parent_key\": \"f362fa37dff209c08522e35cda3430ad373c3937482a892783683cc679f8ea30\"}, \"architecture\": \"lstm\", \"num_layers\": 2, \"seed\": 44, \"train_rows\": 66667, \"validation_rows\": 33333, \"epochs\": 8, \"best_epoch\": 8, \"f1\": 0.69055448, \"precision\": 0.593322632423756, \"recall\": 0.8259004379301099, \"runtime_seconds\": 84.52382769300016, \"checkpoint\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold2_seed44_634468c6ae/best.pt\", \"checkpoint_sha256\": \"073e895227cbc20824b03900948d96f124d821003a43a745f4c82ec9077af031\", \"run_dir\": \"/content/drive/MyDrive/KSE_AI301_A3/results/clean_final/runs/CV_CLEAN_LSTM2_100k_fold2_seed44_634468c6ae\", \"encoding_sha256\": \"cb0e33bb8eaf5e67a1417d7530368861f1266a3a8a7461d3dbab4222fc7bd5c7\", \"class_weights\": [1.0, 7.225557327270508, 8.0], \"word_tag_counts\": [1172137, 22451, 5591], \"pretrained_coverage\": null, \"precision_fallbacks\": [], \"public_lb_f1\": null, \"hardware\": [{\"epoch\": 1, \"next_batch\": 0, \"name\": \"NVIDIA L4\", \"vram_gb\": 22.0343017578125, \"free_vram_gb\": 21.7178955078125, \"bf16_supported\": true, \"torch\": \"2.14.1+cu130\", \"cuda\": \"13.0\", \"cudnn\": 92800, \"tf32\": false, \"capability\": [8, 9]}], \"training_amp\": \"bf16\", \"batch_size\": 128, \"evaluation\": \"NER outer CV fold; fixed epochs\", \"folder\": \"CV_CLEAN_LSTM2_100k_fold2_seed44_634468c6ae\"}], \"main_history\": [{\"epoch\": \"1\", \"train_loss\": \"0.24933555665546595\", \"mean_gradient_norm\": \"0.18618067253924087\", \"max_gradient_norm\": \"0.8813950419425964\", \"train_seconds\": \"10.806788247999293\", \"validation_seconds\": \"0.6737825479999628\", \"epoch_seconds\": \"11.480570795999256\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10216283798217773\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.57984679\", \"precision\": \"0.47398843930635837\", \"recall\": \"0.7465857359635811\", \"val_loss\": \"0.07250768833623716\", \"train_monitor_loss\": \"0.0620396073621525\", \"train_monitor_f1\": \"0.64651163\", \"exact_span_f1\": \"0.57543041\", \"tp\": \"984\", \"fp\": \"1092\", \"fn\": \"334\", \"predicted_entity_surfaces\": \"2076\", \"embedding_norm\": \"1369.3759765625\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"2\", \"train_loss\": \"0.12829856807019613\", \"mean_gradient_norm\": \"0.15136348372301484\", \"max_gradient_norm\": \"0.37006616592407227\", \"train_seconds\": \"9.4148751239984\", \"validation_seconds\": \"0.5061443820000022\", \"epoch_seconds\": \"9.921019505998402\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.1017160415649414\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.63231282\", \"precision\": \"0.5265017667844523\", \"recall\": \"0.791350531107739\", \"val_loss\": \"0.05760714052319674\", \"train_monitor_loss\": \"0.04581125268353683\", \"train_monitor_f1\": \"0.71090047\", \"exact_span_f1\": \"0.62699128\", \"tp\": \"1043\", \"fp\": \"938\", \"fn\": \"275\", \"predicted_entity_surfaces\": \"1981\", \"embedding_norm\": \"1353.94384765625\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"3\", \"train_loss\": \"0.09803097005173542\", \"mean_gradient_norm\": \"0.13134222096094236\", \"max_gradient_norm\": \"0.2890947163105011\", \"train_seconds\": \"8.961933939998403\", \"validation_seconds\": \"0.5021977279999419\", \"epoch_seconds\": \"9.464131667998345\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.1024169921875\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.70909091\", \"precision\": \"0.637409200968523\", \"recall\": \"0.798937784522003\", \"val_loss\": \"0.040970905038722534\", \"train_monitor_loss\": \"0.027101636909482345\", \"train_monitor_f1\": \"0.80407125\", \"exact_span_f1\": \"0.7027027\", \"tp\": \"1053\", \"fp\": \"599\", \"fn\": \"265\", \"predicted_entity_surfaces\": \"1652\", \"embedding_norm\": \"1338.387939453125\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"4\", \"train_loss\": \"0.08165321432493132\", \"mean_gradient_norm\": \"0.1197871530423765\", \"max_gradient_norm\": \"0.3518797755241394\", \"train_seconds\": \"9.121438842001567\", \"validation_seconds\": \"0.512815823999972\", \"epoch_seconds\": \"9.634254666001539\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10202503204345703\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.65942685\", \"precision\": \"0.5473209814722083\", \"recall\": \"0.8292867981790591\", \"val_loss\": \"0.04717350854657036\", \"train_monitor_loss\": \"0.03026075067342715\", \"train_monitor_f1\": \"0.77434679\", \"exact_span_f1\": \"0.65390368\", \"tp\": \"1093\", \"fp\": \"904\", \"fn\": \"225\", \"predicted_entity_surfaces\": \"1997\", \"embedding_norm\": \"1322.958740234375\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"5\", \"train_loss\": \"0.06977256510053978\", \"mean_gradient_norm\": \"0.11367654757540854\", \"max_gradient_norm\": \"0.33308371901512146\", \"train_seconds\": \"9.115108625999596\", \"validation_seconds\": \"0.5033511020000105\", \"epoch_seconds\": \"9.618459727999607\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10184764862060547\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.68177585\", \"precision\": \"0.5769837099316868\", \"recall\": \"0.8330804248861912\", \"val_loss\": \"0.04362103024373334\", \"train_monitor_loss\": \"0.023847115440686042\", \"train_monitor_f1\": \"0.80392157\", \"exact_span_f1\": \"0.6773399\", \"tp\": \"1098\", \"fp\": \"805\", \"fn\": \"220\", \"predicted_entity_surfaces\": \"1903\", \"embedding_norm\": \"1307.815185546875\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"6\", \"train_loss\": \"0.0619268931048376\", \"mean_gradient_norm\": \"0.10790201468045449\", \"max_gradient_norm\": \"0.43549254536628723\", \"train_seconds\": \"9.030682706001926\", \"validation_seconds\": \"0.5018362550000575\", \"epoch_seconds\": \"9.532518961001983\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10201263427734375\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.73184165\", \"precision\": \"0.6698172652804033\", \"recall\": \"0.8065250379362671\", \"val_loss\": \"0.03775216355468046\", \"train_monitor_loss\": \"0.017417366772828045\", \"train_monitor_f1\": \"0.85488127\", \"exact_span_f1\": \"0.72677596\", \"tp\": \"1063\", \"fp\": \"524\", \"fn\": \"255\", \"predicted_entity_surfaces\": \"1587\", \"embedding_norm\": \"1292.68408203125\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"7\", \"train_loss\": \"0.05428174594882665\", \"mean_gradient_norm\": \"0.10131575748600695\", \"max_gradient_norm\": \"0.5169426202774048\", \"train_seconds\": \"9.036116386998401\", \"validation_seconds\": \"0.5146463550000817\", \"epoch_seconds\": \"9.550762741998483\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10197877883911133\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.73511294\", \"precision\": \"0.669576059850374\", \"recall\": \"0.8148710166919575\", \"val_loss\": \"0.037505464340418834\", \"train_monitor_loss\": \"0.015354508174624245\", \"train_monitor_f1\": \"0.87301587\", \"exact_span_f1\": \"0.72930801\", \"tp\": \"1074\", \"fp\": \"530\", \"fn\": \"244\", \"predicted_entity_surfaces\": \"1604\", \"embedding_norm\": \"1277.501708984375\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"8\", \"train_loss\": \"0.049186068583556605\", \"mean_gradient_norm\": \"0.09766826772219156\", \"max_gradient_norm\": \"0.36012813448905945\", \"train_seconds\": \"9.076973076001309\", \"validation_seconds\": \"0.5075169689999939\", \"epoch_seconds\": \"9.584490045001303\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10226631164550781\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.7638484\", \"precision\": \"0.7349228611500701\", \"recall\": \"0.795144157814871\", \"val_loss\": \"0.03827367520608539\", \"train_monitor_loss\": \"0.010984015579564285\", \"train_monitor_f1\": \"0.90607735\", \"exact_span_f1\": \"0.76111312\", \"tp\": \"1048\", \"fp\": \"378\", \"fn\": \"270\", \"predicted_entity_surfaces\": \"1426\", \"embedding_norm\": \"1262.6038818359375\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"9\", \"train_loss\": \"0.04451690755459275\", \"mean_gradient_norm\": \"0.09633684043755845\", \"max_gradient_norm\": \"0.4670160710811615\", \"train_seconds\": \"9.129776558001254\", \"validation_seconds\": \"0.5019239999999172\", \"epoch_seconds\": \"9.631700558001171\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10183382034301758\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.74120083\", \"precision\": \"0.6797468354430379\", \"recall\": \"0.8148710166919575\", \"val_loss\": \"0.0394676305648389\", \"train_monitor_loss\": \"0.013326212741457646\", \"train_monitor_f1\": \"0.8847185\", \"exact_span_f1\": \"0.7357974\", \"tp\": \"1074\", \"fp\": \"506\", \"fn\": \"244\", \"predicted_entity_surfaces\": \"1580\", \"embedding_norm\": \"1247.7369384765625\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"10\", \"train_loss\": \"0.04137062263270169\", \"mean_gradient_norm\": \"0.09422284130321439\", \"max_gradient_norm\": \"0.7186269164085388\", \"train_seconds\": \"9.19547200300326\", \"validation_seconds\": \"0.5023494050000181\", \"epoch_seconds\": \"9.697821408003279\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10204935073852539\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.73597246\", \"precision\": \"0.6735979836168872\", \"recall\": \"0.8110773899848255\", \"val_loss\": \"0.04082454244231625\", \"train_monitor_loss\": \"0.01241676661529409\", \"train_monitor_f1\": \"0.90322581\", \"exact_span_f1\": \"0.73019126\", \"tp\": \"1069\", \"fp\": \"518\", \"fn\": \"249\", \"predicted_entity_surfaces\": \"1587\", \"embedding_norm\": \"1233.2791748046875\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"11\", \"train_loss\": \"0.037278773552395485\", \"mean_gradient_norm\": \"0.09105359783272265\", \"max_gradient_norm\": \"0.41602012515068054\", \"train_seconds\": \"9.327544339999577\", \"validation_seconds\": \"0.5024715799999058\", \"epoch_seconds\": \"9.830015919999482\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10185766220092773\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.7311973\", \"precision\": \"0.6581663630843959\", \"recall\": \"0.8224582701062215\", \"val_loss\": \"0.046950446237842176\", \"train_monitor_loss\": \"0.013109449387107466\", \"train_monitor_f1\": \"0.86528497\", \"exact_span_f1\": \"0.7270903\", \"tp\": \"1084\", \"fp\": \"563\", \"fn\": \"234\", \"predicted_entity_surfaces\": \"1647\", \"embedding_norm\": \"1218.7802734375\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"12\", \"train_loss\": \"0.03434542696220649\", \"mean_gradient_norm\": \"0.08456066463742871\", \"max_gradient_norm\": \"0.41140198707580566\", \"train_seconds\": \"9.22691062099932\", \"validation_seconds\": \"0.507770757000003\", \"epoch_seconds\": \"9.734681377999323\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.1018819808959961\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.75607181\", \"precision\": \"0.7051871306631649\", \"recall\": \"0.8148710166919575\", \"val_loss\": \"0.04114515135227344\", \"train_monitor_loss\": \"0.008206960162069166\", \"train_monitor_f1\": \"0.93854749\", \"exact_span_f1\": \"0.75183246\", \"tp\": \"1074\", \"fp\": \"449\", \"fn\": \"244\", \"predicted_entity_surfaces\": \"1523\", \"embedding_norm\": \"1204.2626953125\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"13\", \"train_loss\": \"0.03201559344755621\", \"mean_gradient_norm\": \"0.08339002359506038\", \"max_gradient_norm\": \"0.6362873315811157\", \"train_seconds\": \"8.879508421998707\", \"validation_seconds\": \"0.5119033020000643\", \"epoch_seconds\": \"9.391411723998772\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10196065902709961\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.7432385\", \"precision\": \"0.6919555264879006\", \"recall\": \"0.802731411229135\", \"val_loss\": \"0.0428867867942076\", \"train_monitor_loss\": \"0.009463690810660445\", \"train_monitor_f1\": \"0.9260274\", \"exact_span_f1\": \"0.73797909\", \"tp\": \"1058\", \"fp\": \"471\", \"fn\": \"260\", \"predicted_entity_surfaces\": \"1529\", \"embedding_norm\": \"1189.96044921875\", \"next_learning_rate\": \"0.002\"}, {\"epoch\": \"14\", \"train_loss\": \"0.029904829142254806\", \"mean_gradient_norm\": \"0.08264069163533466\", \"max_gradient_norm\": \"0.5907886624336243\", \"train_seconds\": \"8.969640269999331\", \"validation_seconds\": \"0.5083278189999874\", \"epoch_seconds\": \"9.477968088999319\", \"learning_rate\": \"0.002\", \"gpu\": \"NVIDIA L4\", \"torch_version\": \"2.14.1+cu130\", \"peak_gpu_gb\": \"0.10221624374389648\", \"amp\": \"fp16\", \"batch_size\": \"128\", \"f1\": \"0.75070822\", \"precision\": \"0.703851261620186\", \"recall\": \"0.8042488619119879\", \"val_loss\": \"0.04327470922721567\", \"train_monitor_loss\": \"0.00790801539286066\", \"train_monitor_f1\": \"0.92307692\", \"exact_span_f1\": \"0.74719101\", \"tp\": \"1060\", \"fp\": \"446\", \"fn\": \"258\", \"predicted_entity_surfaces\": \"1506\", \"embedding_norm\": \"1175.83447265625\", \"next_learning_rate\": \"0.002\"}]}")

WB_PROJECT = 'kse-ai301-a3'
WB_ENTITY = None
WB_MODE = 'offline'

def tracking_rng(function):
    def wrapped(*args,**kwargs):
        rng=base.rng_state()
        try:return function(*args,**kwargs)
        finally:base.restore_rng(rng)
    return wrapped


# W&B is optional. CSV/checkpoints remain the source of truth.
def tracking_rows(folder):
    path=Path(folder)/'history.csv'
    return pd.read_csv(path).to_dict('records') if path.exists() and path.stat().st_size>2 else []


def tracking_metrics(row):
    mapping=dict(train_loss='train/loss',val_loss='validation/loss',f1='validation/f1_raw',
        precision='validation/precision_raw',recall='validation/recall_raw',strict_f1='validation/f1_strict',
        loc_oov_recall='locations/oov_recall_strict',loc_iv_recall='locations/iv_recall_strict',
        learning_rate='optimizer/learning_rate',gradient_norm='gradients/norm_mean',
        max_gradient_norm='gradients/norm_max',epoch_seconds='time/epoch_seconds',peak_gpu_gb='gpu/peak_gb')
    values={'epoch':int(row['epoch'])}
    for key,name in mapping.items():
        value=row.get(key)
        if value is not None and isinstance(value,(float,int)) and np.isfinite(value):values[name]=float(value)
    if 'gradient_norm' not in row and isinstance(row.get('mean_gradient_norm'),(float,int)):
        values['gradients/norm_mean']=float(row['mean_gradient_norm'])
    flow=row.get('gradient_flow')
    if isinstance(flow,str):
        values.update({'gradient_flow/'+name:float(value) for name,value in json.loads(flow).items()})
    return values


@tracking_rng
def tracking_start(folder, config):
    folder=Path(folder);run_id=source.fingerprint(dict(run=config['identity'],project=WB_PROJECT))[:16]
    previous=json.loads((folder/'wandb_index.json').read_text()) if (folder/'wandb_index.json').exists() else {}
    try:
        import wandb
        local=WORK/'wandb'/run_id;local.mkdir(parents=True,exist_ok=True)
        mode=WB_MODE
        args=dict(project=WB_PROJECT,entity=WB_ENTITY or None,id=run_id,name=config.get('name',folder.parent.name+'-'+folder.name),
            group=CODE_HASH[:10]+'-'+config.get('kind',folder.parent.name),job_type=config.get('job_type','cv' if config.get('fold') is not None else 'main'),allow_val_change=True,
            config=dict(config,representation_policy={'bpe':dict(pooling='mean',vocab_limit=30000,min_frequency=2,key='NFC/casefold'),
                'ngrams':dict(pooling='mean_add',ngrams=[3,4,5],buckets=POLICY['buckets'],word_limit=30000,key='NFC/casefold')}.get(config.get('kind'),{}),training_code_hash=CODE_HASH,tracking_code_hash=TRACKING_HASH,threshold=.8,prediction_filter='strict'),
            dir=str(local),save_code=False,settings=wandb.Settings(init_timeout=12,finish_timeout=10,console='off',disable_git=True))
        try:
            run=wandb.init(**args,mode=mode,**({'resume':'allow'} if mode=='online' else {}))
        except Exception:
            wandb.finish(exit_code=1)
            mode='offline';run=wandb.init(**args,mode='offline')
        run.define_metric('epoch')
        run.define_metric('*',step_metric='epoch')
        tracker=dict(run=run,folder=folder,id=run_id,mode=mode,online_before=previous.get('online_epoch',0),last_epoch=previous.get('online_epoch',0) if mode=='online' else 0)
        for row in tracking_rows(folder):tracking_log(tracker,row)
        return tracker
    except Exception as error:
        print('W&B unavailable; local logging continues:',type(error).__name__)
        return None


@tracking_rng
def tracking_log(tracker,row):
    if tracker is None or tracker.get('failed'):return
    try:
        epoch=int(row['epoch'])
        if epoch>tracker['last_epoch']:
            tracker['run'].log(tracking_metrics(row));tracker['last_epoch']=epoch
    except Exception as error:
        tracker['failed']=True
        print('W&B log skipped; epoch remains in CSV:',type(error).__name__)


@tracking_rng
def tracking_finish(tracker, summary=None, digest=None):
    if tracker is None:return
    try:
        run=tracker['run'];folder=tracker['folder']
        if summary:run.summary.update(summary)
        media=folder.parent if folder.name=='main' else folder
        for name in ['training_curves.png','gradient_flow.png']:
            path=media/name
            if path.exists():
                import wandb
                run.log({'plots/'+name.removesuffix('.png'):wandb.Image(str(path))})
        run.finish()
        import shutil
        run_dir=Path(run.dir).parent
        target=folder/'wandb'/run_dir.name
        shutil.copytree(run_dir,target,dirs_exist_ok=True)
        write_json(folder/'wandb_index.json',dict(id=tracker['id'],mode=tracker['mode'],directory=str(target),
            online_epoch=tracker['last_epoch'] if tracker['mode']=='online' else tracker.get('online_before',0),
            export_digest=digest if not tracker.get('failed') else None,
            note='Offline resume is a full history snapshot. Sync only the latest directory for this run ID.'))
    except Exception as error:
        print('W&B finish/export skipped; local results are safe:',type(error).__name__)


def tracking_saved(folder, result):
    folder=Path(folder)
    config_path=folder/'run_config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else dict(identity=result['identity'],name=folder.name)
    summary=dict(complete=result.get('complete',False),epochs=result.get('epochs'),best_epoch=result.get('best_epoch'),
                 training_seconds=result.get('seconds'),parameters=result.get('parameters'))
    for label,path in [('checkpoint',folder.parent/'checkpoint/metrics.json'),('calibration',folder.parent/'calibration/metrics.json'),
                       ('heldout',folder/'evaluation/metrics.json')]:
        if path.exists():
            data=json.loads(path.read_text())
            summary.update({label+'/'+k:v for k,v in data.items() if isinstance(v,(float,int))})
    public=folder.parent/'kaggle_public.json'
    if public.exists():summary['kaggle/public_f1']=json.loads(public.read_text())['score']
    media=folder.parent if folder.name=='main' else folder
    images={p.name:source.file_hash(p) for p in [media/'training_curves.png',media/'gradient_flow.png'] if p.exists()}
    digest=source.fingerprint(dict(project=WB_PROJECT,entity=WB_ENTITY,mode=WB_MODE,tracking_version=TRACKING_HASH,config=config,history=tracking_rows(folder),summary=summary,images=images))
    index=folder/'wandb_index.json'
    if index.exists() and json.loads(index.read_text()).get('export_digest')==digest:return
    tracker=tracking_start(folder,config)
    tracking_finish(tracker,{k:v for k,v in summary.items() if v is not None},digest)


def tracking_cv(kind, result):
    folder=OUT/kind/'cv_tracking';folder.mkdir(parents=True,exist_ok=True)
    config=dict(identity=source.fingerprint(dict(code=CODE_HASH,kind=kind,split=SPEC['content_sha256'],stage='cv-summary')),
                name=kind+'-cv-summary',kind=kind,fold=None,job_type='cv-summary',folds=3,cv_epochs=8)
    summary={prefix+'/'+k:v for prefix,data in [('baseline',result['baseline']),('candidate',result['candidate']),('paired_interval',result['interval'])]
             for k,v in data.items() if isinstance(v,(float,int))}
    public=OUT/kind/'kaggle_public.json'
    if public.exists():summary['kaggle/public_f1']=json.loads(public.read_text())['score']
    tracker=tracking_start(folder,config)
    if tracker is not None:
        try:
            import wandb
            rows=[]
            for fold in range(3):
                r=json.loads((OUT/kind/f'cv{fold}/result.json').read_text())
                m=json.loads((OUT/kind/f'cv{fold}/evaluation/metrics.json').read_text())
                rows.append([fold,r['f1'],m['f1'],m['oov_recall'],m['iv_recall'],r['seconds']])
            tracker['run'].log({'cv/folds':wandb.Table(columns=['fold','raw_f1','strict_f1','oov_recall','iv_recall','seconds'],data=rows)})
        except Exception:pass
    tracking_finish(tracker,summary)
    write_json(folder/'summary.json',summary)


def record_kaggle_public_f1(model_name, measured_score):
    if model_name not in ['bpe','ngrams'] or not 0<=float(measured_score)<=1:
        raise ValueError('Use an actual score for bpe or ngrams')
    folder=OUT/model_name/'main';result=json.loads((folder/'result.json').read_text())
    write_json(OUT/model_name/'kaggle_public.json',dict(score=float(measured_score),checkpoint_sha256=result['checkpoint_sha256'],source='manual Kaggle result'))
    tracking_saved(folder,result)
    cv=OUT/model_name/'cv_comparison.json'
    if cv.exists():tracking_cv(model_name,json.loads(cv.read_text()))


def tracking_baseline(summaries, parameters):
    config=dict(identity=SPEC['reference']['checkpoint_sha256'],name='baseline-reference',kind='baseline',
        job_type='reference',reference_only=True,parameters=parameters,seed=42,encoder=base.CONFIG['encoder'],
        optimizer='AdamW',lr=.002,weight_decay=.01,batch_size=128,split_sha256=SPEC['content_sha256'])
    tracker=tracking_start(OUT/'baseline',config)
    summary={partition+'/'+k:v for partition,record in summaries.items() for k,v in record.items() if isinstance(v,(float,int))}
    summary.update(reference_only=True,training_seconds=REFERENCE['main']['runtime_seconds'])
    reported=SPEC['reference'].get('public_f1')
    if reported is not None:summary['kaggle/public_f1_user_reported']=reported
    tracking_finish(tracker,summary)

import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers

import part2 as base
import part2_data as source
from metrics import hard_processing, score_details
from ner_data import token_key, decode_bio, collate_ner, training_windows, window_ranges, merge_window_probabilities, IGNORE
from pilot_ner import vocabulary, class_weights, seed_all, documents
from recurrent_model import NERModel

BASE_PARAMETERS = None

POLICY = dict(threshold=.8, filter='strict', cv_epochs=8, batch_size=128, max_epochs=15,
              patience=6, bootstrap=2000, cv_gain=.01, oov_gain=.05, iv_tolerance=.01,
              precision_tolerance=.01, calibration_tolerance=.005, buckets=16384)


class BudgetEnd(Exception):
    pass


def time_left():
    return max(0., DEADLINE-time.monotonic())


def check_time():
    if time_left() < 30:
        raise BudgetEnd('Session budget reached; resume with Run All')


def write_json(path, value):
    source.save_json(path, value)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)


def save_weights(path, value):
    path = Path(path)
    local = WORK/'checkpoint.tmp'
    torch.save(value, local)
    base.copy_file(local, path)


def train_guard(train, valid, fold):
    ids = [d['row_id'] for d in train]
    vi = [d['row_id'] for d in valid]
    assert len(set(ids)) == len(ids) and len(set(vi)) == len(vi)
    assert set(ids).isdisjoint(vi)
    assert {d['group'] for d in train}.isdisjoint(d['group'] for d in valid)
    if fold is None:
        assert ids == SPEC['fit'] and vi == SPEC['checkpoint']
    else:
        assert ids == [i for i,f in zip(SPEC['fit'],SPEC['cv_folds']) if f != fold]
        assert vi == [i for i,f in zip(SPEC['fit'],SPEC['cv_folds']) if f == fold]
    assert all(d['bio'] is not None and len(d['tokens']) for d in train)


def fit_representation(train, kind, folder):
    train_ids = source.fingerprint([d['row_id'] for d in train])
    path = folder/'representation.json'
    if path.exists():
        state = json.loads(path.read_text())
        assert state['train_ids'] == train_ids and state['kind'] == kind
        return state
    words = vocabulary(train)
    state = dict(kind=kind, vocabulary=words, size=len(words)+2, train_ids=train_ids,
                 buckets=POLICY['buckets'])
    if kind == 'bpe':
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2,
                    special_tokens=['[PAD]','[UNK]'], show_progress=False)
        tokenizer.train_from_iterator((token_key(t.value) for d in train for t in d['tokens']), trainer=trainer)
        state.update(tokenizer=tokenizer.to_str(), pieces=tokenizer.get_vocab_size())
    else:
        assert kind == 'ngrams'
        state['pieces'] = POLICY['buckets']
    write_json(path,state)
    return state


def ngrams(word, buckets):
    word = '<'+word+'>'
    return [1+int.from_bytes(hashlib.sha256(word[i:i+n].encode()).digest()[:8],'big')%(buckets-1)
            for n in range(3,6) for i in range(len(word)-n+1)] or [1]


class WordFeatures(nn.Module):
    def __init__(self, state, word_embedding):
        super().__init__()
        self.kind = state['kind']
        self.word = word_embedding if self.kind == 'ngrams' else None
        self.subword = nn.EmbeddingBag(state['pieces'], 64, mode='mean', padding_idx=0)
        for name in ['word_ids','begin','counts','flat']:
            self.register_buffer(name, torch.zeros(1,dtype=torch.long), persistent=False)

    def set_table(self, table):
        device = self.subword.weight.device
        for name, value in table.items():
            setattr(self,name,torch.tensor(value,dtype=torch.long,device=device))

    def forward(self, ids):
        unique, inverse = torch.unique(ids.reshape(-1),return_inverse=True)
        counts = self.counts[unique]
        offsets = torch.cumsum(counts,0)-counts
        positions = (torch.repeat_interleave(self.begin[unique]-offsets,counts)
                     + torch.arange(int(counts.sum()),device=ids.device))
        values = self.subword(self.flat[positions],offsets)
        if self.word is not None:
            values = values+self.word(self.word_ids[unique])
        return values[inverse].reshape(*ids.shape,64)


def feature_documents(docs, state, model):
    # Routing IDs do not create trainable word embeddings for heldout words.
    keys = {}; word_ids=[0]; begin=[0]; counts=[1]; flat=[0]
    tokenizer = Tokenizer.from_str(state['tokenizer']) if state['kind']=='bpe' else None
    encoded=[]
    for d in docs:
        ids=[]
        for token in d['tokens']:
            key=token_key(token.value)
            if key not in keys:
                keys[key]=len(word_ids)
                pieces=(tokenizer.encode(key,add_special_tokens=False).ids if tokenizer else ngrams(key,state['buckets'])) or [1]
                word_ids.append(state['vocabulary'].get(key,1))
                begin.append(len(flat));counts.append(len(pieces));flat.extend(pieces)
            ids.append(keys[key])
        encoded.append(dict(d,ids=ids))
    model.encoder.embedding.set_table(dict(word_ids=word_ids,begin=begin,counts=counts,flat=flat))
    return encoded


def new_model(state):
    model=NERModel(state['size'],**base.CONFIG['encoder'])
    model.encoder.embedding=WordFeatures(state,model.encoder.embedding)
    return model


def load_candidate(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['code_hash']==CODE_HASH and saved['split']==SPEC['content_sha256']
    model=new_model(saved['representation']);model.load_state_dict(saved['model']);model.cuda().eval()
    return model,saved


def probabilities(model, docs):
    rows=[]; metadata=[]; windows=[]
    for i,d in enumerate(docs):
        windows.append(list(window_ranges(len(d['ids']))))
        for a,b in windows[-1]:
            rows.append((d['ids'][a:b],[IGNORE]*(b-a))); metadata.append((i,a,b))
    parts=[[] for d in docs];offset=0;model.eval()
    with torch.inference_mode():
        for ids,lengths,_ in DataLoader(rows,batch_size=128,collate_fn=collate_ner,pin_memory=True):
            check_time()
            p=model(ids.cuda(non_blocking=True),lengths).float().softmax(-1).cpu().numpy()
            for row,n in zip(p,lengths.tolist()):
                i,a,b=metadata[offset];offset+=1;assert n==b-a;parts[i].append(row[:n])
    result=[merge_window_probabilities(len(d['tokens']),ws,pp) for d,ws,pp in zip(docs,windows,parts)]
    assert all(p.shape==(len(d['tokens']),3) and np.isfinite(p).all() for d,p in zip(docs,result))
    return result


def cached_probabilities(path, model, docs, state, model_hash, baseline=False):
    path=Path(path)
    identity=source.fingerprint(dict(model=model_hash,rows=[d['row_id'] for d in docs],
        documents=source.documents_hash(docs),split=SPEC['content_sha256'],code=CODE_HASH))
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text());assert saved['identity']==identity
        assert source.file_hash(path)==saved['sha256']
        with np.load(path) as values:
            lengths=values['lengths'];flat=values['values']
        assert lengths.tolist()==[len(d['tokens']) for d in docs]
        return list(np.split(flat,np.cumsum(lengths)[:-1]))
    encoded=(source.encode_documents(docs,state) if baseline else feature_documents(docs,state,model))
    p=probabilities(model,encoded)
    temp=WORK/'probabilities.npz';np.savez_compressed(temp,values=np.concatenate(p),lengths=np.array([len(x) for x in p]))
    base.copy_file(temp,path);write_json(meta,dict(identity=identity,sha256=source.file_hash(path)))
    return p


def fit_run(kind, train, valid, folder, fold=None):
    base.gpu(); train_guard(train,valid,fold)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=source.fingerprint(dict(code=CODE_HASH,split=SPEC['content_sha256'],policy=POLICY,kind=kind,fold=fold,
        train=[d['row_id'] for d in train],valid=[d['row_id'] for d in valid]))
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        assert result['identity']==identity and result['complete']
        assert source.file_hash(folder/'best.pt')==result['checkpoint_sha256']
        assert source.fingerprint(json.loads((folder/'representation.json').read_text()))==result['representation_sha256']
        tracking_saved(folder,result)
        return result
    check_time();seed_all(42 if fold is None else 42+fold)
    state=fit_representation(train,kind,folder)
    if fold is None:
        assert source.fingerprint(state['vocabulary'])==SPEC['reference']['vocabulary_sha256']
    model=new_model(state).cuda()
    params=sum(p.numel() for p in model.parameters())
    write_json(folder/'run_config.json',dict(identity=identity,parameters=params,kind=kind,fold=fold,
        seed=42 if fold is None else 42+fold,train_rows=len(train),validation_rows=len(valid),encoder=base.CONFIG['encoder'],
        optimizer='AdamW',lr=.002,weight_decay=.01,batch_size=128,epochs=15 if fold is None else 8,patience=6,
        split_sha256=SPEC['content_sha256'],representation_sha256=source.fingerprint(state)))
    encoded=feature_documents(train+valid,state,model);tr,va=encoded[:len(train)],encoded[len(train):]
    examples=[row for d in tr for row in training_windows(d['ids'],d['bio'])]
    weights,counts=class_weights(train,torch.device('cuda'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    # Match actual historical main FP16 and historical CV BF16, not a different automatic AMP policy.
    mode='fp16' if fold is None else 'bf16'
    if mode=='bf16' and not HARDWARE['bf16']:
        del model;torch.cuda.empty_cache();raise RuntimeError('Historical CV used native BF16; this GPU cannot match its protocol')
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.,enabled=mode=='fp16')
    progress=dict(epoch=1,batch=0,best=-1.,bad=0,history=[],totals=[0.,0.,0.,0.,0,0.],seconds=0.,fallbacks=[])
    latest=folder/'latest.pt'
    if latest.exists():
        local=WORK/'resume.pt';base.copy_file(latest,local)
        saved=torch.load(local,map_location='cpu',weights_only=True)
        assert saved['identity']==identity and saved['representation']==state
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        progress=saved['progress'];mode=saved['mode']
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.,enabled=mode=='fp16')
        scaler.load_state_dict(saved['scaler']);base.restore_rng(saved['rng'])
    write_csv(folder/'history.csv',progress['history'])
    tracker=tracking_start(folder,json.loads((folder/'run_config.json').read_text()))
    segment=last_save=time.monotonic()

    def persist():
        nonlocal segment,last_save
        progress['seconds']+=time.monotonic()-segment;segment=time.monotonic()
        save_weights(latest,dict(identity=identity,representation=state,model=model.state_dict(),optimizer=optimizer.state_dict(),
            progress=progress,mode=mode,scaler=scaler.state_dict(),rng=base.rng_state()))
        write_csv(folder/'history.csv',progress['history']);last_save=time.monotonic()

    limit=15 if fold is None else 8
    try:
        for epoch in range(progress['epoch'],limit+1):
            if fold is None and progress['bad']>=6:break
            model.train();torch.cuda.reset_peak_memory_stats()
            batch_ids=base.batches(examples,128,42 if fold is None else 42+fold,epoch)
            loader=DataLoader(examples,batch_sampler=batch_ids[progress['batch']:],collate_fn=collate_ner,
                pin_memory=True,num_workers=2,generator=torch.Generator().manual_seed((42 if fold is None else 42+fold)+epoch))
            for number,batch in enumerate(loader,start=progress['batch']):
                check_time();rng=base.rng_state();tick=time.monotonic()
                try:
                    numerator,denominator,norm=base.update(model,batch,optimizer,scaler,weights,mode)
                except (FloatingPointError,RuntimeError) as error:
                    known=isinstance(error,FloatingPointError) or any(s in str(error).lower() for s in ['bfloat16','not implemented for','cudnn_status_not_supported','cudnn_status_bad_param'])
                    if mode=='fp32' or not known:raise
                    base.restore_rng(rng);optimizer.zero_grad(set_to_none=True)
                    progress['fallbacks'].append(dict(epoch=epoch,batch=number,previous=mode,error=str(error)))
                    mode='fp32';scaler=torch.amp.GradScaler('cuda',enabled=False)
                    numerator,denominator,norm=base.update(model,batch,optimizer,scaler,weights,mode)
                totals=progress['totals'];totals[0]+=numerator;totals[1]+=denominator;totals[2]+=norm
                totals[3]=max(totals[3],norm);totals[4]+=1;totals[5]+=time.monotonic()-tick
                progress['batch']=number+1
                if time.monotonic()-last_save>120:persist()
            check_time();tick=time.monotonic()
            validation_p=probabilities(model,va)
            metric=base.score(va,validation_p)
            loc_counts,_=statistics(valid,validation_p)
            loc_metric=metrics_from_counts(loc_counts)
            flow={name:float(p.grad.detach().norm()) for name,p in model.named_parameters() if p.grad is not None}
            totals=progress['totals']
            record=dict(epoch=epoch,train_loss=totals[0]/totals[1],gradient_norm=totals[2]/totals[4],
                max_gradient_norm=totals[3],train_seconds=totals[5],validation_seconds=time.monotonic()-tick,
                learning_rate=.002,amp=mode,gpu=HARDWARE['name'],torch_version=str(torch.__version__),batch_size=128,peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30,**metric)
            record.update(strict_f1=loc_metric['f1'],loc_oov_recall=loc_metric['oov_recall'],loc_iv_recall=loc_metric['iv_recall'],gradient_flow=json.dumps(flow))
            record['epoch_seconds']=record['train_seconds']+record['validation_seconds'];progress['history'].append(record)
            if fold is not None or metric['f1']>progress['best']:
                progress['best']=metric['f1'];progress['bad']=0
                save_weights(folder/'best.pt',dict(model=model.state_dict(),representation=state,epoch=epoch,
                    code_hash=CODE_HASH,split=SPEC['content_sha256'],identity=identity,metrics=metric))
            else:progress['bad']+=1
            progress.update(epoch=epoch+1,batch=0,totals=[0.,0.,0.,0.,0,0.]);persist()
            tracking_log(tracker,record)
            print(kind,'fold',fold,'epoch',epoch,'F1',metric['f1'],'seconds',round(record['epoch_seconds'],1),
                  'session minutes left',round(time_left()/60,1),flush=True)
    except Exception:
        persist();tracking_finish(tracker,dict(complete=False,epochs=len(progress['history'])))
        del model;torch.cuda.empty_cache();raise
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    result=dict(name=kind,fold=fold,identity=identity,complete=True,checkpoint=str(folder/'best.pt'),
        checkpoint_sha256=source.file_hash(folder/'best.pt'),parameters=params,best_epoch=best['epoch'],
        epochs=len(progress['history']),seconds=progress['seconds'],train_rows=len(train),validation_rows=len(valid),
        train_ids_sha256=source.fingerprint([d['row_id'] for d in train]),valid_ids_sha256=source.fingerprint([d['row_id'] for d in valid]),
        representation_sha256=source.fingerprint(state),class_counts=counts,class_weights=weights.cpu().tolist(),
        seed=42 if fold is None else 42+fold,encoder=base.CONFIG['encoder'],optimizer='AdamW',lr=.002,weight_decay=.01,
        batch_size=128,split_sha256=SPEC['content_sha256'],protocol_ok=not progress['fallbacks'],fallbacks=progress['fallbacks'],hardware=HARDWARE,**best['metrics'])
    write_json(folder/'result.json',result)
    tracking_finish(tracker,dict(complete=True,best_epoch=result['best_epoch'],training_seconds=result['seconds'],parameters=params))
    tracking_saved(folder,result);del model;torch.cuda.empty_cache()
    return result


def metrics_from_counts(rows):
    tp,fp,fn=rows[:,:3].sum(0)
    precision=tp/(tp+fp) if tp+fp else float(tp+fn==0)
    recall=tp/(tp+fn) if tp+fn else 1.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    ov,og,iv,ig=rows[:,3:].sum(0)
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(precision),recall=float(recall),f1=float(f1),
                oov_recall=float(ov/og) if og else None,iv_recall=float(iv/ig) if ig else None,
                oov_entities=int(og),iv_entities=int(ig))


def boundary_errors(d, predicted):
    gold=set(d['trimmed']);pred=set(predicted);exact=gold & pred;g=list(gold-exact);p=list(pred-exact)
    result=dict(exact=len(exact),truncated=0,expanded=0,shifted=0,missed=0,false_positive=0)
    matches=sorted([(min(b,y)-max(a,x),i,j) for i,(a,b) in enumerate(g) for j,(x,y) in enumerate(p)
                    if max(a,x)<min(b,y)],reverse=True)
    gi=set();pi=set()
    for _,i,j in matches:
        if i in gi or j in pi:continue
        gi.add(i);pi.add(j);a,b=g[i];x,y=p[j]
        category='truncated' if a<=x<y<=b else ('expanded' if x<=a<b<=y else 'shifted')
        result[category]+=1
    result['missed']=len(g)-len(gi);result['false_positive']=len(p)-len(pi)
    return result


def statistics(docs, p, threshold=.8, policy='strict'):
    locations,bounds,_=base.decode(docs,p,threshold,policy)
    rows=[];errors=[]
    for d,values,spans in zip(docs,locations,bounds):
        assert all(0<=a<b<=len(d['text']) and d['text'][a:b]==value for (a,b),value in zip(spans,values))
        gold={}
        for a,b in d['spans']:
            name=hard_processing(d['text'][a:b])
            unknown=any(token_key(t.value) not in BASE_VOCAB for t in d['tokens'] if a<t.end and t.start<b)
            gold[name]=gold.get(name,False) or unknown
        pred={hard_processing(s) for s in values};gs=set(gold)
        ov=sum(unknown and s in pred for s,unknown in gold.items());og=sum(gold.values())
        iv=sum(not unknown and s in pred for s,unknown in gold.items());ig=len(gold)-og
        rows.append([len(gs & pred),len(pred-gs),len(gs-pred),ov,og,iv,ig])
        errors.append(boundary_errors(d,spans))
    rows=np.array(rows,dtype=np.int64)
    return rows,errors


def describe(name, partition, docs, p, folder):
    rows,errors=statistics(docs,p);raw,_=statistics(docs,p,0.,'original')
    summary=dict(model=name,partition=partition,threshold=.8,prediction_filter='strict',**metrics_from_counts(rows),raw_f1=metrics_from_counts(raw)['f1'],
                 bio_incompatible=sum(d['bio'] is None for d in docs))
    lengths=np.array([len(d['tokens']) for d in docs])
    length_rows=[]
    for lo,hi in [(1,16),(17,32),(33,64),(65,128),(129,10**9)]:
        ix=(lengths>=lo)&(lengths<=hi)
        if ix.any():length_rows.append(dict(model=name,partition=partition,length=f'{lo}-{hi}',rows=int(ix.sum()),**metrics_from_counts(rows[ix])))
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'metrics.json',summary);write_csv(folder/'length_f1.csv',length_rows)
    confusion=np.zeros((3,3),dtype=np.int64);orphans=0
    for d,values in zip(docs,p):
        tags=values.argmax(1);orphans+=decode_bio(d['tokens'],tags)[1]
        if d['bio'] is not None:np.add.at(confusion,(d['bio'],tags),1)
    write_json(folder/'bio_confusion.json',dict(gold_rows_predicted_columns=confusion.tolist(),orphan_I=orphans,
        note='Raw argmax before filtering; incompatible gold rows excluded from token confusion'))
    write_csv(folder/'boundary_errors.csv',[dict(model=name,partition=partition,**{k:sum(r[k] for r in errors) for k in errors[0]})])
    write_csv(folder/'oov_recall.csv',[dict(group=k,entities=summary[k+'_entities'],recall=summary[k+'_recall']) for k in ['oov','iv']])
    return summary,rows


def paired_interval(reference, candidate, groups):
    assert reference.shape==candidate.shape==(len(groups),7)
    keys={};old=[];new=[]
    for g,a,b in zip(groups,reference,candidate):
        if g not in keys:keys[g]=len(old);old.append(np.zeros(7,dtype=np.int64));new.append(np.zeros(7,dtype=np.int64))
        old[keys[g]]+=a;new[keys[g]]+=b
    old=np.array(old);new=np.array(new);rng=np.random.default_rng(42);deltas=[]
    for i in range(POLICY['bootstrap']):
        if i%50==0:check_time()
        ix=rng.integers(0,len(old),len(old))
        a=old[ix,:3].sum(0);b=new[ix,:3].sum(0)
        fa=2*a[0]/max(1,2*a[0]+a[1]+a[2]);fb=2*b[0]/max(1,2*b[0]+b[1]+b[2])
        deltas.append(fb-fa)
    return dict(delta_f1=metrics_from_counts(candidate)['f1']-metrics_from_counts(reference)['f1'],
        low=float(np.quantile(deltas,.025)),high=float(np.quantile(deltas,.975)),
        groups=len(old),repeats=POLICY['bootstrap'])


def baseline_cv_protocol(fold, path, train, valid):
    expected=REFERENCE['cv'][fold]
    assert source.file_hash(path)==expected['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True);c=saved['config']
    for key,value in dict(encoder=base.CONFIG['encoder'],seed=42+fold,optimizer='AdamW',lr=.002,weight_decay=.01,
            batch_size=128,max_tokens=128,overlap=32,gradient_clip=1.,betas=[.9,.999],eps=1e-8,
            epochs=8,selection='fixed_final_epoch',scheduler='replay',parent_scheduler='none',tokenizer='word').items():
        assert c[key]==value, f'Baseline CV protocol mismatch: {key}'
    assert c['epoch_lrs']==[.002]*8 and saved['epoch']==8 and saved['amp']=='bf16'
    assert c['split_sha256']==SPEC['reference']['source_split_sha256']
    assert c['train_rows']==len(train)
    assert c['train_ids_sha256']==source.fingerprint([d['row_id'] for d in train])
    assert c['validation_ids_sha256']==source.fingerprint([d['row_id'] for d in valid])
    rebuilt=vocabulary(train)
    assert rebuilt==saved['encoding']['vocabulary']
    assert source.fingerprint(saved['encoding'])==expected['encoding_sha256']
    weights,counts=class_weights(train,torch.device('cpu'))
    assert counts==expected['word_tag_counts']
    np.testing.assert_allclose(weights.numpy(),expected['class_weights'],rtol=1e-6)
    report=dict(fold=fold,verified=True,epochs=8,seed=42+fold,training_amp='bf16',evaluation_amp='fp32',
        selection='fixed_final_epoch',primary_threshold=.8,primary_filter='strict',historical_threshold=0.,
        historical_filter='original',checkpoint_sha256=source.file_hash(path),train_ids=c['train_ids_sha256'],
        valid_ids=c['validation_ids_sha256'],word_vocabulary_sha256=source.fingerprint(rebuilt))
    write_json(OUT/'baseline'/f'protocol_fold{fold}.json',report)
    return saved


def compare_cv(kind, baseline_paths):
    cached=OUT/kind/'cv_comparison.json'
    if cached.exists():
        previous=json.loads(cached.read_text())
        for path,expected in previous['checkpoint_hashes'].items():
            assert source.file_hash(Path(path))==expected
        return previous
    if any(path is None for path in baseline_paths):
        raise RuntimeError('Baseline CV checkpoints missing; no equivalent CV comparison or automatic promotion is possible')
    if not HARDWARE['bf16']:
        raise RuntimeError('Native BF16 required to match historical baseline CV')
    verified=[]
    for fold in range(3):
        train=[d for d,f in zip(DATA['fit'],SPEC['cv_folds']) if f!=fold]
        valid=[d for d,f in zip(DATA['fit'],SPEC['cv_folds']) if f==fold]
        train_guard(train,valid,fold)
        verified.append(baseline_cv_protocol(fold,baseline_paths[fold],train,valid))
    rows_base=np.zeros((len(DATA['fit']),7),dtype=np.int64);rows_new=rows_base.copy()
    raw_base=rows_base.copy();raw_new=rows_base.copy();fold_results=[]
    for fold in range(3):
        ix=np.array(SPEC['cv_folds'])==fold
        train=[d for d,f in zip(DATA['fit'],SPEC['cv_folds']) if f!=fold]
        valid=[d for d,f in zip(DATA['fit'],SPEC['cv_folds']) if f==fold]
        train_guard(train,valid,fold)
        saved=verified[fold]
        bm=NERModel(saved['encoding']['size'],**saved['config']['encoder']).cuda()
        bm.load_state_dict(saved['model']);bm.eval()
        bp=cached_probabilities(OUT/'baseline'/f'cv{fold}.npz',bm,valid,saved['encoding'],source.file_hash(baseline_paths[fold]),baseline=True)
        _,rows_base[ix]=describe('baseline',f'cv{fold}',valid,bp,OUT/'baseline'/f'cv{fold}')
        raw_base[ix],_=statistics(valid,bp,0.,'original');del bm;torch.cuda.empty_cache()
        run=fit_run(kind,train,valid,OUT/kind/f'cv{fold}',fold)
        assert run['class_counts']==REFERENCE['cv'][fold]['word_tag_counts']
        np.testing.assert_allclose(run['class_weights'],REFERENCE['cv'][fold]['class_weights'],rtol=1e-6)
        model,ns=load_candidate(run['checkpoint'])
        cp=cached_probabilities(OUT/kind/f'cv{fold}.npz',model,valid,ns['representation'],run['checkpoint_sha256'])
        _,rows_new[ix]=describe(kind,f'cv{fold}',valid,cp,OUT/kind/f'cv{fold}'/'evaluation')
        tracking_saved(OUT/kind/f'cv{fold}',run)
        raw_new[ix],_=statistics(valid,cp,0.,'original')
        fold_results.append(run);del model;torch.cuda.empty_cache()
    old=metrics_from_counts(rows_base);new=metrics_from_counts(rows_new)
    historical=metrics_from_counts(raw_base)
    assert round(historical['f1'],8)==SPEC['reference']['cv_metrics']['f1'], 'Historical raw baseline CV does not reproduce'
    ci=paired_interval(rows_base,rows_new,[d['group'] for d in DATA['fit']])
    result=dict(checkpoint_hashes={str(p):source.file_hash(p) for p in [*baseline_paths,*[Path(r['checkpoint']) for r in fold_results]]},
        baseline=old,candidate=new,baseline_raw=historical,candidate_raw=metrics_from_counts(raw_new),
        interval=ci,protocol_ok=all(r['protocol_ok'] for r in fold_results),
        note='Primary comparison is strict/.8 for both. Historical .70673575 is raw and is not used as the strict baseline.')
    write_json(OUT/kind/'cv_comparison.json',result)
    write_csv(OUT/kind/'cv_oof_counts.csv',[dict(row_id=d['row_id'],group=d['group'],fold=SPEC['cv_folds'][i],
        **{'baseline_'+k:int(v) for k,v in zip(['tp','fp','fn','oov_tp','oov_gold','iv_tp','iv_gold'],a)},
        **{'candidate_'+k:int(v) for k,v in zip(['tp','fp','fn','oov_tp','oov_gold','iv_tp','iv_gold'],b)})
        for i,(d,a,b) in enumerate(zip(DATA['fit'],rows_base,rows_new))])
    return result


def acceptance(cv, calibration, baseline_cal):
    old,new=cv['baseline'],cv['candidate']
    return dict(cv_gain=new['f1']-old['f1']>=POLICY['cv_gain'],
        oov_gain=new['oov_recall']-old['oov_recall']>=POLICY['oov_gain'],
        positive_interval=cv['interval']['low']>0,
        iv_recall=new['iv_recall']>=old['iv_recall']-POLICY['iv_tolerance'],
        precision=new['precision']>=old['precision']-POLICY['precision_tolerance'],
        calibration=calibration['f1']>=baseline_cal['f1']-POLICY['calibration_tolerance'],
        protocol=cv['protocol_ok'])


def plot_curves():
    for kind in ['bpe','ngrams']:
        path=OUT/kind/'main/history.csv'
        if not path.exists() or path.stat().st_size<=2:continue
        table=pd.read_csv(path)
        if table.empty:continue
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        axes[0].plot(table.epoch,table.train_loss,label='train');axes[0].plot(table.epoch,table.val_loss,label='validation');axes[0].legend()
        axes[1].plot(table.epoch,table.f1,label='raw F1');axes[1].plot(table.epoch,table.precision,label='precision');axes[1].plot(table.epoch,table.recall,label='recall');axes[1].legend()
        axes[2].plot(table.epoch,table.gradient_norm,label='gradient norm');axes[2].legend()
        for axis in axes:axis.set_xlabel('Epoch')
        fig.tight_layout();fig.savefig(OUT/kind/'training_curves.png',dpi=120);plt.close(fig)
        if 'gradient_flow' in table and isinstance(table.gradient_flow.iloc[-1],str):
            flow=json.loads(table.gradient_flow.iloc[-1]);fig,axis=plt.subplots(figsize=(10,3))
            axis.bar(range(len(flow)),[max(v,1e-12) for v in flow.values()]);axis.set_yscale('log')
            axis.set_xticks(range(len(flow)),flow.keys(),rotation=90,fontsize=6)
            axis.set_ylabel('Gradient L2 norm after clipping');axis.set_title('Last batch of last completed epoch')
            fig.tight_layout();fig.savefig(OUT/kind/'gradient_flow.png',dpi=120);plt.close(fig)
        result=OUT/kind/'main/result.json'
        if result.exists():tracking_saved(result.parent,json.loads(result.read_text()))


def is_promising(candidate, old, calibration, old_cal, run):
    return (run['protocol_ok'] and candidate['f1']>=old['f1']-.005
        and candidate['oov_recall']>=old['oov_recall']+.03
        and candidate['iv_recall']>=old['iv_recall']-.01
        and candidate['precision']>=old['precision']-.01
        and calibration['f1']>=old_cal['f1']-.005)


def summary_files(comparison, status):
    reference=REFERENCE['main']
    experiments=[dict(model='baseline',stage='main',status='historical_reference',parameters=BASE_PARAMETERS,
        epochs=reference['epochs'],best_epoch=reference['best_epoch'],f1=reference['f1'],seconds=reference['runtime_seconds'],
        checkpoint_sha256=reference['checkpoint_sha256'])]
    for kind in ['bpe','ngrams']:
        for stage in ['main','cv0','cv1','cv2']:
            folder=OUT/kind/stage
            if not folder.exists():
                if stage=='main':experiments.append(dict(model=kind,stage=stage,status='not_started'))
                continue
            path=folder/'result.json'
            if path.exists():
                r=json.loads(path.read_text())
                experiments.append(dict(model=kind,stage=stage,status='complete',**r))
            else:
                path=folder/'history.csv'
                h=pd.read_csv(path) if path.exists() and path.stat().st_size>2 else pd.DataFrame()
                config=json.loads((folder/'run_config.json').read_text()) if (folder/'run_config.json').exists() else {}
                experiments.append(dict(model=kind,stage=stage,status='incomplete',epochs_completed=len(h),**config,
                    checkpoint=str(folder/'latest.pt') if (folder/'latest.pt').exists() else None))
    write_csv(OUT/'experiments.csv',experiments)
    write_csv(OUT/'comparison.csv',comparison)
    plot_curves()
    write_json(OUT/'status.json',status)
    text=['# Representation experiments', '',f"Status: {status['status']}",f"Selected: {status['selected']}",
        '', 'Main selection: raw checkpoint F1, max 15 epochs, patience 6. No two-epoch rejection.',
        'Main comparison and CV decision: strict filter, threshold .8 for both models.',
        'Historical raw CV .70673575 is recorded separately, not mixed with strict CV.',
        'OOV groups use the frozen main baseline vocabulary for all models, including CV diagnostics.',
        'Only main fit/fold fit trains representations and class weights. Routing tables contain no learned heldout vocabulary.',
        'Checkpoint/calibration and existing folds are development evidence, not untouched nested CV.',
        'Organizer examples never enter selection. Audit cannot change selection.',
        'Passing source development gates does not prove a higher Kaggle Public F1.',
        'This notebook and its experimental code are AI-assisted; the student must review and disclose this work.',
        '',f"Details: {json.dumps(status,ensure_ascii=False,indent=2)}",
        '',f'Session wall time: {(time.monotonic()-STARTED)/60:.1f} minutes.',
        'GPU timing is measured in history.csv; a new run is not verified until it completes on Colab.']
    path=OUT/'decision_report.md';tmp=path.with_suffix('.tmp');tmp.write_text('\n'.join(text));tmp.replace(path)


def audit_once(kind, run):
    decision=json.loads((OUT/'decision.json').read_text())
    assert decision['selected']==kind and decision['accepted']
    assert decision['checkpoint_sha256']==run['checkpoint_sha256']
    path=OUT/'final_audit.json'
    if path.exists():
        previous=json.loads(path.read_text());assert previous['checkpoint_sha256']==run['checkpoint_sha256']
        return previous
    check_time()
    # Gold is materialized for this one frozen candidate only after the decision is persisted.
    frame=FRAMES['audit'];audit=documents(frame)
    for d,group in zip(audit,frame.group):d['group']=group
    assert source.documents_hash(audit)==SPEC['documents_sha256']['audit']
    assert {d['group'] for d in audit}.isdisjoint(d['group'] for d in DATA['fit'])
    model,saved=load_candidate(run['checkpoint'])
    p=cached_probabilities(OUT/kind/'audit.npz',model,audit,saved['representation'],run['checkpoint_sha256'])
    summary,_=describe(kind,'audit_once_after_selection',audit,p,OUT/kind/'audit')
    summary['checkpoint_sha256']=run['checkpoint_sha256'];write_json(path,summary)
    del model;torch.cuda.empty_cache()
    return summary


def final_exports(kind, run):
    model,saved=load_candidate(run['checkpoint'])
    p=cached_probabilities(OUT/kind/'test.npz',model,DATA['test'],saved['representation'],run['checkpoint_sha256'])
    path=base.export_submission(DATA['test'],p,OUT/'submission_candidate.csv',.8,'strict')
    write_json(OUT/'submission_identity.json',dict(model=kind,checkpoint_sha256=run['checkpoint_sha256'],
        submission_sha256=source.file_hash(path),split=SPEC['content_sha256'],text_ids=SPEC['test_ids'],threshold=.8,filter='strict'))
    del model;torch.cuda.empty_cache()


def organizer_diagnostic(kind, run):
    path=OUT/kind/'organizer/metrics.json'
    if path.exists():return
    check_time()
    frame=pd.read_csv(DATA_DIR/'labeling_sample.csv',keep_default_na=False)
    sample=documents(frame)
    for d in sample:d['group']=source.text_group(d['text'])
    assert {d['group'] for d in sample}.isdisjoint(d['group'] for d in DATA['fit'])
    model,saved=load_candidate(run['checkpoint'])
    p=cached_probabilities(OUT/kind/'organizer.npz',model,sample,saved['representation'],run['checkpoint_sha256'])
    describe(kind,'organizer_diagnostic_only',sample,p,OUT/kind/'organizer')
    del model;torch.cuda.empty_cache()


def run_experiments(baseline_path, baseline_paths):
    status=dict(status='incomplete',selected='baseline',accepted=False,missing=[],session_gpu=HARDWARE)
    comparison=[];runs={};main={};counts={}
    try:
        status['stage']='baseline checkpoint/calibration evaluation'
        model,bs=base.load_model(baseline_path,SPEC,'cuda',reference=True)
        global BASE_VOCAB
        BASE_VOCAB=bs['vocabulary']
        global BASE_PARAMETERS
        BASE_PARAMETERS=sum(p.numel() for p in model.parameters())
        baseline_parameters=BASE_PARAMETERS
        for partition in ['checkpoint','calibration']:
            p=cached_probabilities(OUT/'baseline'/f'{partition}.npz',model,DATA[partition],bs,
                SPEC['reference']['checkpoint_sha256'],baseline=True)
            main['baseline',partition],counts['baseline',partition]=describe('baseline',partition,DATA[partition],p,OUT/'baseline'/partition)
            comparison.append(main['baseline',partition])
        assert round(main['baseline','checkpoint']['raw_f1'],8)==SPEC['reference']['metrics']['checkpoint_original']['f1']
        assert round(main['baseline','calibration']['f1'],8)==SPEC['reference']['metrics']['calibration']['f1']
        write_json(OUT/'baseline'/'parameters.json',dict(parameters=baseline_parameters,checkpoint_sha256=SPEC['reference']['checkpoint_sha256']))
        del model;torch.cuda.empty_cache()
        tracking_baseline({partition:main['baseline',partition] for partition in ['checkpoint','calibration']},baseline_parameters)
        for kind in ['bpe','ngrams']:
            status['stage']=kind+' main training'
            runs[kind]=fit_run(kind,DATA['fit'],DATA['checkpoint'],OUT/kind/'main')
            status['stage']=kind+' development evaluation and bootstrap'
            model,saved=load_candidate(runs[kind]['checkpoint'])
            for partition in ['checkpoint','calibration']:
                p=cached_probabilities(OUT/kind/f'{partition}.npz',model,DATA[partition],saved['representation'],runs[kind]['checkpoint_sha256'])
                summary,rows=describe(kind,partition,DATA[partition],p,OUT/kind/partition)
                ci=paired_interval(counts['baseline',partition],rows,[d['group'] for d in DATA[partition]])
                summary.update(ci_low=ci['low'],ci_high=ci['high'],delta_f1=ci['delta_f1'])
                main[kind,partition]=summary;comparison.append(summary)
            del model;torch.cuda.empty_cache();summary_files(comparison,status)
        status['stage']='development screen and CV shortlist'
        promising=[k for k in ['bpe','ngrams'] if is_promising(main[k,'checkpoint'],main['baseline','checkpoint'],
            main[k,'calibration'],main['baseline','calibration'],runs[k])]
        chosen=max(promising,key=lambda k:(main[k,'checkpoint']['f1'],main[k,'checkpoint']['oov_recall'],k)) if promising else None
        shortlist=dict(selected_for_cv=chosen,main_checkpoints={k:r['checkpoint_sha256'] for k,r in runs.items()},
            rule='Checkpoint >= baseline-.005, OOV +.03, IV/precision >= baseline-.01, calibration >= baseline-.005; rank checkpoint F1 then OOV recall')
        path=OUT/'shortlist.json'
        if path.exists():assert json.loads(path.read_text())==shortlist
        else:write_json(path,shortlist)
        decision_path=OUT/'decision.json'
        if decision_path.exists():
            decision=json.loads(decision_path.read_text())
            assert decision['main_checkpoints']==shortlist['main_checkpoints']
        elif chosen is None:
            decision=dict(selected='baseline',accepted=False,status='complete',reason='No main model passed the prespecified development screen',
                main_checkpoints=shortlist['main_checkpoints'])
            write_json(decision_path,decision)
        else:
            status.update(cv_candidate=chosen,stage='matched baseline protocol and candidate CV');summary_files(comparison,status)
            cv=compare_cv(chosen,baseline_paths)
            checks=acceptance(cv,main[chosen,'calibration'],main['baseline','calibration'])
            accepted=all(checks.values())
            decision=dict(selected=chosen if accepted else 'baseline',accepted=accepted,status='complete',checks=checks,
                checkpoint_sha256=runs[chosen]['checkpoint_sha256'] if accepted else SPEC['reference']['checkpoint_sha256'],
                main_checkpoints=shortlist['main_checkpoints'],cv=cv)
            write_json(decision_path,decision)
        status.update(decision)
        if 'cv' in decision:
            cv=decision['cv']
            for fold in range(3):
                folder=OUT/chosen/f'cv{fold}'
                if (folder/'result.json').exists():tracking_saved(folder,json.loads((folder/'result.json').read_text()))
            tracking_cv(chosen,cv)
            for name,metrics in [('baseline',cv['baseline']),(chosen,cv['candidate'])]:
                comparison.append(dict(model=name,partition='cv_oof_strict_0.8',threshold=.8,prediction_filter='strict',**metrics,
                    raw_f1=cv['baseline_raw' if name=='baseline' else 'candidate_raw']['f1'],
                    ci_low=cv['interval']['low'],ci_high=cv['interval']['high']))
        if decision['accepted']:
            # The audit result is not an argument to acceptance or model ranking.
            status.update(audit_status='pending',stage='audit of fixed accepted candidate');summary_files(comparison,status)
            audit_once(decision['selected'],runs[decision['selected']]);status['audit_status']='complete'
            status['stage']='candidate test predictions and export'
            final_exports(decision['selected'],runs[decision['selected']]);status['submission']='submission_candidate.csv'
            status['stage']='organizer diagnostics after selection'
            organizer_diagnostic(decision['selected'],runs[decision['selected']]);status['organizer_status']='diagnostic_complete'
        else:
            status['submission']='No candidate submission; keep the original baseline submission'
        status['stage']='complete'
    except BudgetEnd as error:
        status.update(status='incomplete',reason=str(error))
        status['missing'].append('Interrupted stage: '+status.get('stage','unknown'))
        status['missing'].append('Continue unfinished stages with Run All; no incomplete experiment is accepted')
    except Exception as error:
        status.update(status='blocked',reason=f'{type(error).__name__}: {error}')
        status['missing'].append('Blocked stage: '+status.get('stage','unknown'))
        status['missing'].append('Resolve the recorded issue before treating the experiment queue as complete')
        print(status['reason'])
    finally:
        torch.cuda.empty_cache();summary_files(comparison,status)
    print(json.dumps(status,indent=2))
    return status


def run(root, work, baseline_source, budget_minutes=45):
    global BUDGET_MINUTES,DATA,DATA_DIR,DEADLINE,FRAMES,HARDWARE,OUT,PINNED,ROOT,SPEC,STARTED,WB_ENTITY,WB_MODE,WB_PROJECT,WORK,baseline_path,baseline_paths,candidates,destination,dirname,expected,i,identity,k,key,local,name,path,r,value
    ROOT=Path(root);WORK=Path(work);WORK.mkdir(parents=True,exist_ok=True)
    OUT=ROOT/"results/word_representations_v1";OUT.mkdir(parents=True,exist_ok=True)
    PINNED=Path(baseline_source);BUDGET_MINUTES=budget_minutes
    for name,digest in RUNTIME_HASHES.items():
        if source.file_hash(PINNED/name)!=digest:raise ValueError("Historical baseline source differs: "+name)
    if source.file_hash(base.__file__)!=RUNTIME_HASHES["part2.py"]:raise ValueError("Load the historical baseline in a fresh process; do not replace repository part2.py")
    WB_PROJECT = os.environ.get('WANDB_PROJECT','kse-ai301-a3')
    WB_ENTITY = os.environ.get('WANDB_ENTITY')
    WB_MODE = os.environ.get('WANDB_MODE','online' if os.environ.get('WANDB_API_KEY') else 'offline')
    assert WB_MODE in {'online','offline'}
    HARDWARE = base.gpu()
    STARTED = time.monotonic()
    DEADLINE = STARTED + BUDGET_MINUTES*60
    print(HARDWARE)
    assert POLICY['threshold'] == .8 and POLICY['filter'] == 'strict'
    identity = dict(code=CODE_HASH, baseline=BASELINE_COMMIT, assets=ASSET_HASHES,
                    policy=POLICY, main_config=base.CONFIG)
    path = OUT / 'experiment_identity.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity, 'Existing experiment code or policy differs; keep its results and use another output folder'
    else:
        write_json(path,identity)
    write_json(OUT / 'tracking_version.json',dict(training_code_hash=CODE_HASH,tracking_code_hash=TRACKING_HASH,mode=WB_MODE))
    write_json(OUT / 'environment.json',dict(python=sys.version,torch=str(torch.__version__),
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=HARDWARE,
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','tokenizers','scikit-learn','matplotlib']}))
    DATA_DIR = WORK / 'data'
    DATA_DIR.mkdir(exist_ok=True)
    SPEC = source.read_split(PINNED / 'part2_split.json')
    write_json(OUT / 'grouping_runtime.json',dict(python=sys.version,baseline_commit=BASELINE_COMMIT,
        original_source_sha256=ASSET_HASHES['part2_data.py'],runtime_source_sha256=RUNTIME_HASHES['part2_data.py'],
        canonical_grouping_sha256=source.grouping_hash(),expected_grouping_sha256=SPEC['grouping']['implementation_sha256'],
        frozen_split_sha256=ASSET_HASHES['part2_split.json']))
    for name, expected in SPEC['files'].items():
        destination = DATA_DIR / name
        if not destination.exists() or source.file_hash(destination) != expected:
            base.copy_file(ROOT / 'data' / name,destination)
    FRAMES, SPEC = source.load_data(DATA_DIR, PINNED / 'part2_split.json')
    # Keep audit gold outside the development document dictionary.
    DATA = source.prepare_data({k:FRAMES[k] for k in ['fit','checkpoint','calibration','test']},SPEC)
    assert len(DATA['fit']) == 100000
    print({key:len(value) for key,value in DATA.items()})

    def reference_path(folder, expected, main=False):
        candidates = ([ROOT/'part2_model/checkpoint.pt'] if main else [ROOT/'part2_model/cv'/f'fold{folder[0]}.pt'])
        dirname = folder if main else folder[1]
        candidates += [ROOT/'results/clean_final/runs'/dirname/'best.pt', ROOT/'part2_model'/dirname/'best.pt']
        for path in candidates:
            if path.is_file():
                assert source.file_hash(path) == expected, 'Baseline artifact hash mismatch: '+str(path)
                local = WORK / ('baseline_main.pt' if main else f'baseline_cv{folder[0]}.pt')
                if not local.exists() or source.file_hash(local) != expected:
                    base.copy_file(path,local)
                return local
        return None

    baseline_path = reference_path(REFERENCE['main_folder'],SPEC['reference']['checkpoint_sha256'],True)
    assert baseline_path is not None, 'Missing baseline checkpoint: part2_model/checkpoint.pt or original clean_final run'
    baseline_paths = [reference_path((i,r['folder']),r['checkpoint_sha256']) for i,r in enumerate(REFERENCE['cv'])]
    for i,path in enumerate(baseline_paths):
        if path is None:
            print('CV prerequisite missing:',ROOT/'results/clean_final/runs'/REFERENCE['cv'][i]['folder']/'best.pt')
    write_csv(OUT/'baseline/history.csv',REFERENCE['main_history'])
    print('Results:',OUT)
    return run_experiments(baseline_path,baseline_paths)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--work',required=True)
    parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--minutes',type=float,default=45)
    args=parser.parse_args();run(args.root,args.work,args.baseline_source,args.minutes)
