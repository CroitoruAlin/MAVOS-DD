# [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)

## ⏳ Quick Start

### 1. Installation
(option 1) You can run the following script to configure the necessary environment:

```
git clone git@github.com:SCLBD/DeepfakeBench.git
cd DeepfakeBench
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh
```

(option 2) You can also utilize the supplied [`Dockerfile`](./Dockerfile) to set up the entire environment using Docker. This will allow you to execute all the codes in the benchmark without encountering any environment-related problems. Simply run the following commands to enter the Docker environment.

```
docker build -t DeepfakeBench .
docker run --gpus all -itd -v /path/to/this/repository:/app/ --shm-size 64G DeepfakeBench
```
Note we used Docker version `19.03.14` in our setup. We highly recommend using this version for consistency, but later versions of Docker may also be compatible.

### 2. Download Data

The MAVOSS-DD dataset can be downloaded from [here](https://huggingface.co/datasets/unibuc-cs/MAVOS-DD)


Upon downloading the datasets, please ensure to store them in the [`./datasets`](./datasets/) folder, arranging them in accordance with the directory structure outlined below:

```
datasets
├── MAVOS-DD
|   ├── english
|   |   |--liveportrait
|   |   | ...
    ...

Other datasets are similar to the above structure
```

If you choose to store your datasets in a different folder, you may specified the `rgb_dir` or `lmdb_dir` in `training\test_config.yaml` and `training\train_config.yaml`.

The downloaded json configurations should be arranged as:
```
preprocessing
├── dataset_json
|   ├── FaceForensics++.json
```

You may also store your configurations in a different folder by specifying the `dataset_json_folder` in `training\test_config.yaml` and `training\train_config.yaml`.

### 3. Preprocessing 

<a href="#top">[Back to top]</a>


DeepfakeBench follows a sequential workflow for face detection, alignment, and cropping. The processed data, including face images, landmarks, and masks, are saved in separate folders for further analysis.

To start preprocessing your dataset, please follow these steps:

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`. This file is necessary for Dlib's face detection functionality.

2. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `FaceForensics++`. (This step should be already done)

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:

```
cd preprocessing

python preprocess.py
```
You may skip the preprocessing step by downloading the provided data.

### 4. Rearrangement
To simplify the handling of different datasets, we propose a unified and convenient way to load them. The function eliminates the need to write separate input/output (I/O) code for each dataset, reducing duplication of effort and easing data management.

After the preprocessing above, you will obtain the processed data (*i.e., frames, landmarks, and masks*) for each dataset you specify. Similarly, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following line:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 


### 5. Training (optional)

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for TALL. You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./training/pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose the detector to be trained. 

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py \
--detector_path ./training/config/config/detector/tall.yaml  \
--train_dataset "MAVOS-DD" \
--test_dataset "MAVOS-DD" \
--no-save_ckpt \
--no-save_feat
```


For **multi-gpus training** (DDP), please refer to [`train.sh`](./train.sh) file for details.

To train other detectors using the code mentioned above, you can specify the config file accordingly. However, for the Face X-ray detector, an additional step is required before training. To save training time, a pickle file is generated to store the Top-N nearest images for each given image. To generate this file, you should run the [`generate_xray_nearest.py`](./training/dataset/generate_xray_nearest.py) file. Once the pickle file is created, you can train the Face X-ray detector using the same way above. If you want to check/use the files I have already generated, please refer to the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.2).


### 6. Evaluation
You can run directly the inference using our fine-tuned model on MAVOS-DD from [here](https://drive.google.com/drive/folders/1TVPIrGykPJtRieESx1zn81-ATnGKY6hk)
If you only want to evaluate the detectors to produce the results of the cross-dataset evaluation, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py --detector_path ./training/config/config/detector/tall.yaml --test_dataset "MAVOS-DD" --weights_path ./logs/training/tall_TALL_2025-05-12-19-38-38/test/MAVOS-DD/ckpt_best.pth --output_path ./predictions/finetuned/tall_test
```


## 🏆 Results
To get the results reported in the paper, please run the `evaluation.py` script and change the paths accordingly.









## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}

@article{chen2024textit,
  title={$$\backslash$textit $\{$X$\}$\^{} 2$-DFD: A framework for e $$\{$X$\}$ $ plainable and e $$\{$X$\}$ $ tendable Deepfake Detection},
  author={Chen, Yize and Yan, Zhiyuan and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}

@article{cheng2024stacking,
  title={Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Hao, Li and Ai, Jiaxin and Zou, Qin and Li, Chen and Wang, Zhongyuan},
  journal={arXiv preprint arXiv:2411.11396},
  year={2024}
}
```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
