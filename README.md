## Proﬁle Consistency Identiﬁcation for Open-domain Dialogue Agents
[<img src="_static/pytorch-logo.png" width="10%">](https://github.com/pytorch/pytorch) [<img src="https://www.apache.org/img/ASF20thAnniversary.jpg" width="6%">](https://www.apache.org/licenses/LICENSE-2.0)

[<img align="right" src="_static/scir.png" width="20%">](http://ir.hit.edu.cn/)

This repository contains resources for EMNLP-20 main conference paper:

**Proﬁle Consistency Identiﬁcation for Open-domain Dialogue Agents**.
[[arXiv]](https://arxiv.org/abs/2009.09680)

The code here is ready for running. And all resources are ready.

## Resources

* Source codes for KvBERT model: [[Github]](https://github.com/songhaoyu/KvPI)

* Download the full KvPI dataset: [[GoogleDrive]](https://drive.google.com/file/d/1BVhk0_KnH9y-qiA1Rw5UV5vLG0d2FWUA/view?usp=sharing), [[BaiduNetdisk]](https://pan.baidu.com/s/1npsnLwanLYK-9iVcQXbPGg) pwd: ewjn

* Download checkpoint to reproduce the reported results: [[GoogleDrive]](https://drive.google.com/file/d/1WBMctI_9HmvhM-4OGyuBzWY4P3Tftu58/view?usp=sharing), [[BaiduNetdisk]](https://pan.baidu.com/s/1F4b2TTjqje6SifwF_HwaZQ) pwd: pt4g; MD5 for the checkpoint: 0993c09872f074a04d29a4851cf2cfce



## Introduction

Here is an example that shows the process of understanding profile consistency. The table on the left is the profiles, consisting of several key-value pairs. And an open-domain dialogue session is on the right, with an input message, and two different responses:

<p align="center">
<img src="_static/introduction.png" width="50%" />
</p>

We can see that both responses incorporate the location word, *Beijing*, in the given profile.  The first response, which is marked green, **expresses the meaning of welcoming others to come to their places**. It indicates the speaker is currently in *Beijing*. Therefore, it is consistent with the given profile. However, for the red marked response, it **expresses the hope of going to Beijing once**, thus indicates the speaker had never been to *Beijing* before. Obviously, this response contradicts the profile. 

For humans, they can easily understand the differences between these responses. But for machines, currently, they can hardly tell the differences. This work is intended to address this issue.




## KvPI Dataset

### 1. EXAMPLES
<p align="center">
<img src="_static/kvpi.png" width="85%" />
</p>

### 2. GUIDPOST

Here are some explanations for the above example:

|  Elements				| Explanations  |
|  ----  					| ----  |
| Profile					| Attribute information of the respondent, including three groups of attributes: *gender*, *location*, and *constellation*.	|
| Post					| Input information in a single-turn dialogue. Notice that the speaker on this side is not profiled. |
| Response				| Responses in a single-turn dialogue. It contains attribute related information, but not necessarily related to the response speaker's own attributes.	|
| Domain					| Attribute field to which the dialogue response belongs.	|
| Annotated Attributes	| Human-extracted attribute information from the dialogue responses. Different from the given profile under some circumstances.	|
| Label		| Human annotated labels for consistency relations between Profile and Response, including *Irrelevant*, *Entailed*, and *Contradicted*. For details of the consistency relations please refer to the next section.	|

### 3. DEFINITIONS OF CONSISTENCY RELATIONS
* **ENTAILED**: The response is exactly talking about the dialogue agent’s attribute information, and the attribute is consistent with its key-value proﬁle.
* **CONTRADICTED**: Although the response is talking about the dialogue agent’s attribute information, it is contradicted to at least one of the given key-value pairs. For example, given the proﬁle “{location: Beijing}”, “I am in Seattle” is contradicted to the proﬁle, while “She lives in Seattle” is not, because the latter is not talking about the dialogue agent’s attribute.
* **IRRELEVANT**: The response contains proﬁle-related information, but the information does not reveal the dialogue agent’s own attributes. As exempliﬁed above, “She lives in Seattle” is irrelevant, rather than contradicted, to the dialogue agent’s proﬁle “{location: Beijing}”. Another example is “I’m interested in the history of Beijing”. Although there is the attribute word “Beijing”, this response still does not reveal the dialogue agent’s location.

## KvBERT
<p align="center">
<img src="_static/kvbert.png" width="80%" />
</p>


## How to Run
### Requirements

The released codes have been tested with the following environments:

- [pytorch=1.3.0](https://pytorch.org/get-started/locally/)
- [cudatoolkit=9.2](https://developer.nvidia.com/cuda-toolkit)
- python=3.6
- tqdm
- sklearn

Higher cudatoolkit version may encounter unexpected errors. The pytorch/python dependencies can be installed using [Anaconda](https://www.anaconda.com/) virtual environment. For example:

```
conda create -n kvpi python=3.6
conda activate kvpi
conda install pytorch=1.3.0 torchvision cudatoolkit=9.2 -c pytorch
```

Then in your environment install the following dependencies:

```
pip install sklearn
pip install tqdm
```

`sklearn` is used to calculate f1 score and accuracy. `tqdm` is a lib for the progress bar.

### Usage

First download the following data and put it into the ./ckpt folder:

  - [kvbert\_epoch\_3](https://drive.google.com/file/d/1WBMctI_9HmvhM-4OGyuBzWY4P3Tftu58/view?usp=sharing) (trained checkpoints)


And make sure the data folder has the [KvPI_test.txt](./data/KvPI_test.txt) file, which is organized in a format that the model can read and is already in the repository.

Then run the following script:

```
./inference.sh
```

Run the script will make predictions on the test data, and the output is redirected to test\_prediction.txt. When finishing the prediction, the script will call f1\_acc.py to present final scores. In the end, there should be something like:

```
              precision    recall  f1-score   support

    Entailed      0.927     0.939     0.933      5116
Contradicted      0.902     0.918     0.910      3041
  Irrelevant      0.920     0.882     0.901      2843

    accuracy                          0.918     11000
   macro avg      0.917     0.913     0.915     11000
weighted avg      0.919     0.918     0.918     11000

0.9184545454545454
```

## What Can We Do Using KvPI?
Details will be updated later.

## MISC
* If the datasets, codes or checkpoints are of help to your work, please cite the following papers:

	<pre>
	@inproceedings{song-2020-kvpi,
	    title = {Profile Consistency Identification for Open-domain Dialogue Agents},
	    author = {Song, Haoyu and Wang, Yan and Zhang, Wei-Nan and Zhao, Zhengyu and Liu, Ting and Liu, Xiaojiang},
	    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
	    month = {November},
	    year = {2020},
	    publisher = {Association for Computational Linguistics},
	}
	</pre>

* Notice that we trained the KvBERT model from a private Chinese BERT-base checkpoint and thus didn't provide the training codes and scripts in this repository. If you have a reasonable purpose and indeed need the training scripts, please email *hysong@ir.hit.edu.cn*.
