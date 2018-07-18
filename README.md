# MojiTalk
Xianda Zhou, and William Yang Wang. 2018. Mojitalk: Generating emotional responses at scale. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1128–1137. Association for Computational Linguistics.

Paper: https://arxiv.org/abs/1711.04090

Our lab: http://nlp.cs.ucsb.edu/index.html

## Emojis
The file ```emoji-test.txt``` (http://unicode.org/Public/emoji/5.0/emoji-test.txt) provides data for loading and testing emojis. The 64 emojis that we used in our work are marked with '64' in our modified ```emoji-test.txt``` file.

Unicode and the Unicode Logo are registered trademarks of Unicode, Inc. in the U.S. and other countries.

For terms of use, see http://www.unicode.org/terms_of_use.html

## Dependencies
* Python 3.5.2
* TensorFlow 1.2.1

## Usage
1. Preparation:
	
	Set up an environment according to the dependencies.

	Dataset: https://drive.google.com/file/d/1l0fAfxvoNZRviAMVLecPZvFZ0Qexr7yU/view?usp=sharing
	
	Unzip ```mojitalk_data.zip``` to the current path, creating ```mojitalk_data``` directory where our dataset is stored. Read the ```readme.txt``` in it for the format of the dataset.

2. Base model:
	1. Set the ```is_seq2seq``` variable in the ```cvae_run.py``` to ```True```
	2. Train, test and generate: ```python3 cvae_run.py```

		This will save several breakpoints, a log file and generation output in ```mojitalk_data/seq2seq/<timestamp>/```
	
3. CVAE model:
	1. Set the ```is_seq2seq``` variable in the ```cvae_run.py``` to ```False```
	2. Set path of pretrain model: Modify line 67 of ```cvae_run.py``` to load a previously trained base model. e.g.: ```saver.restore(sess, "seq2seq/07-17_05-49-50/breakpoints/at_step_18000.ckpt")``` 
	3. Train, test and generate: ```python3 cvae_run.py```
	
		This will save several breakpoints, a log file and generation output in ```mojitalk_data/cvae/<timestamp>/```.
	
		Note that the choice of base model breakpoint as the pretrain setting would influence the result of CVAE training. A overfitted base model may cause the CVAE to diverge.
	
4. Reinforced CVAE model:
	1. Train the emoji classifier: ```CUDA_VISIBLE_DEVICES=0 python3 classifier.py``` 
		
		The trained model will be saved in ```mojitalk_data/classifier/<timestamp>/breakpoints``` as a tensorflow breakpoint.
	
	2. Set path of pretrain model: Modify line 63/74 of ```rl_run.py``` to load a previously trained CVAE model and the classifier.
	3. Train, test and generate: ```python3 rl_run.py```
	
		This will save several breakpoints, a log file and generation output in ```mojitalk_data/cvae/<timestamp>/```.
