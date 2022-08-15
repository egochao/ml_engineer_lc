### Language confidence ML engineer

- [Language confidence ML engineer](#language-confidence-ml-engineer)
- [1. Task definition](#1-task-definition)
- [2. My take on the task](#2-my-take-on-the-task)
  - [a. The problems is well known](#a-the-problems-is-well-known)
  - [b. The existing open source code is messy](#b-the-existing-open-source-code-is-messy)
  - [c. The problems is interesting - I would like to build the solution as one of my opensource project](#c-the-problems-is-interesting---i-would-like-to-build-the-solution-as-one-of-my-opensource-project)
- [3. My approach to this challenge](#3-my-approach-to-this-challenge)
  - [a. My assumption](#a-my-assumption)
  - [b. My approach](#b-my-approach)
  - [c. My techlonogies](#c-my-techlonogies)
- [4. My result](#4-my-result)
  - [a. Model include in the work - No parameters search](#a-model-include-in-the-work---no-parameters-search)
  - [b. Model optimized with Optuna](#b-model-optimized-with-optuna)
  - [b. Model train with distillation loss](#b-model-train-with-distillation-loss)
  - [d. Hightlight point](#d-hightlight-point)
- [5. Challenge I face](#5-challenge-i-face)
  - [a. New technology](#a-new-technology)
  - [b. Issue with torch version and random seed](#b-issue-with-torch-version-and-random-seed)
  - [c. Issue with GPU shareable development env](#c-issue-with-gpu-shareable-development-env)
- [6. Thing I would like to do better.](#6-thing-i-would-like-to-do-better)
  - [a. Test](#a-test)

### 1. Task definition
Build a proof of concept speech keyword spotting model using modern deep learning architectures and techniques.

- Dataset : [Speech Command](https://pytorch.org/audio/stable/datasets.html#speechcommands)

- Accuracy > 90%

- Show problem solving skills

### 2. My take on the task

####  a. The problems is well known
- We have a lot of model existing on the internet + code with accuracy > 90%
- The best solution for this task(and audio tasks in general) is using some kind of spectrogram as input
- Most solution include some really large model - usually Vision transformer and variations

#### b. The existing open source code is messy
- Most of the code are not well organized
- Lot of publication have no code to reproduce
- The training code some authors provide dont produce same accuracy as in paper

#### c. The problems is interesting - I would like to build the solution as one of my opensource project
- Here is my [repo](https://github.com/egochao/speech_commands_distillation_torch_lightling)


### 3. My approach to this challenge

Because on github there are many good opensource model - I dont want to just copy and run them again.

#### a. My assumption
- We have good model from other repo with high accuracy
- We want to deploy model on edge device so the model have to be as small as possible
- We want the accuracy to be above 90%


#### b. My approach
- Use logit from a state of art model to distill knowledge to a super small model
- Use docker as share able development envinroment - and deployment target also
- Use Optuna to search for optimized parameters set


#### c. My techlonogies
I want to use same technologies as Language confidence team

Some of these technologies are new to me => I want to show my ability to learn new thing here
| Technology      | Description |   |
| ----------- | ----------- | ----------- |
| Torch lightling      | Boilerplace for torch    | New |
| Weight and bias   | Experiment logging        | New |
| Optuna   | Hyperparameters search | New | 
| Model distillation   | Transfer knowledge to small model | New |

### 4. My result

#### a. Model include in the work - No parameters search
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 26900 | 94.2% |
| BC Resnet   | Experiment logging        | 10600 | 95.6% |  |

#### b. Model optimized with Optuna
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 35000 | 95.1% | |
| BC Resnet   | Experiment logging        | 22000 | 98.3% - best | |

#### b. Model train with distillation loss
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 28600 | 90.3% | |
| BC Resnet   | Experiment logging        |  |  | |


#### d. Hightlight point
- My best model have 22k parameters and accuracy on test set = 98.3% (Optuna optimized)
- Almost beat the state-of-art(98.5)
- The model size is superior compare with all other state-of-art model by some order of magnitude
- The distillation process is not success and it causing the model perform worst than non distill

![image]([data/my_best_result.png](https://github.com/egochao/ml_engineer_lc/blob/main/data/my_best_result.png?raw=true))


### 5. Challenge I face

#### a. New technology
- It take me quite some time to know new technology well

#### b. Issue with torch version and random seed
- I first train a simple conv on colab with > 90%. This work with torch 1.7. Torch lightling require torch>=1.9 and the accuracy drop to 86% on both colab and my server - all the code are the same. It take me 1-2 day to figure out that the seeding I did with torchlightling is the issue.

#### c. Issue with GPU shareable development env
- I am already work with docker CPU and docker GPU posed some issues about build speed and cache size. It take some time to get it right.


### 6. Thing I would like to do better.
#### a. Test
- I make a classic mistake. Thought I can finish this fast and test for model is a bit confusing. This prove to be a fatal mistake. I was slowdown massively due to breaking change that I found late and other issue with work flow. Lesson learnt

