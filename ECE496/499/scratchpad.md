FightSurvDataset
- https://arxiv.org/pdf/2002.04355v1.pdf
- 150 fight, 150 non-fight
	- mainly not movies and hockey games
- Baseline was 71% accuracy on 10 frames, 72% accuracy one 5 frames
- 92% current SOTA from "Efficient Spatio-Temporal Modeling Methods for Real-Time Violence Recognition"

RWF-2000
- 1000 examples of each
- 5 seconds, video-level, surveillance
- Results
	- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9439904
	- 92% accuracy w/ 1.3M Params
	- >85 ok, >90 great
	- Efficient Spatio-Temporal Modeling Methods for Real-Time Violence Recognition

RealLifeViolence
- 1000 examples of each
- Results
	- Efficient Spatio-Temporal Modeling Methods for Real-Time Violence Recognition
	- 97.8% acc

Youtube-Small
- Our custom dataset 
- Current SOTA? 
- ~82
: Efficient Spatio-Temporal Modeling Methods for Real-Time Violence Recognition
- acc are averaged across 5 fold cross-val except for rwf-2000 

XD-Violence downloaded but not processed

preprocessing&augmentation is left to you but pelase use imagenet mean/std preprocessing