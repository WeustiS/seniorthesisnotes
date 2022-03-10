[V-MoE Paper](https://arxiv.org/pdf/2106.05974.pdf)
* Serving & Training Dense models is expensive [[Energy Impact of Models]]
* [[Conditional Computation]] can increase model capacity while limiting training/inference cost (but applying a subste of the model)
	* MoE is a version of this [[test page]]
		* Seen prev. in [[MoE - Shazeer ]], [[GShard - Lepikhin]], [[Switch Transformers - Fedus]]


V-MoE introduces a model combining MoE with ViT
![[Pasted image 20220126172900.png | 500]]
V-MoE replaces the MLP layer with a sparse MoE layer
![[Pasted image 20220126172956.png | 800]]

Each image patch is routes to a subset of expects. 

V-MoE also introduces **Batch Prioritized Routing**
* This allows for image-level sparsity to 'ignore' certain patches (reducing compute in 'uninformative image regions')


Model Architecture Specifics - **MLP**
* in ViT, each MLP contained two layers & a GeLU |  $MLP(x) = \textbf{W}_2 \sigma_{gelu}(\textbf{W}_1x)$
* in V-MoE, they have the same architecture (but different weights for each expert)
	* Similar to M4 Translation model seen in [[GShard - Lepikhin]]
	
Model Architecture Specifics - **Routing**
* $g(x) = TOP_k(softmax(Wx + \epsilon))$
	* $\epsilon \sim \mathcal{N}(0,\,\frac{1}{E^{2}})$
	* E == Number of Experts 

[[Load Balancing]]

**Buffer Capacity**
* Each expert can only process a limited number of tokens
* Residual connections preserve unassigned tokens


Model Adjustments - **N MoE Layers**
* MoEs every other layer originally from [[GShard - Lepikhin]]
 > In Appendix E.1 we observe that, although using fewer MoE layers decreases the number of parameters of the model, it has typically little impact on quality and can speed-up the models significantly, since less communication overhead is incurred.

Model Adjustments - **k Selected Experts**
* generally, k=1 or k=2
* E = 32

Model Adjustments - **C Buffer Capacity** 
* C = 1.05 (slight slack)


Post Training Adjustments 
* k - Number of Experts Selected
* C - Capacity of some Expert
* positioning and quantity of expert layers is effectively fixed
	* Why? 
	* Let expert vector be some encoding, don't softmax
	* have model output weights for some layer based on that encoding (similar to attn. in that it is fast weights? maybe)
* BPR can be added late (since capacity is fixed and we teach zero padding)
	* Downstream finetuning is highly dependent on C 
	* BPR can recover some drop in perf from now C during finetuning
	* High capacity during fine tuning > high cap during pretraining

**Batch Prioritized Routing** / [[Patch Routing]]
* Very valuable with low capacity. Competitive at only 15-30% of tokens, equal at 50-70%.
* Routing function is applied on some batch of inputs $X \in \mathbb{R}^{N*PxD}$
	* read: an image has shape N\*PxD. N is batch, P is patch #, D is dimension
* For i<j, every TOP-i assignment has priority over TOP-j assignment. Read: All first choice routings are routed, then 2nd choice.
* Normal routing:
	* given top-i routing,try to assign position
*  patch routing
	* Sort patches by score
		* for each choices in k
			* assign patch to some expert 
	* Maximum Routing weight (how strongly it should be routed to a specific expert) is a good baseline and hard to beat 
* Extension:
	* Sort at patch-expert assignment level rather than global patch level
* Skip-Patch also explored
	* Discard some % of low priority patches, then follow patch routing
	* How is it different? 
		* Mid-tier top priority patches are discarded. We would attend to the lower-priority routings of patches with a high routing score (since they were not discarded) before attending to the mid-priority routings of a patch (which was discarded)

**Analysis**
* Specialized Experts
	* Intuition of experts should be classes is poor. What about k>1? there are experts at many depths. Routing is token-level (not image-level)
	* Later experts discrimate small sets of classes, early experts correlate well with low level features& are correlated w/ patch location
* Value of Routing
	* Failure case: Routing is load balancing for E copys of the same expert
		* What happens if we replace the router in layer i with random routing?
			* Later layers are more important, but has some variance
			* Why have routing earlier then? 
		* What happens if we replace the router in multiple layers with andom routing?
			* Performance quickly degrades. Token routing is not meaningless. Nearly 0 accuracy at the end. 50% accuracy after 4 random routes
* Routing weights distributions
	* Distribution of weights varies greatly across expert layers
	* Early layers are more indifferent to expert routing
	* TOP-1 and TOP-2 distributions diverge. The former to 1, the latter to 0
		* Why? 
			* Top-1 approaches 1 since more confident routing
			* Top-2 approaches 0, so it behaves more like k=1?
	* Generally, images distribute their patches among experts every layer
		* no shit? it's in the loss function
* Changing k at inference time
	* Models are flexible, to a point
	* if k is small, the model hasn't learned to combine expert information (poor experts give input)
	* if k is large, the model may rely on combining.
	* They find k =2 allows for k'=1 to perform almost equally to k=1 and also k'=3 or 4 performs well. For larger k', k=2 degrades in performance, but not as quickly as if k=1. (but more than if k>>2)

**Pretraining with less data**
* Compare dto dense models, V-MoE performs almost equally (or slightly worse) for low amounts of data. yet it performs better for large amounts of data
* Yet, sparse model performs generally better on imagenet 21k IFF images are augmented. RandAugment helps expert models but hurts dense models!
	* Careful use of regularisation and data augmentation is necessary for further exploration
	* Dense transformers [[All Tokens Matter - Jiang]], [[CaiT - Touvron]] beneficial
	* Data efficient vision transformers [[DeiT - Touvron]], [[T2T-ViT - Yuan]]



Related Work
* Conditional Computation
* MoE 
	* [[Adaptive mixtures of local experts]] 
	* [[Twenty years of mixture of experts]]
* MoE for NLP
	* [[MoE - Shazeer]], [[Switch Transformers - Fedus]], [[BASE - Lewis]] (Routing is Linear Assignment)
* MoE for Vision
	* TODO 

To reduce the dimensionality (62) of the collected pose data the authors marginalize into a lower-dimensional latent space (inspired by GPLVM). This latent space can be used as encoding of the data that one can perform inference on. 