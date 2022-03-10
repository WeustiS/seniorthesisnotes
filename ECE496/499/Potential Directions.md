Replace MoE with some sort of look-up - CPU friendly too! 
Allow model to select N patches from anywhere in img. Correllate patch value w/ entropy? possibly with [maximum subarray problem](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.9778&rep=rep1&type=pdf)
it is different to select the top k hjighest entropy patches versus select the k patches that sum to the highest entropy (only if no overlapping)


Why route to fixed number of experts? Allow token to choose how many/which to route to. Change k based on layer. Change k during training. Random k.

Self supervised routing- perform some augmentation and make sure patch is routed similarly

**On Routing**
- How important is early routing? Can we use Self Supervision to assist with better early routing?
- Can we reformulate the routing problem?
	- Hash/Lookup (like VQVAE2's codebook. Good for CPU)
	- Random routing hurts performance (sometimes), but too much destroys performance... why?
	- As a set-matching (DETR/hungarian matching) problem

**On Patch Selection**
- Non grid-based patch sampling
- Can entropy act as a placeholder for patch importance?

**On Model Architecture**
- Why are experts all 2 layer MLPs?
	- Why not deeper
	- Why not different layer types (attn, conv, LUT, etc or some combination)
- What % of OPs, weights are in experts?
- "Routing is all you need?"
	- Can we just do repeated mixture of experts? why ever have normal layers? Is this effective?
- Larger scale routing
	- i.e., simple images can be sent to smaller continuations of the architecture. 
- Can we generate an expert on the fly? I.e. output some encoding vector, generate weights based on that vector. apply weights 
	- really nice because reduce param count 

**On Sparsity** 
- We have data level sparsity (BPR), we have model level sparsity (MoE), what other types of sparsity can we exploit?

**On Data Efficiency**
- Dense transformers [[All Tokens Matter - Jiang]], [[CaiT - Touvron]] 
	- mentioned in [[V-MoE]] but unsure how this is relavent to data eff.
- Data efficient vision transformers [[DeiT - Touvron]], [[T2T-ViT - Yuan]]


**On Conditional Computing**
- grab one patch at a time, sample another if needed (w/ preprocess pot. w/ self sup)

**big ideas**
- locality is important. leverage early
- are attn layers important? 
- seperate idea of routing early and processing late
	- process visual input, understand what you're seeing, perform some prediction? 
- What if our blocks were
	- Self Supervision -> Routing -> Self Supervision
	- SS layers are trained sequentially. First we train on images, then on outputs from the first SS layer.
		- If you do SS on a SS model's output, what happens? same output? will two models approach the same conclusion?


Justification for SS
* Wayyy to many params, easy to get into local minima. Can help with not getting stuck



Quantify the Impact of different 'knobs' in reimplementation
- parameter sweep, build table of impact. see what matters
- can we reduce the size of a model?