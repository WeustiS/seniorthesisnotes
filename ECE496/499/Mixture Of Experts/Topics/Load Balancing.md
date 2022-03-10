
V-MoE 
* Importance Loss
	* The importance of some expert i for a batch of images X is the normalized routing weight for expert i over all images
	* Importance loss is the stddev of the Imporantance of the experts divided by the mean of the experts (all squared) (prop. to the variance)
	* In [[MoE - Shazeer]], token x only contributed to the importance of some expert if that expert was select for x. 
* Load Loss
	* To ensure that experts have similar output routing weights. More specifically, to balance the number of assignments. This is discrete, so we use a proxy
	* From [[MoE - Shazeer]], we compute the probability of some expert i being selected (among the top-k).  If we resamply ONLY the noise for expert i. 
		* Simplified: $threshold_k(x) = max_{k-th}(Wx+\epsilon)$
		* We compute the probability of i being above the threshold if we resample the noise as follows:
		* ![[Pasted image 20220126175110.png]]
		*  We define the load to be the sum of this probability for all images
		* Similar to Imporance loss, our loss is defined as the stddev of the load of all experts divided by the mean (all squared)
	* Auxiliary Loss
		* An average of the two previous losses.
	* Final Loss
		* ![[Pasted image 20220126175329.png]]
		* $lambda \approx .01$
