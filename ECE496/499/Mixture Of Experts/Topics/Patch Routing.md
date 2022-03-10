Explore: 
Patch-expert level sorting 
> A natural extension of this algorithm consists in sorting at the patch-expert assignment level, rather than at the global patch level. The main difference with Algorithm 2 is that the sorting then looks at (patch p, TOP−i expert for p) scores for 1 ≤ i ≤ k. For example, assume k = 2 and we have two patches, p1 and p2. Suppose p1 selects experts (e11, e12) with routing weights (0.7, 0.2), while p2 selects (e21, e22) with weights (0.5, 0.4). Under Algorithm 2 the order in which patch-expert assignments would be attempted is: (p1, e11), (p2, e21), (p1, e12), (p2, e22). If we use sorting at the patch-expert level, however, we would end up with: (p1, e11), (p2, e21), (p2, e22), (p1, e12). The latter could make more sense as the second assignment for p2 could be more relevant than the second assignment for p1 given their weights. We have not empirically tried this approach, however.