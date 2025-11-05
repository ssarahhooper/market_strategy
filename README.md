# market strategy
Sarah Hooper

## Dependencies
` pip install networkx matplotlib `

## Run
`python ./market_strategy.py <file_name>.gml --plot --interactive`

## Implementation

### Market Data Class
* G: the full bipartite network

* sellers and buyers: sets partitioning the nodes

* valuations: a nested dictionary storing buyer valuations for each seller

* prices: a dictionary of seller prices, updated each round

### read_market_from_gml
* Reads the .gml file using NetworkX

* Ensures nodes are labeled continuously (0..2n-1)

* Splits nodes into sellers (0..n-1) and buyers (n..2n-1)

* Extracts seller prices (default 0 if missing)

* Extracts valuations on edges (valuation, value, or weight)

* Validates structure and exits 


### preferred_sellers_of
given every buyer:
* Computes utility for each seller as u = valuation - price

* Identifies the maximum utility and the set of sellers getting it

* Returns both (max_utility, {best_sellers})


### build_preferred_graph
* Adds all buyers and sellers as nodes

* For each buyer, adds edges to all sellers that maximize their utility

* Stores buyer utilities and seller prices as node attributes


### maximum_matching_on_preferred
Computes a maximum bipartite matching:

* Converts the directed preferred graph into an undirected one for matching

* Uses networkx.algorithms.bipartite.matching.maximum_matching

* Returns a dictionary mapping buyer to seller

### alternating_reachable_sets
Identifies constricted sets:

* Starts from unmatched buyers

* Alternates between unmatched preferred edges and matched edges

* Collects reachable buyers (RB) and sellers (RS)

### compute_price_increment

Determines the minimum price increase needed to make a buyer in RB become indifferent toward a seller outside RS.

### market_clearing
main loop implementing market clearing:

* Builds preferred graph

* Finds maximum matching

* Checks if the market is fully matched (stop if yes)

* Computes constricted sets and epsilon

* Updates seller prices

* Repeats until equilibrium or maximum rounds reached



