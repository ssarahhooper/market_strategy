#!/usr/bin/env python3

import argparse
import math
import sys
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Iterable, Optional, List

try:
    import networkx as nx
except Exception as e:
    print("[ERROR] networkx is required. Install with: pip install networkx", file=sys.stderr)
    raise


@dataclass
class Market:
    G: nx.Graph
    sellers: Set[int]
    buyers: Set[int]
    valuations: Dict[int, Dict[int, float]]  # valuations[buyer][seller] -> v
    prices: Dict[int, float]                 # prices[seller] -> p




def read_market_from_gml(path: str) -> Market:
    try:
        G = nx.read_gml(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Failed to read GML: {e}", file=sys.stderr)
        sys.exit(2)

    if G.number_of_nodes() == 0:
        print("[ERROR] The graph is empty.", file=sys.stderr)
        sys.exit(2)

    # Expect integer-labeled nodes consistent with assignment: 0..2n-1
    # --- FIX: Convert string node labels like "0" -> 0 for consistency ---
    try:
        # If GML stored node labels as strings, convert them to integers
        mapping = {n: int(n) for n in G.nodes}
        G = nx.relabel_nodes(G, mapping)
    except Exception:
        pass

    node_ids = sorted(G.nodes)

    # Infer n assuming nodes are 0..2n-1 contiguous
    if node_ids[0] != 0 or node_ids[-1] != len(node_ids) - 1 or any(node_ids[i] != i for i in range(len(node_ids))):
        print("[ERROR] Expected nodes labeled 0..(2n-1) contiguously.", file=sys.stderr)
        sys.exit(2)

    if len(node_ids) % 2 != 0:
        print("[ERROR] Expected an even number of nodes (2n).", file=sys.stderr)
        sys.exit(2)

    n = len(node_ids) // 2
    sellers = set(range(0, n))
    buyers = set(range(n, 2 * n))

    # Initialize prices for sellers
    prices: Dict[int, float] = {}
    for s in sellers:
        price = 0.0
        # seller node may have attribute 'price'
        data = G.nodes[s]
        if 'price' in data:
            try:
                price = float(data['price'])
            except Exception:
                print(f"[WARN] Seller {s} has non-numeric price; defaulting to 0.")
                price = 0.0
        prices[s] = price

    # Extract valuations from edges
    valuations: Dict[int, Dict[int, float]] = defaultdict(dict)
    accepted_keys = ('valuation', 'value', 'weight')
    missing_any = True
    for u, v, data in G.edges(data=True):
        try:
            a, b = int(u), int(v)
        except Exception:
            print("[ERROR] Edge endpoints must be integers.", file=sys.stderr)
            sys.exit(2)
        # Ensure buyer-seller orientation regardless of original edge order
        if a in buyers and b in sellers:
            buyer, seller = a, b
        elif b in buyers and a in sellers:
            buyer, seller = b, a
        else:
            # Skip edges that don't connect across the bipartition
            continue

        val = None
        for k in accepted_keys:
            if k in data:
                try:
                    val = float(data[k])
                except Exception:
                    pass
                break
        if val is None:
            print(f"[WARN] Missing/invalid valuation on edge (buyer {buyer}, seller {seller}); treating as 0.")
            val = 0.0
        else:
            missing_any = False
        valuations[buyer][seller] = val

    if not valuations:
        print("[ERROR] No cross-partition edges with valuations were found.", file=sys.stderr)
        sys.exit(2)

    if missing_any:
        print("[INFO] Some valuations missing or invalid; treated as 0.")

    # Validate that each buyer has at least one incident valuation
    for b in buyers:
        if not valuations.get(b):
            print(f"[ERROR] Buyer {b} has no incident valuations.", file=sys.stderr)
            sys.exit(2)

    return Market(G=G, sellers=sellers, buyers=buyers, valuations=valuations, prices=prices)


def preferred_sellers_of(buyer: int, prices: Dict[int, float], valuations: Dict[int, Dict[int, float]]) -> Tuple[float, Set[int]]:
    best_u = -math.inf
    best: Set[int] = set()
    for s, v in valuations.get(buyer, {}).items():
        u = v - prices.get(s, 0.0)
        if u > best_u + 1e-12:
            best_u = u
            best = {s}
        elif abs(u - best_u) <= 1e-12:
            best.add(s)
    return (best_u if best_u != -math.inf else -1e18, best)


def build_preferred_graph(market: Market) -> nx.DiGraph:
    Gp = nx.DiGraph()
    for s in market.sellers:
        Gp.add_node(s, bipartite=0, kind='seller', price=market.prices[s])
    for b in market.buyers:
        maxu, bestS = preferred_sellers_of(b, market.prices, market.valuations)
        Gp.add_node(b, bipartite=1, kind='buyer', maxu=maxu)
        for s in bestS:
            Gp.add_edge(b, s)
    return Gp


def maximum_matching_on_preferred(Gp: nx.DiGraph, sellers: Set[int], buyers: Set[int]) -> Dict[int, int]:
    H = nx.Graph()
    H.add_nodes_from(sellers, bipartite=0)
    H.add_nodes_from(buyers, bipartite=1)
    for b in buyers:
        for s in Gp.successors(b):
            H.add_edge(b, s)
    # NetworkX maximum_matching returns dict mapping node->mate for matched nodes
    M = nx.algorithms.bipartite.matching.maximum_matching(H, top_nodes=buyers)
    # Convert to buyer->seller mapping only
    buyer_to_seller = {b: s for b, s in M.items() if b in buyers}
    return buyer_to_seller


def alternating_reachable_sets(Gp: nx.DiGraph, matching: Dict[int, int], buyers: Set[int]) -> Tuple[Set[int], Set[int]]:
    matched_seller_of = {s: b for b, s in matching.items()}
    matched_buyer_of = matching.copy()

    start_buyers = {b for b in buyers if b not in matched_buyer_of}
    RB: Set[int] = set(start_buyers)
    RS: Set[int] = set()

    q = deque(start_buyers)
    while q:
        b = q.popleft()
        # along preferred edges to sellers
        for s in Gp.successors(b):
            if s not in RS:
                RS.add(s)
                # along matched edge back to a buyer (if any)
                if s in matched_seller_of:
                    b2 = matched_seller_of[s]
                    if b2 not in RB:
                        RB.add(b2)
                        q.append(b2)
    return RB, RS


def compute_price_increment(market: Market, Gp: nx.DiGraph, RB: Set[int], RS: Set[int]) -> float:
    eps = math.inf
    for b in RB:
        maxu = Gp.nodes[b].get('maxu', None)
        if maxu is None:
            maxu, _ = preferred_sellers_of(b, market.prices, market.valuations)
        for s_prime, v in market.valuations[b].items():
            if s_prime in RS:
                continue
            # current net utility to s_prime under current prices
            u_sp = v - market.prices.get(s_prime, 0.0)
            # epsilon needed so that u_sp becomes equal to current maxu
            # i.e., (v - (p_s'+eps)) == maxu  => eps = v - p_s' - maxu
            eps_candidate = (maxu - u_sp)
            if eps_candidate > 1e-12:
                eps = min(eps, eps_candidate)
    if eps is math.inf or eps <= 0:
        # Fallback tiny step to ensure progress
        eps = 1e-3
    return eps


def update_prices(prices: Dict[int, float], RS: Set[int], eps: float) -> None:
    for s in RS:
        prices[s] = prices.get(s, 0.0) + eps


# Plotting

def plot_preferred_graph(round_id: int, market: Market, Gp: nx.DiGraph, matching: Dict[int, int]) -> None:
    # Build an undirected helper to draw
    H = nx.Graph()
    H.add_nodes_from(market.sellers)
    H.add_nodes_from(market.buyers)
    for b in market.buyers:
        for s in Gp.successors(b):
            H.add_edge(b, s)

    # Positions: sellers on top row, buyers on bottom row
    n = len(market.sellers)
    x_s = {s: i for i, s in enumerate(sorted(market.sellers))}
    x_b = {b: i for i, b in enumerate(sorted(market.buyers))}
    pos = {}
    for s, i in x_s.items():
        pos[s] = (i, 1)
    for b, i in x_b.items():
        pos[b] = (i, 0)

    plt.figure(figsize=(max(8, n*0.9), 6))
    nx.draw_networkx_nodes(H, pos, nodelist=sorted(market.sellers), node_shape='s', node_size=900)
    nx.draw_networkx_nodes(H, pos, nodelist=sorted(market.buyers), node_shape='o', node_size=900)
    nx.draw_networkx_labels(H, pos, labels={s: f"S{s}\n$p$={market.prices[s]:.2f}" for s in market.sellers})
    nx.draw_networkx_labels(H, pos, labels={b: f"B{b}" for b in market.buyers})

    # Preferred edges as thin lines
    nx.draw_networkx_edges(H, pos, width=1.5, alpha=0.6)

    # Highlight matched edges thicker
    matched_edges = [(b, s) for b, s in matching.items()]
    nx.draw_networkx_edges(H, pos, edgelist=matched_edges, width=3.5)

    plt.title(f"Preferred-seller graph — Round {round_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)



def print_round(round_id: int, market: Market, Gp: nx.DiGraph, matching: Dict[int, int], RB: Set[int], RS: Set[int], eps: Optional[float]) -> None:
    print(f"\n=== Round {round_id} ===")
    # Preferred edges summary
    edges_summary = []
    for b in sorted(market.buyers):
        bestS = list(Gp.successors(b))
        edges_summary.append(f"B{b} -> {sorted(bestS)} (u*={Gp.nodes[b]['maxu']:.3f})")
    print("Preferred edges:")
    for line in edges_summary:
        print("  ", line)

    print("Matching (buyer->seller):", {f"B{b}": f"S{s}" for b, s in sorted(matching.items())})

    print("Constricted sets / reachable:")
    print("  Buyers RB:", sorted(RB))
    print("  Sellers RS:", sorted(RS))

    print("Prices:")
    print({f"S{s}": f"{market.prices[s]:.3f}" for s in sorted(market.sellers)})

    if eps is not None:
        print(f"Price increment eps = {eps:.6f} applied to sellers in RS")

# driver

def market_clearing(market: Market, do_plot: bool=False, interactive: bool=False, max_rounds: int=5000) -> Tuple[Dict[int, int], Dict[int, float], int]:
    round_id = 0
    while round_id < max_rounds:
        round_id += 1
        Gp = build_preferred_graph(market)
        matching = maximum_matching_on_preferred(Gp, market.sellers, market.buyers)

        if do_plot:
            plot_preferred_graph(round_id, market, Gp, matching)

        # Check if matching covers all buyers (perfect)
        if len(matching) == len(market.buyers):
            if interactive:
                print_round(round_id, market, Gp, matching, set(), set(), None)
            return matching, market.prices, round_id

        # Compute constricted (reachable) sets via alternating paths
        RB, RS = alternating_reachable_sets(Gp, matching, market.buyers)

        # Compute epsilon and raise prices of RS
        eps = compute_price_increment(market, Gp, RB, RS)
        update_prices(market.prices, RS, eps)

        if interactive:
            print_round(round_id, market, Gp, matching, RB, RS, eps)

    print("[WARN] Reached max rounds without full clearing. Returning best-so-far matching.")
    return matching, market.prices, round_id


def main():
    parser = argparse.ArgumentParser(description="Market Clearing on Bipartite Graph (GML)")
    parser.add_argument('gml_path', help='Path to market.gml')
    parser.add_argument('--plot', action='store_true', help='Plot preferred-seller graph each round')
    parser.add_argument('--interactive', action='store_true', help='Print round-by-round status')
    args = parser.parse_args()

    market = read_market_from_gml(args.gml_path)

    matching, prices, rounds = market_clearing(market, do_plot=args.plot, interactive=args.interactive)

    # Final report
    print("\n=== RESULT ===")
    print(f"Rounds: {rounds}")
    print("Final prices:")
    for s in sorted(market.sellers):
        print(f"  Seller S{s}: price = {prices[s]:.4f}")

    print("Final matching (buyer -> seller):")
    for b in sorted(market.buyers):
        mate = matching.get(b)
        if mate is None:
            print(f"  Buyer B{b}: UNMATCHED")
        else:
            print(f"  Buyer B{b}: S{mate}")

    if args.plot:
        # Plot once more with the final preferred graph under final prices
        Gp_final = build_preferred_graph(Market(market.G, market.sellers, market.buyers, market.valuations, prices))
        final_matching = maximum_matching_on_preferred(Gp_final, market.sellers, market.buyers)
        plot_preferred_graph(rounds, Market(market.G, market.sellers, market.buyers, market.valuations, prices), Gp_final, final_matching)
        if plt is not None:
            plt.show()


if __name__ == '__main__':
    main()
