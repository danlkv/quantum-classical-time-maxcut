import gurobipy as gb
import networkx as nx
SEED = 12

def maxcut(G: nx.Graph, max_time=10*60):
    p = gb.Model()
    p.setParam('TimeLimit', max_time)
    p.setParam('Threads', 1)
    p.setParam('Symmetry', 0)
    p.setParam('PreQLinearize', 2)

    vdict = {}
    for n in G.nodes:
        vdict[n] = p.addVar(name='v_'+str(n), vtype=gb.GRB.BINARY)
    C_i = [vdict[i] + vdict[j] - 2*vdict[i]*vdict[j] for i, j in G.edges]
    p.setObjective(sum(C_i), gb.GRB.MAXIMIZE)
    p.optimize()
    reverse_map = {v:k for k, v in vdict.items()}
    return p, [int(vdict[n].x) for n in G.nodes]


if __name__=='__main__':
    sizes = [432, 512]
    gaps = []
    for N in sizes:
        G = nx.random_regular_graph(3, N, seed=SEED)
        model, sol = maxcut(G, max_time=15*60)
        bound, cost = model.ObjBound, model.ObjVal
        gaps.append((bound-cost)/cost)
        

