import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

draw_real_net=True

if draw_real_net:
    df = pd.read_csv("/Users/lyh/c-elegans-control-master/data.csv", header=None)

    G = nx.from_pandas_adjacency(df,nx.DiGraph)
    G = nx.DiGraph(G)
    print(nx.info(G))
    mapping = {
        0: 'ADELa1',
        1: 'ADELb1',
        2: 'ADELa2',
        3: 'ADELb2',
        4: 'ADELa3',
        5: 'ADELb3',
        6: 'ADELa4',
        7: 'ADELb4',
        8: 'ADELa5',
        9: 'ADELb5',
        10: 'MDL05a1',
        11: 'MDL05b1',
        12: 'MDL05a2',
        13: 'MDL05b2',
        14: 'MDL05a3',
        15: 'MDL05b3',
        16: 'MDL05a4',
        17: 'MDL05b4',
        18: 'MDL05a5',
        19: 'MDL05b5',
        20: 'AS01a1',
        21: 'AS01b1',
        22: 'AS01a2',
        23: 'AS01b2',
        24: 'AS01a3',
        25: 'AS01b3',
        26: 'AS01a4',
        27: 'AS01b4',
        28: 'AS01a5',
        29: 'AS01b5'}
    G = nx.relabel_nodes(G, mapping)
    pos = nx.circular_layout(G)
    nx.draw(G, pos,arrows=True, arrowsize=20, arrowstyle='-|>', node_size=1000, node_color=range(30), cmap=plt.cm.Blues)
    labels = nx.draw_networkx_labels(G, pos)
    plt.show()




else:
    g = nx.DiGraph()
    g.clear()
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)
    g.add_node(6)
    g.add_node(7)
    g.add_node(8)
    g.add_node('s')

    g.add_edges_from([(1,5)], color='red')
    g.add_edges_from([(1,7)], color='red')
    g.add_edges_from([(2,4)], color='red')
    g.add_edges_from([(3,6)], color='red')
    g.add_edges_from([(4,7)], color='red')
    g.add_edges_from([(5,1)], color='red')
    g.add_edges_from([(5,7)], color='red')
    g.add_edges_from([(6,4)], color='red')
    g.add_edges_from([(8,2)], color='red')

    pic=plt.subplot(2,2,1)
    g1 = nx.DiGraph(g)
    g1.add_weighted_edges_from([('s',1,1)], color='red')
    g1.add_weighted_edges_from([('s',5,1)], color='red')
    #DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75), (1,4,0.3)]) # 添加带权值的边
    nx.draw(g1, pos=nx.circular_layout(g), nodecolor='b', edge_color=['b','b','b','b','b','b','b','b','b','r','r'])
    nx.draw_networkx_labels(g1,pos=nx.circular_layout(g1))
    nx.draw_networkx_edge_labels(g1, pos=nx.circular_layout(g1),
                              label_pos=0.5, font_size=8,
                            edge_labels={('s',1):1,('s',5):1}, rotate=True)
    pic.text(0, -1.5, '(a)', ha='center')


    pic=plt.subplot(2,2,2)
    g2 = nx.DiGraph(g)
    g2.add_weighted_edges_from([('s',1,1)], color='red')
    g2.add_weighted_edges_from([('s',5,-1)], color='red')
    #DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75), (1,4,0.3)]) # 添加带权值的边
    nx.draw(g2, pos=nx.circular_layout(g), nodecolor='b', edge_color=['b','b','b','b','b','b','b','b','b','r','r'])
    nx.draw_networkx_labels(g2,pos=nx.circular_layout(g2))
    nx.draw_networkx_edge_labels(g2, pos=nx.circular_layout(g2),
                              label_pos=0.5, font_size=8,
                            edge_labels={('s',1):1,('s',5):-1}, rotate=True)
    pic.text(0, -1.5, '(b)', ha='center')

    pic=plt.subplot(2,2,3)
    g3 = nx.DiGraph(g)
    g3.add_weighted_edges_from([('s',1,-2)], color='red')
    g3.add_weighted_edges_from([('s',2,1)], color='red')
    g3.add_weighted_edges_from([('s',3,1)], color='red')
    g3.add_weighted_edges_from([('s',4,1)], color='red')
    g3.add_weighted_edges_from([('s',5,-3)], color='red')
    g3.add_weighted_edges_from([('s',6,1)], color='red')
    g3.add_weighted_edges_from([('s',7,1)], color='red')
    g3.add_weighted_edges_from([('s',8,1)], color='red')
    #DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75), (1,4,0.3)]) # 添加带权值的边
    nx.draw(g3, pos=nx.circular_layout(g), nodecolor='b', edge_color=['b','b','b','b','b','b','b','b','b','r','r','r','r','r','r','r','r'])
    nx.draw_networkx_labels(g3,pos=nx.circular_layout(g3))
    nx.draw_networkx_edge_labels(g3, pos=nx.circular_layout(g3),
                              label_pos=0.5, font_size=8,
                            edge_labels={('s',1):-2,('s',2):1,('s',3):1,('s',4):1,('s',5):-3,('s',6):1,('s',7):1,('s',8):1},
                            rotate=True)
    pic.text(0, -1.5, '(c)', ha='center')

    pic=plt.subplot(2,2,4)
    g4 = nx.DiGraph(g)
    g4.add_weighted_edges_from([('s',3,-1)], color='red')
    g4.add_weighted_edges_from([('s',6,-1)], color='red')
    #DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75), (1,4,0.3)]) # 添加带权值的边
    nx.draw(g4, pos=nx.circular_layout(g), nodecolor='b', edge_color=['b','b','b','b','b','b','b','b','b','r','r'])
    nx.draw_networkx_labels(g4,pos=nx.circular_layout(g4))
    nx.draw_networkx_edge_labels(g4, pos=nx.circular_layout(g4),
                              label_pos=0.5, font_size=8,
                            edge_labels={('s',3):-1,('s',6):-1}, rotate=True)
    pic.text(0, -1.5, '(d)', ha='center')

    plt.show()
