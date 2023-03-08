import matplotlib.pyplot as plt
import networkx as nx

G = nx.path_graph(5)
attrs = {0: {'attr1': 20, 'attr2': 'nothing'}, 1: {'attr2': 3}, 2: {'attr1': 42}, 3: {'attr3': 'hello'}, 4: {'attr1': 54, 'attr3': '33'}}
nx.set_node_attributes(G, attrs)

fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax)
nx.draw_networkx_edges(G, pos=pos, ax=ax)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):
    node = ind["ind"][0]
    xy = pos[node]
    annot.xy = xy
    node_attr = {'node': node}
    node_attr.update(G.nodes[node])
    text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
    annot.set_text(text)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = nodes.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
