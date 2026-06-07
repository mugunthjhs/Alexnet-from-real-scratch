from graphviz import Digraph
import numpy as np


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def preview(arr, max_items=6):

    if arr is None:
        return "None"

    flat = np.array(arr).flatten()

    return np.array2string(
        flat[:max_items],
        precision=3,
        separator=', '
    )


def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: LR | TB
    """

    assert rankdir in ['LR', 'TB']

    nodes, edges = trace(root)

    dot = Digraph(
        format=format,
        graph_attr={'rankdir': rankdir}
    )

    for n in nodes:

        dot.node(
            name=str(id(n)),
            label=(
                "{ "
                f"shape {n.shape}"
                f" | value {preview(n.data)}"
                f" | grad {preview(0 if n.grad is None else n.grad)}"
                " }"
            ),
            shape='record'
        )

        if n._op:
            dot.node(
                name=str(id(n)) + n._op,
                label=n._op
            )

            dot.edge(
                str(id(n)) + n._op,
                str(id(n))
            )

    for n1, n2 in edges:

        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot