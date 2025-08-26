import equinox as eqx


def save_model(model, filepath):
    eqx.tree_serialise_leaves(filepath, model)
