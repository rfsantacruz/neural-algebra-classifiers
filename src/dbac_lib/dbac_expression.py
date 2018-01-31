import logging
import random
from anytree import RenderTree, PostOrderIter, Node
from pyparsing import Word, alphas, infixNotation, Keyword, opAssoc
import itertools
import numpy as np
import functools
from dbac_lib import dbac_util
import copy

logger = logging.getLogger(__name__)
OPS = ['NOT', 'AND', 'OR']
FUNC_DIC = {OPS[0]: np.logical_not, OPS[1]: np.logical_and, OPS[2]: np.logical_or}

# Expressions grammar
_G_TRUE = Keyword("TRUE")
_G_FALSE = Keyword("FALSE")
_G_OPERAND = _G_TRUE | _G_FALSE | Word(alphas)
_G_EXP = infixNotation(_G_OPERAND, [("NOT", 1, opAssoc.RIGHT), ("AND", 2, opAssoc.LEFT), ("OR", 2, opAssoc.LEFT)])


# binary tree creator
def _create_btree(tree_list):
    if isinstance(tree_list, (str, int, float, bool, type(None))):
        return Node(tree_list)
    elif isinstance(tree_list, (list, tuple, np.ndarray)):
        n, l, r = tree_list
        node, left_tree, right_tree = Node(n), _create_btree(l), _create_btree(r)
        left_tree.parent, right_tree.parent = node, node
        return node
    else:
        raise ValueError("Not well formatted expression tree".format(tree_list))


# lst tree creator
def _create_lst_tree(exp_tree):
    if exp_tree.is_leaf:
        return exp_tree.name
    else:
        return (exp_tree.name, _create_lst_tree(exp_tree.children[0]), _create_lst_tree(exp_tree.children[1]))


# parse from string
def str2exp_parse(exp_tree_str):
    exp_tree_list = _G_EXP.parseString(exp_tree_str).asList()[0]
    return _create_btree(exp_tree_list)


# parse from list
def list2exp_parse(exp_tree_list):
    return _create_btree(exp_tree_list)


# parser to list
def exp2list_parse(exp_tree):
    return _create_lst_tree(exp_tree)


# compare if two expressions are the same
def exp_tree_eq(exp_tree_a, exp_tree_b):
    if exp_tree_a.is_leaf and exp_tree_b.is_leaf:
        return exp_tree_a.name == exp_tree_b.name
    else:
        return (exp_tree_a.name == exp_tree_b.name and
                ((exp_tree_eq(exp_tree_a.children[0], exp_tree_b.children[0]) and exp_tree_eq(exp_tree_a.children[1],
                                                                                              exp_tree_b.children[
                                                                                                  1])) or
                 (exp_tree_eq(exp_tree_a.children[0], exp_tree_b.children[1]) and exp_tree_eq(exp_tree_a.children[1],
                                                                                              exp_tree_b.children[0]))))


# compute all ops between primitives using operations
def all_ops(primitives, operations, nogroup=False):
    exps = set()
    operations.sort()
    operands_gen = itertools.combinations(itertools.chain.from_iterable(primitives), 2) if nogroup else itertools.product(*primitives)
    for operands in operands_gen:
        operands = sorted(list(operands))
        for op in operations:
            exp = []
            if op == OPS[0]:
                exp = [(op, operand, None) for operand in operands]
            elif op in OPS[1:]:
                exp = [(op, operand_a, operand_b) for operand_a, operand_b in itertools.combinations(operands, 2)]
            else:
                raise ValueError("Not supported operation. Try {}.".format(OPS))
            exps.update(exp)
    return sorted(list(exps))


# evaluate allowed boolean operations
def eval_op(op, var_a, var_b=None, ops_dic={}):
    res = None
    if op == (OPS[0] or 0) and var_b is None:
        func = ops_dic.get(OPS[0], FUNC_DIC.get(OPS[0]))
        res = func(var_a)
    elif op == (OPS[1] or 1) and var_b is not None:
        func = ops_dic.get(OPS[1], FUNC_DIC.get(OPS[1]))
        res = func(var_a, var_b)
    elif op == (OPS[2] or 2) and var_b is not None:
        func = ops_dic.get(OPS[2], FUNC_DIC.get(OPS[2]))
        res = func(var_a, var_b)
    else:
        raise ValueError("Not Well formatted operations: ({}, {}, {})".format(op, var_a, var_b, ops_dic))
    return res


# evaluate expressions
def eval_exp(exp_tree, values_dic, ops_dic={}):
    # leaf node
    if exp_tree.is_leaf:
        return values_dic.get(exp_tree.name, None)
    # evaluate left tree
    left_res = eval_exp(exp_tree.children[0], values_dic, ops_dic)
    right_res = eval_exp(exp_tree.children[1], values_dic, ops_dic)
    return eval_op(exp_tree.name, left_res, right_res, ops_dic)


# check if the expression is valid
def check_exp(exp_tree, variables):
    for node in PostOrderIter(exp_tree):
        # check leaves are primitives
        if node.is_leaf:
            if node.name and node.name not in variables:
                logger.warning("Leaf node {} not in variables {}".format(node.name, variables))
                return False
        # check if the operations are binary and supported
        else:
            if node.name is None or node.name not in OPS or len(node.children) != 2:
                logger.warning("Not well-formatted node. Name={}, Children={}".format(node.name, len(node.children)))
                return False
    return True


# get variables from expression tree
def get_vars(exp_tree):
    return [node.name for node in PostOrderIter(exp_tree) if node.is_leaf and node.name is not None]


# get operators from expression tree
def get_ops(exp_tree):
    return [node.name for node in PostOrderIter(exp_tree) if not node.is_leaf]


# get terms from expression tree
def get_terms(exp_tree):
    return [node for node in PostOrderIter(exp_tree) if not node.is_leaf]


# generator of expressions in normal forms
def exp_gen(variables, complexity, form='D', allow_not=False, shuffle=True):
    def _combine(acc, el):
        node = Node(OPS[2] if form == 'D' else OPS[1])
        acc.parent = node
        el.parent = node
        return node

    # configure form
    assert form in ['C', 'D']
    # configure variables sampler
    var_ite = dbac_util.CycleIterator(variables, shuffle=shuffle)
    # sample operations
    terms = []
    for c in range(complexity):
        term_node = Node(OPS[1] if form == 'D' else OPS[2])
        for _ in range(2):
            if allow_not and np.random.rand() > 0.5:
                child_node = Node(OPS[0], parent=term_node)
                var_1 = Node(next(var_ite), parent=child_node)
                var_2 = Node(None, parent=child_node)
            else:
                child_node = Node(next(var_ite), parent=term_node)
        terms.append(term_node)
    # combine operations
    exp = functools.reduce(_combine, terms)
    yield exp


def exp_comb_gen(single_exp_list, complexity, form='D', allow_not=False, shuffle=True):
    def _combine(acc, el):
        node = Node(OPS[2] if form == 'D' else OPS[1])
        acc.parent = node
        el.parent = node
        return node

    # transform in list of expression trees
    single_exp_trees = [list2exp_parse(exp_lst) for exp_lst in single_exp_list]

    # combine according to complexity
    # for terms in itertools.combinations(single_exp_trees, complexity):
    exp_ite = dbac_util.CycleIterator(single_exp_trees, shuffle=shuffle)
    while True:
        terms = [copy.deepcopy(next(exp_ite)) for _ in range(complexity)]
        exp = functools.reduce(_combine, terms)
        if allow_not:
            sampled_leaves = [node for node in PostOrderIter(exp) if node.is_leaf and node.name is not None and np.random.rand() > 0.5]
            for node in sampled_leaves:
                not_op = Node(OPS[0], parent=node.parent)
                node.parent = not_op
                none_node = Node(None, parent=not_op)
        yield exp
