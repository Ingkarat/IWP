import sys
import os
import ast
import astor
import pickle

import config
from equality import Impl
from typing import List, Tuple
import pprint2
from itertools import starmap


class Simplifier(ast.NodeTransformer):
    """
    - Collapse nested sequences of AND/OR BoolOp, e.g., (a and b) and (c and d).
    - Rewrite equality.Impl node `p => q === not p or q`
    - Collapse double negation, e.g.,  not (x not in [1,2,3])
    - Reduce simple BoolOp negations, e.g., not(a or b or c), not(not a) etc.
    - Convert tuple to list
    - Rewrite `x <= min(a, b) === x <= a and x <= b`
    """

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        def _expand(kind, n: ast.AST) -> List[ast.AST]:
            if isinstance(n, ast.BoolOp) and isinstance(n.op, kind):
                return [v for m in n.values for v in _expand(kind, m)]
            return [n]

        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            values = _expand(ast.And, node)
            return ast.BoolOp(op=ast.And(), values=values)
        if isinstance(node.op, ast.Or):
            values = _expand(ast.Or, node)
            return ast.BoolOp(op=ast.Or(), values=values)
        if isinstance(node.op, Impl):
            assert len(node.values) == 2
            return self.visit(
                ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=node.values[0]),
                        node.values[1],
                    ],
                )
            )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(
                node.operand.op, ast.Not
            ):
                return node.operand.operand
            if isinstance(node.operand, ast.Compare):
                if len(node.operand.ops) == 1:
                    op = node.operand.ops[0]
                    comparators = node.operand.comparators
                    if isinstance(op, ast.NotEq):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Eq()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Eq):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.NotEq()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.NotIn):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.In()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.In):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.NotIn()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.IsNot):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Is()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Is):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.IsNot()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Lt):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.GtE()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.LtE):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Gt()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.Gt):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.LtE()],
                            comparators=comparators,
                        )
                    if isinstance(op, ast.GtE):
                        return ast.Compare(
                            left=node.operand.left,
                            ops=[ast.Lt()],
                            comparators=comparators,
                        )
            if isinstance(node.operand, ast.BoolOp):
                values = [
                    self.visit(ast.UnaryOp(op=ast.Not(), operand=v))
                    for v in node.operand.values
                ]
                if isinstance(node.operand.op, ast.And):
                    return ast.BoolOp(op=ast.Or(), values=values)
                if isinstance(node.operand.op, ast.Or):
                    return ast.BoolOp(op=ast.And(), values=values)
                return node
            return node
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        comparisons = list(
            zip([node.left] + node.comparators, node.ops, node.comparators)
        )

        def min_max(node):
            assert len(node.ops) == 1 and len(node.comparators) == 1
            o = node.ops[0]
            c = node.comparators[0]
            if (
                isinstance(o, (ast.Lt, ast.LtE))
                and isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "min"
            ):
                values = [
                    ast.Compare(left=node.left, ops=[o], comparators=[c])
                    for c in c.args
                ]
                return ast.BoolOp(op=ast.And(), values=values)
            if (
                isinstance(o, (ast.Gt, ast.GtE))
                and isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "max"
            ):
                values = [
                    ast.Compare(left=node.left, ops=[o], comparators=[c])
                    for c in c.args
                ]
                return ast.BoolOp(op=ast.And(), values=values)
            return node

        values = [
            min_max(ast.Compare(left=l, ops=[o], comparators=[c]))
            for l, o, c in comparisons
        ]
        if len(values) == 1:
            return values[0]
        return ast.BoolOp(op=ast.And(), values=values)

    def visit_Tuple(self, node):
        return ast.List(elts=node.elts)

    def visit_Constant(self, node):
        return node


class JSONPrinter(ast.NodeTransformer):
    """
    Convert boolean constraint into a JSON schema (python dict)
    """

    def _dump(self, node: ast.AST):
        #src = astor.to_source(ast.fix_missing_locations(node)).replace("\n", " ")
        src = ast.unparse(node)
        #if isinstance(node.operand, ast.Call) and node.operand.func.id == "hasattr":
        #print("DUMP:", ast.dump(node),"\n") 
        return ast.Dict(
            keys=[ast.Constant(value="XXX TODO XXX")],
            values=[ast.Constant(value=src)],
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Dict:
        precond = isinstance(node.op, (ast.And, ast.Or))
        if not precond:
            # TODO, Don't know what to do yet
            return self._dump(node)

        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            values = ast.List(elts=node.values)
            return ast.Dict(keys=[ast.Constant(value="allOf")], values=[values])
        if isinstance(node.op, ast.Or):
            values = ast.List(elts=node.values)
            return ast.Dict(keys=[ast.Constant(value="anyOf")], values=[values])
        assert False

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Dict:
        precond = (
            isinstance(node.operand, ast.Name)
            or (
                isinstance(node.operand, ast.Attribute)
                and isinstance(node.operand.value, ast.Name)
                and node.operand.value.id == "self"
            )
            or (
                isinstance(node.operand, ast.Call)
                and (
                    (
                        isinstance(node.operand.func, ast.Attribute)
                        and isinstance(node.operand.func.value, ast.Name)
                        and node.operand.func.attr == "issparse"
                    )
                    or (
                        isinstance(node.operand.func, ast.Name)
                        and node.operand.func.id == "issparse"
                    )
                )
            )
            or (
                isinstance(node.operand, ast.Call)
                and isinstance(node.operand.func, ast.Name)
                and node.operand.func.id == "isinstance"
            )
        )
        #UnaryOp(op=Not(), operand=Call(func=Name(id='hasattr', ctx=Load()), args=[Name(id='self', ctx=Load()), Constant(value='sparse')], keywords=[]))
        #Call(func=Name(id='hasattr', ctx=Load()), args=[Name(id='self', ctx=Load()), Constant(value='sparse')], keywords=[])

        if not precond:
            # TODO, Don't kwno what to do at this point
            return self._dump(node)

        if isinstance(node.operand, ast.Call):
            if (
                isinstance(node.operand.func, ast.Name)
                and node.operand.func.id == "isinstance"
            ):
                # Push negation inside the type
                return self.visit_Call(node.operand, negation=True)
            else:
                # TODO: We assume that a call to issparse is always for `X/issparse`
                return ast.Dict(
                    keys=[ast.Constant(value="type"), ast.Constant(value="laleNot")],
                    values=[ast.Constant(value="object"), ast.Constant("X/isSparse")],
                )

        # TODO: we assume that the negation of a variable alone (or self.param) is used to refer to the value False for the corresponding parameter
        if isinstance(node.operand, ast.Name):
            name = node.operand.id
        if isinstance(node.operand, ast.Attribute):
            name = node.operand.attr

        body = ast.Dict(
            keys=[ast.Constant(value=name)],
            values=[
                ast.Dict(
                    keys=[ast.Constant(value="enum")],
                    values=[ast.List(elts=[ast.Constant(value=False)])],
                )
            ],
        )

        return ast.Dict(
            keys=[ast.Constant(value="type"), ast.Constant(value="properties")],
            values=[ast.Constant(value="object"), body],
        )

    def visit_Name(self, node: ast.Name) -> ast.Dict:
        # TODO: we assume that a variable alone is used to refer to the value True for the corresponding parameter
        body = ast.Dict(
            keys=[ast.Constant(value=node.id)],
            values=[
                ast.Dict(
                    keys=[ast.Constant(value="enum")],
                    values=[ast.List(elts=[ast.Constant(value=True)])],
                )
            ],
        )
        return ast.Dict(
            keys=[ast.Constant(value="type"), ast.Constant(value="properties")],
            values=[ast.Constant(value="object"), body],
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.Dict:
        # TODO: we assume that a variable alone is used to refer to the value True for the corresponding parameter
        body = ast.Dict(
            keys=[ast.Constant(value=node.attr)],
            values=[
                ast.Dict(
                    keys=[ast.Constant(value="enum")],
                    values=[ast.List(elts=[ast.Constant(value=True)])],
                )
            ],
        )
        return ast.Dict(
            keys=[ast.Constant(value="type"), ast.Constant(value="properties")],
            values=[ast.Constant(value="object"), body],
        )

    def visit_Compare(self, node: ast.Compare) -> ast.Dict:

        comparisons = list(
            zip([node.left] + node.comparators, node.ops, node.comparators)
        )

        precond = all(
            list(
                starmap(
                    lambda a, op, b: (
                        (
                            isinstance(a, ast.Name)
                            or (
                                isinstance(a, ast.Attribute)
                                and isinstance(a.value, ast.Name)
                                and a.value.id == "self"
                            )
                        )
                        and (
                            isinstance(b, (ast.Constant, ast.List))
                            or (
                                isinstance(b, ast.Subscript)
                                and isinstance(b.value, ast.Attribute)
                                and isinstance(b.value.value, ast.Name)
                                and b.value.value.id == "X"
                                and b.value.attr == "shape"
                                and isinstance(op, (ast.GtE, ast.LtE, ast.Lt, ast.Gt))
                            )
                        )
                    )
                    or (
                        (
                            isinstance(b, ast.Name)
                            or (
                                isinstance(b, ast.Attribute)
                                and isinstance(b.value, ast.Name)
                                and b.value.id == "self"
                            )
                        )
                        and (
                            isinstance(a, ast.Constant)
                            or (
                                isinstance(a, ast.Subscript)
                                and isinstance(a.value, ast.Attribute)
                                and isinstance(a.value.value, ast.Name)
                                and a.value.value.id == "X"
                                and a.value.attr == "shape"
                                and isinstance(op, (ast.LtE, ast.Lt, ast.Gt))
                            )
                        )
                        and isinstance(op, (ast.GtE, ast.LtE, ast.Gt, ast.Lt))
                    ),
                    comparisons,
                )
            )
        )

        if not precond:
            # TODO, Don't know what to do yet
            return self._dump(node)

        flip_op = {
            ast.GtE: ast.LtE,
            ast.LtE: ast.GtE,
            ast.Gt: ast.Lt,
            ast.Lt: ast.Gt,
        }

        def flip(a, op, b):
            if isinstance(a, ast.Constant) and isinstance(b, (ast.Name, ast.Attribute)):
                return b, flip_op[type(op)](), a
            return a, op, b

        def get_range(comp):
            var, op, comparator = comp
            name = var.id if isinstance(var, ast.Name) else var.attr

            if isinstance(op, (ast.NotEq, ast.Eq, ast.Is, ast.IsNot)):
                comparator = ast.List(elts=node.comparators)
            if isinstance(op, (ast.In, ast.Eq, ast.Is)):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(keys=[ast.Constant(value="enum")], values=[comparator])
                    ],
                )
            if isinstance(op, (ast.NotIn, ast.NotEq, ast.IsNot)):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[ast.Constant(value="not")],
                            values=[
                                ast.Dict(
                                    keys=[ast.Constant(value="enum")],
                                    values=[comparator],
                                )
                            ],
                        )
                    ],
                )
            if isinstance(op, ast.GtE):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[
                                ast.Constant(value="type"),
                                ast.Constant(value="minimum"),
                            ],
                            values=[ast.Constant(value="number"), comparator],
                        )
                    ],
                )
            if isinstance(op, ast.Gt):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[
                                ast.Constant(value="type"),
                                ast.Constant(value="minimum"),
                                ast.Constant(value="exclusiveMinimum"),
                            ],
                            values=[
                                ast.Constant(value="number"),
                                comparator,
                                ast.Constant(value=True),
                            ],
                        )
                    ],
                )
            if isinstance(op, ast.LtE):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[
                                ast.Constant(value="type"),
                                ast.Constant(value="maximum"),
                            ],
                            values=[ast.Constant(value="number"), comparator],
                        )
                    ],
                )
            if isinstance(op, ast.Lt):
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[
                                ast.Constant(value="type"),
                                ast.Constant(value="maximum"),
                                ast.Constant(value="exclusiveMaximum"),
                            ],
                            values=[
                                ast.Constant(value="number"),
                                comparator,
                                ast.Constant(value=True),
                            ],
                        )
                    ],
                )

            if isinstance(comparator, ast.Subscript):
                if comparator.slice.value == 0:
                    v = "X/maxItems"
                if comparator.slice.value == 1:
                    v = "X/items/maxItems"
                if isinstance(op, (ast.LtE, ast.Lt)):
                    k = ast.Constant(value="laleMaximum")
                if isinstance(op, (ast.GtE, ast.Gt)):
                    k = ast.Constant(value="laleMinimum")
                body = ast.Dict(
                    keys=[ast.Constant(value=name)],
                    values=[
                        ast.Dict(
                            keys=[k],
                            values=[ast.Constant(value=v)],
                        )
                    ],
                )

            return ast.Dict(
                keys=[ast.Constant(value="type"), ast.Constant(value="properties")],
                values=[ast.Constant(value="object"), body],
            )

        comparisons = list(starmap(flip, comparisons))
        ranges = list(map(get_range, comparisons))
        if len(ranges) == 1:
            return ranges[0]
        else:
            values = ast.List(elts=ranges)
            return ast.Dict(keys=[ast.Constant(value="allOf")], values=[values])

    def visit_Call(self, node: ast.Call, negation: bool = False) -> ast.Dict:
        precond = (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and (
                isinstance(node.args[0], (ast.Name, ast.Constant))
                or (
                    isinstance(node.args[0], ast.Attribute)
                    and isinstance(node.args[0].value, ast.Name)
                    and node.args[0].value.id == "self"
                )
            )
            and (
                (
                    isinstance(node.args[1], ast.Attribute)
                    and (node.args[1].attr in ["Number", "Real", "Integral"])
                )
                or (
                    isinstance(node.args[1], ast.Name)
                    and node.args[1].id in ["str", "bool", "int", "float"]
                )
            )
        )
        if not precond:
            # TODO, Don't know what to do yet
            return self._dump(node)
        if isinstance(node.args[0], ast.Constant):
            name = node.args[0].value
        if isinstance(node.args[0], ast.Name):
            name = node.args[0].id
        if isinstance(node.args[0], ast.Attribute):
            name = node.args[0].attr
        if isinstance(node.args[1], ast.Attribute):
            if node.args[1].attr in ["Number", "Real"]:
                ty = "number"
            elif node.args[1].attr == "Integral":
                ty = "integer"
        elif isinstance(node.args[1], ast.Name):
            if node.args[1].id == "str":
                ty = "string"
            elif node.args[1].id == "bool":
                ty = "boolean"
            elif node.args[1].id == "int":
                ty = "integer"
            elif node.args[1].id == "float":
                ty = "number"
        if negation:
            body = ast.Dict(
                keys=[ast.Constant(value=name)],
                values=[
                    ast.Dict(
                        keys=[ast.Constant(value="not")],
                        values=[
                            ast.Dict(
                                keys=[ast.Constant(value="type")],
                                values=[ast.Constant(value=ty)],
                            )
                        ],
                    )
                ],
            )
        else:
            body = ast.Dict(
                keys=[ast.Constant(value=name)],
                values=[
                    ast.Dict(
                        keys=[ast.Constant(value="type")],
                        values=[ast.Constant(value=ty)],
                    )
                ],
            )
        return ast.Dict(
            keys=[ast.Constant(value="type"), ast.Constant(value="properties")],
            values=[ast.Constant(value="object"), body],
        )

    def visit_Subscript(self, node):
        return self._dump(node)



def exn_message(e):
    return astor.to_source(ast.fix_missing_locations(e)).replace("\n", " ")

def rpl(x):
    return x.replace(config.PATH_SHORTENING,"")

contain = False
def containFilteredASTConstant(wp):
  cc = 0
  if isinstance(wp,ast.Constant):
    if isinstance(wp.value,str):
        #print(">",ast.dump(wp))
        if "FILTERED WP. IT HAS" in wp.value:
            global contain
            contain = True
  else:
    for attr in wp.__dict__.keys():
      if isinstance(wp.__dict__[attr],ast.AST):
        containFilteredASTConstant(wp.__dict__[attr])
      elif isinstance(wp.__dict__[attr],list):
        for elt in wp.__dict__[attr]:
          containFilteredASTConstant(elt)

# Take path of the .pkl file and output path
def to_json(pkl_file, output_path):

    # input file in pkl format
    # output path. JSON format in Python file

    elts = []
    with open(pkl_file, "rb") as f3:
        main_wps = pickle.load(f3)
        nub = 0
        for wp in main_wps:
            t = main_wps[wp]
            #print(ast.dump(t))
            #print("\n",pprint2.pprint_top(t))
            global contain
            contain = False
            containFilteredASTConstant(t)
            if contain:
                pass
            else:
                t = Simplifier().visit(t)
                t = JSONPrinter().visit(t)
                if isinstance(t, ast.Dict):  # Some pre-condition are just False (or True)
                    nub += 1
                    #print(nub)
                    elts.append(
                        ast.Dict(
                            keys=[ast.Constant("description")] + t.keys,
                            values=[
                                ast.Constant(
                                    f"From {rpl(wp[1])}, Exception: {exn_message(wp[0])}"
                                )
                            ]
                            + t.values,
                        )
                    )


    with open(output_path, "w") as f:
        print(astor.to_source(ast.List(elts=elts)), file=f)


if __name__ == "__main__":
    print("> json_printer.py: NOTHING IS HERE") 