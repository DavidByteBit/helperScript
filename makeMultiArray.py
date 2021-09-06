from Compiler.types import *
from Compiler.library import *
from Compiler import ml

import json

# program.options_from_args()

populator_len = 1000

bounds1 = [10, 5, 2, 5]
bounds2 = [10, 30]
bounds3 = [200]

# bounds1 = json.loads(program.args[1])  # 4d
# bounds2 = json.loads(program.args[2])  # 2d
# bounds3 = json.loads(program.args[3])  # 1d

sum_of_bounds1 = sum(bounds1)
sum_of_bounds2 = sum(bounds2)
sum_of_bounds3 = sum(bounds3)

a = sfix.Tensor(bounds1)
b = sfix.Tensor(bounds2)
c = sfix.Tensor(bounds3)

populator = sfix.Array(populator_len)


@for_range(populator_len)
def _(i):
    populator[i] = sfix(i)


a.assign_vector(populator.get_part_vector(0, sum_of_bounds1))
b.assign_vector(populator.get_part_vector(sum_of_bounds1, sum_of_bounds1 + sum_of_bounds2))
c.assign_vector(
    populator.get_part_vector(sum_of_bounds1 + sum_of_bounds2, sum_of_bounds1 + sum_of_bounds2 + sum_of_bounds3))

print_ln("%s", a.reveal_nested())
print_ln("%s", b.reveal_nested())
print_ln("%s", c.reveal_nested())

# ####### TEST 1: use the bounds to populate arrays 1 element at a time #######
# sum_of_bounds1 = sum(bounds1)
# @for_range(sum_of_bounds1)
# def _(i):
#     a.
