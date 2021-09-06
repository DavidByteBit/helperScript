from Compiler.types import *
from Compiler.library import *

from functools import reduce
import operator


populator_len = 1000

bounds1 = [10, 5, 2, 5]  # 4d Tensor
bounds2 = [10, 50]  # 2d Tensor

prod_of_bounds1 = reduce(operator.mul, bounds1)  # Tell us how many elements are in tensor
prod_of_bounds2 = reduce(operator.mul, bounds2)  # Tell us how many elements are in tensor

a = sfix.Tensor(bounds1)
b = sfix.Tensor(bounds2)

populator = sfix.Array(populator_len) # Make 1d Array - will be used to populate tensors


@for_range(populator_len)
def _(i):
    populator[i] = sfix(i)


print_ln("%s", populator.reveal())

# Works exactly how you may think - takes the flat 'populator' array and gives it structure in the tensor
a.assign_vector(populator.get_part_vector(0, prod_of_bounds1))
b.assign_vector(populator.get_part_vector(prod_of_bounds1, prod_of_bounds2))

print_ln("%s", a.reveal_nested())
print_ln("%s", b.reveal_nested())

