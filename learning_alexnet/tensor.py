import numpy as np
from typing import Union, Tuple, Optional, List

TensorLike = Union[
    'Tensor',
    int,
    float,
    list,
    tuple,
    np.ndarray,
    np.number,
]

class Tensor:

    def __init__(
        self,
        data: Union['Tensor', np.ndarray, float, int, list],
        requires_grad: bool = False,
        is_leaf: bool = True,
        dtype: np.dtype = np.float32,
    ):
        if not isinstance(requires_grad, bool):
            raise TypeError(
                f"requires_grad must be bool, got {type(requires_grad).__name__}"
            )
        if not isinstance(is_leaf, bool):
            raise TypeError(
                f"is_leaf must be bool, got {type(is_leaf).__name__}"
            )

        if isinstance(data, Tensor):
            data = data.data

        try:
            self.data = np.array(data, dtype=dtype)
        except Exception:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = is_leaf
        self._backward = lambda: None
        self._prev = set()
        self._op = ""

    def __repr__(self) -> str:
        return (
            f"Tensor({self.data.tolist()}, shape={self.shape}, "
            f"dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    # ------------------------------------------------------------------ #
    #  Core utilities                                                      #
    # ------------------------------------------------------------------ #

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------------ #
    #  Autograd engine                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sum_to_shape(
        grad: np.ndarray,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad


    def backward(
        self,
        grad: Optional[TensorLike] = None
    ) -> None:
        if not self.requires_grad:
            raise RuntimeError(
                "Tensor does not require grad"
            )
        if grad is not None:
            if isinstance(grad, Tensor):
                grad = grad.data
            else:
                grad = np.asarray(
                    grad,
                    dtype=self.dtype
                )
        if self.data.size == 1:

            if grad is None:
                grad = np.ones_like(
                    self.data
                )
        else:
            if grad is None:
                raise RuntimeError(
                    "grad must be specified "
                    "for non-scalar tensors"
                )
            if grad.shape != self.shape:
                raise ValueError(
                    f"grad shape {grad.shape} "
                    f"does not match tensor shape "
                    f"{self.shape}"
                )
        self.grad = grad
        topo: List['Tensor'] = []
        visited: set = set()
        def _build_topo(
            v: 'Tensor'
        ) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                _build_topo(child)
            topo.append(v)
        _build_topo(self)
        for node in reversed(topo):
            node._backward()


    # ------------------------------------------------------------------ #
    #  Arithmetic operators                                                #
    # ------------------------------------------------------------------ #

    def __add__(self, other: TensorLike) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        result = self.data + other.data
        out = Tensor(
            result,
            requires_grad=(self.requires_grad or other.requires_grad),
            is_leaf=False,
            dtype=result.dtype,
        )
        if out.requires_grad:  # FIX: was `if self.requires_grad` — missed case where only other needs grad
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    g = Tensor._sum_to_shape(out.grad, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other.requires_grad:
                    g = Tensor._sum_to_shape(out.grad, other.shape)
                    other.grad = g if other.grad is None else other.grad + g
            out._backward = _backward
            out._prev = {self, other}
            out._op = "+"
        return out

    def __radd__(self, other: TensorLike) -> 'Tensor':
        return self + other

    def __mul__(self, other: TensorLike) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        result = self.data * other.data
        out = Tensor(
            result,
            requires_grad=(self.requires_grad or other.requires_grad),
            is_leaf=False,
            dtype=result.dtype,
        )
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    g = Tensor._sum_to_shape(out.grad * other.data, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other.requires_grad:
                    g = Tensor._sum_to_shape(out.grad * self.data, other.shape)
                    other.grad = g if other.grad is None else other.grad + g
            out._backward = _backward
            out._prev = {self, other}
            out._op = "*"
        return out

    def __rmul__(self, other: TensorLike) -> 'Tensor':
        return self * other

    def __neg__(self) -> 'Tensor':
        return self * (-1)

    def __sub__(self, other: TensorLike) -> 'Tensor':
        return self + (-other)  # FIX: was redundant isinstance branch — both sides did -other

    def __rsub__(self, other: TensorLike) -> 'Tensor':
        return (-self) + other

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        if not isinstance(power, (int, float)):
            raise TypeError(
                f"power must be int or float, got {type(power).__name__}"
            )
        result = self.data ** power
        out = Tensor(
            result,
            requires_grad=self.requires_grad,
            is_leaf=False,
            dtype=result.dtype,
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * power * (self.data ** (power - 1))
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = f"**{power}"
        return out

    def __truediv__(self, other: TensorLike) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return self * (other ** -1)

    def __rtruediv__(self, other: TensorLike) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other * (self ** -1)

    # ------------------------------------------------------------------ #
    #  Matrix multiply                                                     #
    # ------------------------------------------------------------------ #

    def __matmul__(self, other: TensorLike) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        try:
            result = self.data @ other.data
        except ValueError as e:
            raise ValueError(
                f"Cannot matmul tensors with shapes {self.shape} and {other.shape}"
            ) from e
        out = Tensor(
            result,
            requires_grad=(self.requires_grad or other.requires_grad),
            is_leaf=False,
            dtype=result.dtype,
        )
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    g = Tensor._sum_to_shape(
                        out.grad @ np.swapaxes(other.data, -1, -2), self.shape
                    )
                    self.grad = g if self.grad is None else self.grad + g
                if other.requires_grad:
                    g = Tensor._sum_to_shape(
                        np.swapaxes(self.data, -1, -2) @ out.grad, other.shape
                    )
                    other.grad = g if other.grad is None else other.grad + g
            out._backward = _backward
            out._prev = {self, other}
            out._op = "matmul"
        return out

    def __rmatmul__(self, other: TensorLike) -> 'Tensor':  # FIX: was missing entirely
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other @ self

    # ------------------------------------------------------------------ #
    #  Shape operations                                                    #
    # ------------------------------------------------------------------ #

    def reshape(self, *shape: int) -> 'Tensor':
        if len(shape) == 1:
            if isinstance(shape[0], (tuple, list)):
                new_shape = tuple(shape[0])
            elif isinstance(shape[0], int):
                new_shape = (shape[0],)
            else:
                raise TypeError(
                    f"reshape() expected ints, tuple, or list, "
                    f"got {type(shape[0]).__name__}"
                )
        else:
            new_shape = tuple(shape)

        for dim in new_shape:
            if not isinstance(dim, int):
                raise TypeError(
                    f"shape dimensions must be integers, got {type(dim).__name__}"
                )

        input_shape = self.shape
        try:
            result = self.data.reshape(new_shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape tensor of shape {self.shape} to shape {new_shape}"
            ) from e

        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad.reshape(input_shape)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "reshape"
        return out

    def transpose(self, axes=None) -> 'Tensor':
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        if not isinstance(axes, tuple):
            raise TypeError(f"axes must be a tuple, got {type(axes).__name__}")
        if len(axes) != self.ndim:
            raise ValueError(
                f"axes length ({len(axes)}) must match tensor ndim ({self.ndim})"
            )
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(f"invalid axis permutation {axes}")

        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            is_leaf=False,
            dtype=self.dtype,
        )
        if self.requires_grad:
            reverse_axes = tuple(np.argsort(axes))
            def _backward():
                if out.grad is None:
                    return
                grad = np.transpose(out.grad, reverse_axes)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "transpose"
        return out

    def flatten(self) -> 'Tensor':
        original_shape = self.shape
        out = Tensor(
            self.data.flatten(),
            requires_grad=self.requires_grad,
            is_leaf=False,
            dtype=self.dtype,
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad.reshape(original_shape)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "flatten"
        return out

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> 'Tensor':
        original_shape = self.shape
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            elif not isinstance(axis, tuple):
                raise TypeError(
                    f"axis must be int, tuple of ints, or None, "
                    f"got {type(axis).__name__}"
                )
            normalized_axes = []
            for ax in axis:
                if not isinstance(ax, int):
                    raise TypeError(
                        f"axis values must be integers, got {type(ax).__name__}"
                    )
                if ax < -self.ndim or ax >= self.ndim:
                    raise ValueError(
                        f"axis {ax} is out of bounds for tensor of dimension {self.ndim}"
                    )
                if ax < 0:
                    ax += self.ndim
                if self.shape[ax] != 1:
                    raise ValueError(
                        f"cannot squeeze axis {ax} with size {self.shape[ax]}"
                    )
                normalized_axes.append(ax)
            axis = tuple(sorted(set(normalized_axes)))

        try:
            result = np.squeeze(self.data, axis=axis)
        except ValueError as e:
            raise ValueError(str(e)) from e

        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad.reshape(original_shape)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "squeeze"
        return out

    def expand_dims(self, axis: int) -> 'Tensor':
        if not isinstance(axis, int):
            raise TypeError(f"axis must be int, got {type(axis).__name__}")
        if axis < -(self.ndim + 1) or axis > self.ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for tensor of dimension {self.ndim}"
            )
        original_shape = self.shape
        if axis < 0:
            axis += self.ndim + 1
        result = np.expand_dims(self.data, axis)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = np.squeeze(out.grad, axis=axis)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "expand_dims"
        return out

    def __getitem__(self, key) -> 'Tensor':
        result = self.data[key]
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = np.zeros_like(self.data)
                grad[key] = out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "getitem"
        return out

    # ------------------------------------------------------------------ #
    #  Reduction operations                                                #
    # ------------------------------------------------------------------ #

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> 'Tensor':
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            original_shape = self.shape
            ndim = self.ndim  # capture at creation, not at backward time
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(a if a >= 0 else a + ndim for a in axes):
                        grad = np.expand_dims(grad, ax)
                # FIX: .copy() — broadcast_to returns read-only view; grad += later would crash
                grad = np.broadcast_to(grad, original_shape).copy()
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "sum"
        return out

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> 'Tensor':
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            if axis is None:
                count = self.data.size
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                count = 1
                for ax in axes:
                    count *= self.shape[ax]
            original_shape = self.shape
            ndim = self.ndim
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad
                if axis is not None and not keepdims:
                    axes_local = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(a if a >= 0 else a + ndim for a in axes_local):
                        grad = np.expand_dims(grad, ax)
                # FIX: .copy() same as sum — read-only view must be copied before accumulation
                grad = np.broadcast_to(grad, original_shape).copy() / count
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "mean"
        return out

    def max(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> 'Tensor':
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if axis is None:
                    mask = self.data == result
                    grad = (mask * 1.0) * out.grad
                else:
                    max_vals = result if keepdims else np.expand_dims(result, axis)
                    mask = self.data == max_vals
                    g = out.grad if keepdims else np.expand_dims(out.grad, axis)
                    grad = (mask * 1.0) * g
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "max"
        return out

    # ------------------------------------------------------------------ #
    #  Element-wise math                                                   #
    # ------------------------------------------------------------------ #

    def exp(self) -> 'Tensor':
        result = np.exp(self.data)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * result
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "exp"
        return out

    def log(self) -> 'Tensor':
        result = np.log(self.data)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad / self.data
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "log"
        return out

    def sqrt(self) -> 'Tensor':
        result = np.sqrt(self.data)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad / (2 * result)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "sqrt"
        return out

    def abs(self) -> 'Tensor':
        result = np.abs(self.data)
        out = Tensor(
            result, requires_grad=self.requires_grad, is_leaf=False, dtype=result.dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * np.sign(self.data)
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = "abs"
        return out

    def astype(self, dtype: Union[np.dtype, type, str]) -> 'Tensor':
        try:
            target_dtype = np.dtype(dtype)
        except (TypeError, ValueError):
            raise TypeError(f"Invalid dtype: {dtype}")

        original_dtype = self.dtype
        out = Tensor(
            self.data, requires_grad=self.requires_grad, is_leaf=False, dtype=target_dtype
        )
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                grad = Tensor(out.grad, dtype=original_dtype).data
                self.grad = grad if self.grad is None else self.grad + grad
            out._backward = _backward
            out._prev = {self}
            out._op = f"astype({target_dtype})"
        return out

    # ------------------------------------------------------------------ #
    #  Static constructors                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def zeros(
        *shape,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> 'Tensor':
        if len(shape) == 1:
            if isinstance(shape[0], Tensor):
                shape = tuple(int(x) for x in shape[0].data.flatten())
            elif isinstance(shape[0], (tuple, list, np.ndarray)):
                shape = tuple(int(x) for x in shape[0])
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def ones(
        *shape,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> 'Tensor':
        if len(shape) == 1:
            if isinstance(shape[0], Tensor):
                shape = tuple(int(x) for x in shape[0].data.flatten())
            elif isinstance(shape[0], (tuple, list, np.ndarray)):
                shape = tuple(int(x) for x in shape[0])
        return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def randn(
        *shape,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> 'Tensor':
        if len(shape) == 1:
            if isinstance(shape[0], Tensor):
                shape = tuple(int(x) for x in shape[0].data.flatten())
            elif isinstance(shape[0], (tuple, list, np.ndarray)):
                shape = tuple(int(x) for x in shape[0])
        return Tensor(
            np.random.randn(*shape),
            requires_grad=requires_grad,
            dtype=dtype,
        )

    @staticmethod
    def rand(
        *shape,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> 'Tensor':
        if len(shape) == 1:
            if isinstance(shape[0], Tensor):
                shape = tuple(int(x) for x in shape[0].data.flatten())
            elif isinstance(shape[0], (tuple, list, np.ndarray)):
                shape = tuple(int(x) for x in shape[0])
        return Tensor(
            np.random.rand(*shape),
            requires_grad=requires_grad,
            dtype=dtype,
        )

    @staticmethod
    def arange(
        start,
        stop=None,
        step=1,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> 'Tensor':
        if isinstance(start, Tensor):
            start = start.data.item()
        if isinstance(stop, Tensor):
            stop = stop.data.item()
        if isinstance(step, Tensor):
            step = step.data.item()
        if stop is None:
            start, stop = 0, start
        return Tensor(
            np.arange(start, stop, step, dtype=dtype),
            requires_grad=requires_grad,
            dtype=dtype,
        )











# def test_tensor():
#     print("\n" + "=" * 60)
#     print("TESTING TENSOR CLASS")
#     print("=" * 60)

#     print("\n===== 1. Creation =====")
#     print("zeros:", Tensor.zeros(2, 3).shape)
#     print("ones:", Tensor.ones((2, 3)).shape)
#     print("randn:", Tensor.randn(2, 2).shape)
#     print("rand:", Tensor.rand(2, 2).shape)
#     print("arange:", Tensor.arange(5).shape)
#     print("✓ Creation works")

#     print("\n===== 2. Arithmetic Operations =====")
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([4, 5, 6], requires_grad=True)
#     assert (a + b).shape == (3,)
#     assert (a - b).shape == (3,)
#     assert (a * b).shape == (3,)
#     assert (a / b).shape == (3,)
#     assert (a ** 2).shape == (3,)
#     assert (-a).shape == (3,)
#     print("✓ All arithmetic operations work")

#     print("\n===== 3. Shape Operations =====")
#     x = Tensor([[1, 2, 3], [4, 5, 6]])
#     assert x.reshape(3, 2).shape == (3, 2)
#     assert x.T.shape == (3, 2)
#     assert x.flatten().shape == (6,)
#     assert x.expand_dims(0).shape == (1, 2, 3)
#     assert Tensor([[[1, 2, 3]]]).squeeze().shape == (3,)
#     print("✓ All shape operations work")

#     print("\n===== 4. Indexing =====")
#     x = Tensor([[1, 2, 3], [4, 5, 6]])
#     assert x[0].shape == (3,)
#     assert x[:, 1].shape == (2,)
#     print("✓ Indexing works")

#     print("\n===== 5. Matrix Operations =====")
#     m1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
#     m2 = Tensor([[5, 6], [7, 8]], requires_grad=True)
#     result = m1 @ m2
#     assert result.shape == (2, 2)
#     print("✓ Matrix multiply works")

#     print("\n===== 6. Reduction Operations =====")
#     r = Tensor([[1., 2., 3.], [4., 5., 6.]])
#     assert r.sum().shape == ()
#     assert r.sum(axis=0).shape == (3,)
#     assert r.mean().shape == ()
#     assert r.mean(axis=1).shape == (2,)
#     assert r.max().shape == ()
#     assert r.max(axis=1).shape == (2,)
#     print("✓ All reduction operations work")

#     print("\n===== 7. Element-wise Math =====")
#     t = Tensor([1., 2., 4.])
#     assert t.exp().shape == (3,)
#     assert t.log().shape == (3,)
#     assert t.sqrt().shape == (3,)
#     assert Tensor([-1., -2., 3.]).abs().shape == (3,)
#     print("✓ All element-wise operations work")

#     print("\n===== 8. astype() - DTYPE CONVERSION =====")
#     x_f32 = Tensor([1.5, 2.7, 3.2], requires_grad=True, dtype=np.float32)
#     print(f"Original dtype: {x_f32.dtype}")

#     x_f64 = x_f32.astype(np.float64)
#     print(f"Converted to float64: {x_f64.dtype}")
#     assert x_f64.dtype == np.float64
#     print("✓ astype to float64 works")

#     x_f16 = x_f32.astype(np.float16)
#     print(f"Converted to float16: {x_f16.dtype}")
#     assert x_f16.dtype == np.float16
#     print("✓ astype to float16 works")

#     print("\n===== 9. astype() with Gradients =====")
#     x = Tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=np.float32)
#     y = x.astype(np.float64)
#     z = y * 2.0
#     z.backward(np.ones_like(z.data))
#     print(f"Input dtype: {x.dtype}")
#     print(f"Converted dtype: {y.dtype}")
#     print(f"Gradient dtype: {x.grad.dtype}")
#     print(f"Gradient values: {x.grad}")
#     assert x.grad.dtype == np.float32
#     assert np.allclose(x.grad, [2.0, 2.0, 2.0])
#     print("✓ Gradient flow through astype works")

#     print("\n===== 10. Autograd: Basic Operations =====")
#     x = Tensor([1., 2., 3.], requires_grad=True)
#     y = Tensor([4., 5., 6.], requires_grad=True)
#     z = x + y
#     z.backward(np.ones_like(z.data))
#     assert np.allclose(x.grad, [1., 1., 1.])
#     assert np.allclose(y.grad, [1., 1., 1.])
#     print("✓ Addition backward works")

#     x.zero_grad()
#     y.zero_grad()
#     z = x * y
#     z.backward(np.ones_like(z.data))
#     assert np.allclose(x.grad, y.data)
#     assert np.allclose(y.grad, x.data)
#     print("✓ Multiplication backward works")

#     print("\n===== 11. Autograd: Complex Graph =====")
#     x = Tensor([1., 2., 3.], requires_grad=True)
#     loss = (x * x).sum()
#     loss.backward()
#     print(f"Loss: {loss.data}")
#     print(f"x.grad: {x.grad}")
#     assert np.allclose(x.grad, [2., 4., 6.])
#     print("✓ Complex autograd works")

#     print("\n===== 12. Autograd: Matrix Multiply =====")
#     a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
#     b = Tensor([[5., 6.], [7., 8.]], requires_grad=True)
#     out = (a @ b).sum()
#     out.backward()
#     assert a.grad is not None
#     assert b.grad is not None
#     print("✓ Matrix multiply backward works")

#     print("\n===== 13. max() with Boolean Mask (No np.astype) =====")
#     x = Tensor([-2.0, 1.0, 3.0, -1.0], requires_grad=True)
#     y = x.max()
#     y.backward()
#     print(f"Input: {x.data}")
#     print(f"Max: {y.data}")
#     print(f"Gradients: {x.grad}")
#     assert x.grad[2] == 1.0
#     assert x.grad[0] == 0.0
#     print("✓ max() with boolean mask works")

#     print("\n===== 14. Mixed Operations =====")
#     x = Tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=np.float32)
#     x_f64 = x.astype(np.float64)
#     y = x_f64 * 2.0
#     z = y.sum()
#     z.backward()
#     print(f"x dtype: {x.dtype}, x.grad dtype: {x.grad.dtype}")
#     print(f"Gradient: {x.grad}")
#     assert np.allclose(x.grad, [2.0, 2.0, 2.0])
#     print("✓ Mixed dtype operations work")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED ✓✓✓")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_tensor()
