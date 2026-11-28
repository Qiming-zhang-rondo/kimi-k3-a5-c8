"""Real address-range/alias tests of the startup arena, without Torch or NPU."""
import ctypes
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framework_integration.sfa_recent128.pd_arena import Arena


_SIZES = {"torch.uint8": 1, "torch.bfloat16": 2, "torch.int64": 8}


class Tensor:
    def __init__(self, shape, dtype, device, *, storage=None, offset=0):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.offset = offset
        self.storage = storage if storage is not None else ctypes.create_string_buffer(self.nbytes)

    @property
    def nbytes(self):return self.numel() * self.element_size()
    def numel(self):
        count = 1
        for size in self.shape:count *= size
        return count
    def element_size(self):return _SIZES[self.dtype]
    def data_ptr(self):return ctypes.addressof(self.storage) + self.offset
    def is_contiguous(self):return True
    def narrow(self, dim, start, length):
        assert dim == 0 and len(self.shape) == 1
        assert 0 <= start <= start + length <= self.shape[0]
        return Tensor((length,), self.dtype, self.device, storage=self.storage,
                      offset=self.offset + start * self.element_size())
    def view(self, target):
        if isinstance(target, str):
            assert self.nbytes % _SIZES[target] == 0
            shape, dtype = (self.nbytes // _SIZES[target],), target
        else:
            shape, dtype = tuple(target), self.dtype
            size = 1
            for value in shape:size *= value
            assert size == self.numel()
        return Tensor(shape, dtype, self.device, storage=self.storage, offset=self.offset)


class Torch:
    uint8, bfloat16, int64 = "torch.uint8", "torch.bfloat16", "torch.int64"
    device = staticmethod(lambda device: device)
    empty = staticmethod(lambda shape, *, dtype, device: Tensor(shape, dtype, device))


class Engine:
    def __init__(self):self.calls=[]
    def register_memory(self, pointer, nbytes):self.calls.append((pointer,nbytes));return 0


class ArenaTests(unittest.TestCase):
    def make(self, slots=2):
        engine=Engine()
        return Arena(engine,Torch,"npu:3",("L0","L1"),slots),engine

    def test_full_extent_and_all_views_are_registered_once_and_aligned(self):
        arena,engine=self.make()
        extent=arena.registered_extent
        self.assertEqual(extent["pointer"] % (2 * 1024 * 1024),0)
        self.assertEqual(extent["nbytes"] % (2 * 1024 * 1024),0)
        self.assertEqual(engine.calls,[(extent["pointer"],extent["nbytes"])])
        layers=arena.acquire("r",13)
        for tail in layers.values():
            for field,shape,dtype in (("latent",(13,512),Torch.bfloat16),
                    ("rope",(13,64),Torch.bfloat16),("position_tags",(13,),Torch.int64),
                    ("epoch_tags",(13,),Torch.int64)):
                value=getattr(tail,field)
                self.assertEqual((value.shape,value.dtype,value.device),(shape,dtype,"npu:3"))
                self.assertTrue(arena.contains(value.data_ptr(),value.nbytes))
        self.assertEqual(len(engine.calls),1)

    def test_slots_do_not_alias_and_reuse_only_after_release(self):
        arena,engine=self.make()
        first=arena.acquire("a",128)
        second=arena.acquire("b",128)
        a=first["L0"].latent;b=second["L0"].latent
        ctypes.memset(a.data_ptr(),23,a.nbytes)
        ctypes.memset(b.data_ptr(),47,b.nbytes)
        self.assertEqual(ctypes.string_at(a.data_ptr(),a.nbytes),bytes([23])*a.nbytes)
        with self.assertRaisesRegex(RuntimeError,"exhausted"):arena.acquire("c",128)
        self.assertTrue(arena.release("a"))
        self.assertFalse(arena.release("a"))
        self.assertEqual(arena.acquire("c",128)["L0"].latent.data_ptr(),a.data_ptr())
        self.assertEqual(ctypes.string_at(b.data_ptr(),b.nbytes),bytes([47])*b.nbytes)
        self.assertEqual(len(engine.calls),1)

    def test_retry_is_same_allocation_and_cannot_change_length(self):
        arena,_=self.make()
        first=arena.acquire(("request","nonce"),13)
        self.assertIs(arena.acquire(("request","nonce"),13),first)
        with self.assertRaisesRegex(RuntimeError,"changed"):arena.acquire(("request","nonce"),128)

    def test_invalid_counts_and_geometry_fail_before_registration(self):
        arena,_=self.make()
        for count in (0,129,-1,True):
            with self.assertRaisesRegex(RuntimeError,"length"):arena.acquire("r",count)
        engine=Engine()
        for slots,alignment,keys in ((0,8,("L",)),(1,7,("L",)),(1,8,()),(1,8,("L","L"))):
            with self.assertRaises(RuntimeError):Arena(engine,Torch,"npu:0",keys,slots,alignment)
        self.assertEqual(engine.calls,[])

    def test_startup_registration_failure_is_explicit(self):
        class Failure(Engine):
            def register_memory(self,pointer,nbytes):return -1
        with self.assertRaisesRegex(RuntimeError,"register_memory failed"):
            Arena(Failure(),Torch,"npu:0",("L",),1)


if __name__=="__main__":unittest.main()
