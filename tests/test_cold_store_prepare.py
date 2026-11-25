"""Actual kernel-body CPU model and setup/runtime metadata controls; no NPU claim."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/"direct_bridge/cold_store_prepare.py"
spec = importlib.util.spec_from_file_location("cold_store_prepare_under_test", SOURCE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
ORDER = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")

class Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype):
        return Tensor(np.asarray(self).astype(dtype))


class Pointer:
    def __init__(self, array, name, offset=0):
        self.array, self.name, self.offset = array.reshape(-1), name, np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.array, self.name, self.offset + offset)


class TL:
    constexpr, int32, int64 = object, np.int32, np.int64

    def __init__(self):
        self.row = 0
        self.writes = {}
        self.scans = 0

    def program_id(self, axis):
        assert axis == 0
        return Tensor(np.int32(self.row))

    @staticmethod
    def arange(start, end):
        return Tensor(np.arange(start, end, dtype=np.int32))

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, offsets.shape)
        assert np.all(offsets[active] >= 0) and np.all(offsets[active] < pointer.array.size)
        result = np.full(offsets.shape, other, pointer.array.dtype)
        result[active] = pointer.array[offsets[active]]
        return Tensor(result)

    def store(self, pointer, values, mask=True):
        offsets = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, offsets.shape)
        assert np.all(offsets[active] >= 0) and np.all(offsets[active] < pointer.array.size)
        indices = offsets[active]
        # Track every physical output slot, including repeated scatter offsets.
        # A correct partition writes each slot exactly once, independent of
        # instruction scheduling; this catches fill/scatter write races.
        counts = self.writes.setdefault(pointer.name, np.zeros(pointer.array.size, np.int32))
        np.add.at(counts, indices, 1)
        pointer.array[indices] = np.broadcast_to(values, offsets.shape)[active]

    def cumsum(self, value, axis):
        self.scans += 1
        return Tensor(np.cumsum(value, axis=axis, dtype=np.int32))

    @staticmethod
    def sum(value, axis):
        return Tensor(np.sum(value, axis=axis, dtype=np.int32))

    @staticmethod
    def maximum(left, right):
        return Tensor(np.maximum(left, right))

    @staticmethod
    def max(value, axis):
        return Tensor(np.max(value, axis=axis))

SIZES = {"bf16": 2, "fp32": 4, "uint8": 1, "int8": 1, "int32": 4,
         "int64": 8, "fp8_e4m3fn": 1, "fp4_e2m1fn_x2": 1}


class MetaTensor:
    """Metadata-only tensor; only physical slot maps allow host value access."""

    def __init__(self, shape, dtype, *, device="npu:0", contiguous=True, values=None,
                 events=None, storage=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.contiguous, self.values = contiguous, values
        self.events = [] if events is None else events
        self.storage = object() if storage is None else storage

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return SIZES[self.dtype]

    def is_contiguous(self):
        return self.contiguous

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            dtype = args[0]
            nbytes = self.numel() * self.element_size()
            if nbytes % SIZES[dtype]:
                raise RuntimeError("Invalid byte view")
            shape = (nbytes // SIZES[dtype],)
        else:
            dtype, shape = self.dtype, list(args)
            if shape.count(-1) > 1:
                raise RuntimeError("Invalid reshape")
            if -1 in shape:
                known = math.prod(x for x in shape if x != -1)
                if known == 0 or self.numel() % known:
                    raise RuntimeError("Invalid reshape size")
                shape[shape.index(-1)] = self.numel() // known
            if math.prod(shape) != self.numel():
                raise RuntimeError("Invalid reshape size")
        self.events.append("view")
        return MetaTensor(shape, dtype, device=self.device, contiguous=self.contiguous,
                      events=self.events, storage=self.storage)

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        return self

    def numpy(self):
        self.events.append("numpy")
        if self.values is None:
            raise AssertionError("Only setup slot maps may be copied to host")
        return self.values


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


def kernel_body():
    tree = ast.parse(SOURCE.read_text())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == "_cold_bf16_hot_prepare_kernel")
    fn.decorator_list = []
    language = TL()
    scope = {"tl": language}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SOURCE), "exec"), scope)
    return language, scope[fn.name]


class ActualKernelTests(unittest.TestCase):
    def exercise(self, *, invalid=False, keys=8192):
        rows, sparse, hot_width, ring = 4, 2048, 128, 256
        rng = np.random.default_rng(31)
        positions = np.arange(keys-rows, keys, dtype=np.int64)
        indices = np.stack([rng.permutation(np.r_[np.arange(1920), np.arange(p-127,p+1)])
                            for p in positions]).astype(np.int32)[:,None,:]
        cold_slots = np.array([keys-1,0,512,keys-2], np.int64)
        hot_slots = np.array([0,255,128,7], np.int32)
        if invalid:
            cold_slots[0], cold_slots[1] = -1, keys
            hot_slots[2], hot_slots[3] = -1, ring
            indices[3,0,0] = keys+7
        outputs = {name: np.full((rows,1,sparse if name=="old" else hot_width), 777, np.int32)
                   if i<3 else np.zeros(rows,np.bool_) for i,name in enumerate(ORDER)}
        cold = np.full((keys,400),165,np.uint8)
        latent = np.full((ring,1024),165,np.uint8)
        rope = np.full((ring,128),165,np.uint8)
        # Reuse all output buffers after poison, but pass newly generated bytes.
        for iteration in range(2):
            packed = rng.integers(0,256,(rows,400),np.uint8)
            current = rng.integers(0,256,(rows,1024),np.uint8)
            current_rope = rng.integers(0,256,(rows,128),np.uint8)
            for value in (cold,latent,rope): value.fill(165)
            arrays = [indices,positions,*(outputs[n] for n in ORDER),current,current_rope,hot_slots,
                      latent,rope,packed,cold_slots,cold]
            names = ["indices","positions",*ORDER,"current","current_rope","hot_slots",
                     "latent","rope","packed","cold_slots","cold"]
            language,kernel=kernel_body()
            pointers=[Pointer(a,n) for a,n in zip(arrays,names)]
            for row in range(rows):
                language.row=row
                kernel(*pointers,S=sparse,HW=hot_width,WINDOW=128,CAP=ring,BLOCK=2048,
                       KV_BYTES=1024,ROPE_BYTES=128,HOT_CAPACITY=ring,STORE_BLOCK=1024,
                       COLD_WIDTH=400,COLD_CAPACITY=keys)
            for name,actual,slots,source in (("cold",cold,cold_slots,packed),
                    ("latent",latent,hot_slots,current),("rope",rope,hot_slots,current_rope)):
                expected=np.full_like(actual,165);writes=np.zeros_like(actual,np.int32)
                for row,slot in enumerate(slots):
                    if 0<=slot<len(actual):expected[slot]=source[row];writes[slot]=1
                np.testing.assert_array_equal(actual,expected,err_msg=name)
                np.testing.assert_array_equal(language.writes[name],writes.ravel(),err_msg=name)
            for row,p in enumerate(positions):
                ids=indices[row,0];visible=ids[(ids>=0)&(ids<=p)]
                old=visible[visible<=p-128];recent=visible[visible>p-128]
                for name,values,width in (("old",old,sparse),("recent_global",recent,hot_width),
                                          ("recent_ring",recent%ring,hot_width)):
                    values=values[:width].tolist() if len(values) else [0]
                    np.testing.assert_array_equal(outputs[name][row,0],values+[-1]*(width-len(values)))
                self.assertEqual(bool(outputs["old_present"][row]),bool(len(old)))
                self.assertEqual(bool(outputs["recent_present"][row]),bool(len(recent)))
                bad=(len(recent)!=128 or np.any((ids < -1)|(ids>p)) or
                     not 0<=cold_slots[row]<keys or not 0<=hot_slots[row]<ring)
                self.assertEqual(bool(outputs["invalid_rows"][row]),bad)
            for name in ORDER:
                np.testing.assert_array_equal(language.writes[name],np.ones(outputs[name].size,np.int32))
            self.assertEqual(language.scans,rows*2)

    def test_native_packed_hot_and_partition_bytes_exact_with_unwritten_slots_preserved(self):
        self.exercise()
        self.exercise(keys=32768)

    def test_runtime_bad_cold_or_hot_slots_are_masked_and_flagged_without_oob(self):
        self.exercise(invalid=True)


def fixture(rows=4, keys=8192):
    events=[]
    tensor=lambda shape,dtype,**kwargs:MetaTensor(shape,dtype,events=events,**kwargs)
    torch=SimpleNamespace(uint8="uint8",int8="int8",int32="int32",int64="int64")
    outputs={name:tensor((rows,1,2048 if name=="old" else 128),"int32")
             if i<3 else tensor((rows,),"uint8") for i,name in enumerate(ORDER)}
    old=SimpleNamespace(base=SimpleNamespace(torch=torch),count=rows,outputs=outputs,
        _inputs=(tensor((rows,1,2048),"int32"),tensor((rows,),"int64"),*(outputs[n] for n in ORDER)),
        constants=dict(S=2048,HW=128,WINDOW=128,CAP=1152,BLOCK=2048,KV_BYTES=1024,
                       ROPE_BYTES=128,HOT_CAPACITY=1152,STORE_BLOCK=1024),
        kv_bytes=tensor((rows,1024),"uint8"),rope_bytes=tensor((rows,128),"uint8"),
        hot_slots=tensor((rows,),"int64"),hot_bytes=tensor((1152,1024),"uint8"),
        hot_rope_bytes=tensor((1152,128),"uint8"),call=mock.Mock(side_effect=AssertionError("old launch forbidden")))
    return SimpleNamespace(old=old,cold=tensor((keys//128,128,1,400),"int8"),
        slots=tensor((rows,),"int64",values=np.arange(rows,dtype=np.int64)),
        packed=tensor((rows,400),"uint8"),events=events,tensor=tensor)


class WrapperTests(unittest.TestCase):
    def build(self,data):
        kernel=Kernel()
        with mock.patch.object(prepare,"_load_kernel",return_value=kernel):
            result=prepare.PreparedColdBf16HotStorePartition(data.old,data.cold,data.slots,data.packed)
        return result,kernel

    def test_one_launch_dynamic_current_packed_pointer_no_timed_host_read_or_view(self):
        for rows,keys in ((1,8192),(32,8192),(1024,32768)):
            data=fixture(rows,keys);result,kernel=self.build(data)
            self.assertIn("cpu",data.events)
            data.events.clear()
            for packed in (data.packed,data.tensor((rows,400),"uint8")):
                self.assertIs(result.call(packed),data.old.outputs)
                grid,args,constants=kernel.calls[-1]
                self.assertEqual(grid,(rows,))
                self.assertIs(args[13],packed)
                self.assertIs(args[14],data.slots)
                self.assertIs(args[15].storage,data.cold.storage)
                self.assertEqual(constants["COLD_WIDTH"],400)
                self.assertEqual(constants["COLD_CAPACITY"],keys)
            self.assertEqual(data.events,[])
            data.old.call.assert_not_called()
            self.assertEqual(len(kernel.calls),2)

    def test_setup_rejects_slot_collision_bounds_bad_storage_and_changed_runtime_metadata(self):
        for problem in ("duplicate","negative","past_end","dtype","width","device","grad","count","format"):
            data=fixture()
            if problem=="duplicate":data.slots.values[:2]=1
            elif problem=="negative":data.slots.values[0]=-1
            elif problem=="past_end":data.slots.values[0]=8192
            elif problem=="dtype":data.slots.dtype="fp32"
            elif problem=="width":data.cold.shape=(64,128,1,401)
            elif problem=="device":data.cold.device="npu:1"
            elif problem=="grad":data.packed.requires_grad=True
            elif problem=="count":data.old.count=1025
            else:data.old.constants["HW"]=64
            with self.subTest(problem=problem),self.assertRaises(ValueError):self.build(data)
        data=fixture();result,kernel=self.build(data);data.events.clear()
        for attribute,value in (("dtype","int8"),("shape",(4,401)),("device","npu:1"),
                                ("contiguous",False),("requires_grad",True)):
            packed=data.tensor((4,400),"uint8");setattr(packed,attribute,value)
            with self.subTest(attribute=attribute),self.assertRaises(ValueError):result.call(packed)
        self.assertEqual(kernel.calls,[])
        self.assertEqual(data.events,[])


if __name__=="__main__":unittest.main()
