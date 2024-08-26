import os
import hashlib
import importlib
import importlib.resources
import tempfile

import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
_triton_C_dir = importlib.resources.files(triton).joinpath("_C")

#include_dirs = [os.path.join(_dirname, "include"), "/localdisk/ilyaenko/Intel_VTune_Profiler_2024.2.0_nda/include"]
#library_dirs = [os.path.join(_dirname, "lib"), _triton_C_dir, "/localdisk/ilyaenko/Intel_VTune_Profiler_2024.2.0_nda/lib64"]
#libraries = ["stdc++", "ittnotify"]
include_dirs = [os.path.join(_dirname, "include")]
library_dirs = [os.path.join(_dirname, "lib"), _triton_C_dir]
libraries = ["stdc++", "tbb"]


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            with open("main.cpp", "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        if name == "softmax_kernel" and False:
            import ctypes
            lib = ctypes.cdll.LoadLibrary("/localdisk/ilyaenko/triton-cpu/triton-cpu-tests/02-softmax/softmax_kernel.so")
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "l",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format
    arg_ptrs_list = ', '.join(f"&arg{i}" for i, ty in signature.items())
    kernel_fn_args = [i for i in signature.keys() if i not in constants]
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>
#include <tbb/tbb.h>

#define ENABLE_ITT 0

#if ENABLE_ITT
#include <ittnotify.h>
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

#if ENABLE_ITT
auto dom1 = __itt_domain_create("launcher");
//auto dom2 = __itt_domain_create("omp_launcher");
//auto dom3 = __itt_domain_create("thread");
__itt_string_handle* launcher_call_str = __itt_string_handle_create("launcher call");
__itt_string_handle* omp_launcher_call_str = __itt_string_handle_create("omp launcher call");
__itt_string_handle* kernel_call_str = __itt_string_handle_create("kernel call");
#endif

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{ 
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

extern "C" void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
#if ENABLE_ITT
    __itt_task_begin(dom1, __itt_null, __itt_null, omp_launcher_call_str);
#endif
  // TODO: Consider using omp collapse(3) clause for simplicity?
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

#pragma omp parallel for schedule(static) num_threads(64)
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
  }}

#if ENABLE_ITT
    __itt_task_end(dom1);
#endif
}}

extern "C" void run_tbb_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
#if ENABLE_ITT
    __itt_task_begin(dom1, __itt_null, __itt_null, omp_launcher_call_str);
#endif
  // TODO: Consider using omp collapse(3) clause for simplicity?
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, N),
    [&](const tbb::blocked_range<size_t>& r) {{
    for (size_t i = r.begin(); i != r.end(); ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
  }});

#if ENABLE_ITT
    __itt_task_end(dom1);
#endif
}}

extern "C" void run_omp_kernels_collapse(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
#if ENABLE_ITT
    __itt_task_begin(dom1, __itt_null, __itt_null, omp_launcher_call_str);
#endif
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;

  // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel for collapse(3) schedule(static)
  for (size_t k = 0; k < gridZ; ++k) {{
  for (size_t j = 0; j < gridY; ++j) {{
  for (size_t i = 0; i < gridX; ++i) {{
#if ENABLE_ITT
//    __itt_task_begin(dom1, __itt_null, __itt_null, kernel_call_str);
#endif
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} i, j, k, gridX, gridY, gridZ);
#if ENABLE_ITT
//    __itt_task_end(dom1);
#endif
  }}
  }}
  }}

#if ENABLE_ITT
    __itt_task_end(dom1);
#endif
}}

extern "C" void run_omp_kernels2(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
#if ENABLE_ITT
    __itt_task_begin(dom1, __itt_null, __itt_null, omp_launcher_call_str);
#endif
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;

  // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel
  {{
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t iters_per_thread = (N + num_threads - 1) / num_threads;
    int64_t start = iters_per_thread * tid;
    int64_t end = std::min((int64_t)N, start + iters_per_thread);
    int64_t X = start % gridX;
    int64_t Y = start / gridX % gridY;
    int64_t Z = start / gridX / gridY;
    for (int64_t i = start; i < end; ++i) {{
      /*
      ++X;
      if (X == gridX) {{
        X = 0;
        ++Y;
        if (Y == gridY) {{
          Y = 0;
          ++Z;
        }}
      }}
      */
#if ENABLE_ITT
      __itt_task_begin(dom1, __itt_null, __itt_null, kernel_call_str);
#endif
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} i, Y, Z, gridX, gridY, gridZ);
#if ENABLE_ITT
      __itt_task_end(dom1);
#endif
    }}
  }}

#if ENABLE_ITT
    __itt_task_end(dom1);
#endif
}}

#include <chrono>
#include <thread>

static PyObject* launch(PyObject* self, PyObject* args) {{
#if ENABLE_ITT
  __itt_task_begin(dom1, __itt_null, __itt_null, launcher_call_str);
#endif
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  auto start = std::chrono::high_resolution_clock::now();

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  /*
  DevicePtrInfo ptr_info = getPointer(arg0, 0);
  std::chrono::duration<double, std::micro> total_time;
  do {{
    auto end = std::chrono::high_resolution_clock::now();
    total_time = end - start;
    *((float *)ptr_info.dev_ptr) = total_time.count();
  }} while (total_time.count() < 200);
  return Py_None;
  */

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  run_omp_kernels(gridX, gridY, gridZ, kernel_ptr {',' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> total_time = end - start;
  *((float *)ptr_info0.dev_ptr) = total_time.count();

#if ENABLE_ITT
  __itt_task_end(dom1);
#endif
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        super().__init__()

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        return GPUTarget("cpu", self.cpu_arch, 0)

    @staticmethod
    def is_active():
        return True
