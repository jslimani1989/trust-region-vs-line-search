# -*- coding: utf-8 -*-
"""
Consigne 008c, Partie A (A1) : epinglage du processus sur un coeur logique
(masque d'affinite Windows) et priorite haute, via ctypes (aucune dependance
externe telle que psutil, non installee dans l'environnement).

IMPORTANT : les variables d'environnement OMP_NUM_THREADS / MKL_NUM_THREADS /
OPENBLAS_NUM_THREADS doivent etre positionnees AVANT l'import de numpy dans le
script appelant (comme dans run_BFGSW_v0_3.py, consigne 008b) : ce module ne
les repositionne pas lui-meme pour eviter un import numpy premature.
"""
import ctypes
import ctypes.wintypes as wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

HIGH_PRIORITY_CLASS = 0x00000080
NORMAL_PRIORITY_CLASS = 0x00000020

GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.restype = wintypes.HANDLE

SetProcessAffinityMask = kernel32.SetProcessAffinityMask
SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
SetProcessAffinityMask.restype = wintypes.BOOL

GetProcessAffinityMask = kernel32.GetProcessAffinityMask
GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
GetProcessAffinityMask.restype = wintypes.BOOL

SetPriorityClass = kernel32.SetPriorityClass
SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
SetPriorityClass.restype = wintypes.BOOL

GetPriorityClass = kernel32.GetPriorityClass
GetPriorityClass.argtypes = [wintypes.HANDLE]
GetPriorityClass.restype = wintypes.DWORD


def set_affinity(cpu_index: int) -> bool:
    """Restreint TOUT le processus courant (tous ses threads) au processeur
    logique `cpu_index` (0-based). Retourne True si l'appel Windows a reussi."""
    handle = GetCurrentProcess()
    mask = 1 << cpu_index
    ok = SetProcessAffinityMask(handle, mask)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(ok)


def get_affinity():
    handle = GetCurrentProcess()
    proc_mask = ctypes.c_size_t(0)
    sys_mask = ctypes.c_size_t(0)
    ok = GetProcessAffinityMask(handle, ctypes.byref(proc_mask), ctypes.byref(sys_mask))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return proc_mask.value, sys_mask.value


def n_logical_processors() -> int:
    _, sys_mask = get_affinity()
    return bin(sys_mask).count("1")


def set_high_priority() -> bool:
    handle = GetCurrentProcess()
    ok = SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(ok)


def get_priority_name() -> str:
    handle = GetCurrentProcess()
    val = GetPriorityClass(handle)
    names = {0x00000040: "IDLE_PRIORITY_CLASS", 0x00004000: "BELOW_NORMAL_PRIORITY_CLASS",
             0x00000020: "NORMAL_PRIORITY_CLASS", 0x00008000: "ABOVE_NORMAL_PRIORITY_CLASS",
             0x00000080: "HIGH_PRIORITY_CLASS", 0x00000100: "REALTIME_PRIORITY_CLASS"}
    return names.get(val, f"UNKNOWN(0x{val:x})")
