import llama_cpp
import ctypes

@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def test_callback(level, message, user_data):
    print("CALLBACK FIRED:", message)

llama_cpp.llama_log_set(test_callback, ctypes.c_void_p(0))

print("Calling llama_backend_init to see logs...")
try:
    llama_cpp.llama_backend_init()
    print("Done init")
except Exception as e:
    print("Error:", e)
