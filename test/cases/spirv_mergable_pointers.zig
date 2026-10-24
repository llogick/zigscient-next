export fn a() void {
    var x: *addrspace(.physical_storage_buffer) i32 = undefined;
    _ = &x;
    var y: *addrspace(.physical_storage_buffer) i32 = undefined;
    _ = &y;
    var rt_cond = false;
    _ = &rt_cond;

    var z = if (rt_cond) x else y;
    _ = &z;
}

// compile
// output_mode=Obj
// backend=selfhosted
// target=spirv64-vulkan
// emit_bin=true
