// This tests that switches allow negative values when compiling for SPIR-V.
// We use i32 to test the one-word OpSwitch literal path, which used to fail.

const E = enum(i32) { neg = -1, pos = 1 };

export fn intSwitch() void {
    var x: i32 = -1;
    _ = &x;
    var y: i32 = switch (x) {
        -1 => 1,
        else => 0,
    };
    _ = &y;
}

export fn intLoopSwitch() void {
    var x: i32 = -1;
    _ = &x;
    blk: switch (x) {
        -1 => continue :blk 0,
        else => {},
    }
}

export fn enumSwitch() void {
    var e: E = .neg;
    _ = &e;
    var y: i32 = switch (e) {
        .neg => 1,
        .pos => 2,
    };
    _ = &y;
}

export fn enumLoopSwitch() void {
    var e: E = .neg;
    _ = &e;
    blk: switch (e) {
        .neg => continue :blk .pos,
        .pos => {},
    }
}

// compile
// output_mode=Obj
// backend=selfhosted
// target=spirv64-opencl,spirv64-vulkan
// emit_bin=true
