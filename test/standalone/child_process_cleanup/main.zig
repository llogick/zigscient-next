const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    const child_exe = args[1];

    // Spawn, wait, and cancel a child process, with a deferred `kill` to handle termination/cleanup.
    const id = child: {
        var child = try std.process.spawn(io, .{ .argv = &.{child_exe} });
        defer child.kill(init.io);

        var wait_future = try init.io.concurrent(std.process.Child.wait, .{ &child, init.io });
        defer _ = wait_future.cancel(init.io) catch {};

        break :child switch (builtin.os.tag) {
            // On cleanup, the handle to the child process will be closed, so duplicate
            // it here in order to get a handle that will outlive this block.
            //
            // Another alternative would be to get the unique id here and use that with
            // OpenProcess, but that is more prone to racing the kernel on process close
            // and potential false positives with unique id re-use.
            .windows => try windowsDuplicateHandle(child.id.?),
            else => child.id.?,
        };
    };

    switch (builtin.target.os.tag) {
        .windows => {
            defer windows.CloseHandle(id);
            switch (windows.ntdll.NtTerminateProcess(id, @fromBackingInt(0))) {
                .SUCCESS => return error.ZombieChild,
                .PROCESS_IS_TERMINATING => {},
                else => |status| return windows.unexpectedStatus(status),
            }
        },
        else => {
            var status: if (builtin.link_libc) c_int else i32 = undefined;
            switch (std.posix.errno(std.posix.system.waitpid(id, &status, 0))) {
                .SUCCESS => return error.ZombieChild,
                .CHILD => {}, // already terminated/killed
                else => |err| return std.posix.unexpectedErrno(err),
            }
        },
    }
}

fn windowsDuplicateHandle(handle: windows.HANDLE) !windows.HANDLE {
    var dup_handle: windows.HANDLE = undefined;
    switch (NtDuplicateObject(
        windows.GetCurrentProcess(),
        handle,
        windows.GetCurrentProcess(),
        &dup_handle,
        .{},
        0,
        0x02 | 0x04, // SAME_ACCESS | SAME_ATTRIBUTES
    )) {
        .SUCCESS => return dup_handle,
        else => |status| return windows.unexpectedStatus(status),
    }
}

pub extern "ntdll" fn NtDuplicateObject(
    SourceProcessHandle: windows.HANDLE,
    SourceHandle: windows.HANDLE,
    TargetProcessHandle: ?windows.HANDLE,
    TargetHandle: ?*windows.HANDLE,
    DesiredAccess: windows.ACCESS_MASK,
    HandleAttributes: windows.ULONG,
    Options: windows.ULONG,
) callconv(.winapi) windows.NTSTATUS;
