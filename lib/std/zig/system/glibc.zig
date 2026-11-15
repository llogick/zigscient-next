const builtin = @import("builtin");
const std = @import("../../std.zig");

pub fn verFromLinkName(
    link_name: []const u8,
    prefix: []const u8,
) error{ UnrecognizedGnuLibCFileName, InvalidGnuLibCVersion }!std.SemanticVersion {
    // example: "libc-2.3.4.so"
    // example: "libc-2.27.so"
    // example: "ld-2.33.so"
    const suffix = ".so";
    if (!std.mem.startsWith(u8, link_name, prefix) or !std.mem.endsWith(u8, link_name, suffix)) {
        return error.UnrecognizedGnuLibCFileName;
    }
    // chop off "libc-" and ".so"
    const link_name_chopped = link_name[prefix.len .. link_name.len - suffix.len];
    return std.Target.Query.parseVersion(link_name_chopped) catch |err| switch (err) {
        error.Overflow => return error.InvalidGnuLibCVersion,
        error.InvalidVersion => return error.InvalidGnuLibCVersion,
    };
}

test verFromLinkName {
    try std.testing.expectError(error.UnrecognizedGnuLibCFileName, verFromLinkName("ld-2.37.so", "this-prefix-does-not-exist"));
    try std.testing.expectError(error.UnrecognizedGnuLibCFileName, verFromLinkName("libc-2.37.so-is-not-end", "libc-"));

    try std.testing.expectError(error.InvalidGnuLibCVersion, verFromLinkName("ld-2.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 0 }, try verFromLinkName("ld-2.37.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 0 }, try verFromLinkName("ld-2.37.0.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 1 }, try verFromLinkName("ld-2.37.1.so", "ld-"));
    try std.testing.expectError(error.InvalidGnuLibCVersion, verFromLinkName("ld-2.37.4.5.so", "ld-"));
}

const VerError = error{
    SystemResources,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    GLibCNotFound,
} || std.Io.Cancelable;

pub fn verFromRPath(io: std.Io, rpath: []const u8) VerError!std.SemanticVersion {
    const cwd: std.Io.Dir = .cwd();

    var dir = cwd.openDir(io, rpath, .{}) catch |err| switch (err) {
        error.SystemResources,
        error.SystemFdQuotaExceeded,
        error.ProcessFdQuotaExceeded,
        error.Canceled,
        => |e| return e,

        error.NetworkNotFound => unreachable, // Windows-only

        error.NameTooLong,
        error.BadPathName,
        error.FileNotFound,
        error.NotDir,
        error.AccessDenied,
        error.PermissionDenied,
        error.NoDevice,
        error.SymLinkLoop,
        error.Unexpected,
        => return error.GLibCNotFound,
    };
    defer dir.close(io);

    // Now we have a candidate for the path to libc shared object. In
    // the past, we used readlink() here because the link name would
    // reveal the glibc version. However, in more recent GNU/Linux
    // installations, there is no symlink. Thus we instead use a more
    // robust check of opening the libc shared object and reading the
    // release version out of its banner string.
    return verFromSoPath(io, dir, "libc.so.6");
}

pub fn verFromSoPath(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) VerError!std.SemanticVersion {
    var file = dir.openFile(io, sub_path, .{}) catch |err| switch (err) {
        error.SystemResources,
        error.SystemFdQuotaExceeded,
        error.ProcessFdQuotaExceeded,
        error.Canceled,
        => |e| return e,

        error.PipeBusy,
        error.NetworkNotFound,
        error.AntivirusInterference,
        => unreachable, // Windows-only

        error.NameTooLong,
        error.BadPathName,
        error.FileLocksUnsupported,
        error.NoSpaceLeft,
        error.PathAlreadyExists,
        error.DeviceBusy,
        error.FileBusy,
        error.ReadOnlyFileSystem,
        error.FileTooBig,
        error.WouldBlock,
        error.AccessDenied,
        error.PermissionDenied,
        error.FileNotFound,
        error.NotDir,
        error.IsDir,
        error.NoDevice,
        error.SymLinkLoop,
        error.Unexpected,
        => return error.GLibCNotFound,
    };
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var file_reader: std.Io.File.Reader = .init(file, io, &buffer);

    return verFromSoFile(&file_reader) catch |err| switch (err) {
        error.ReadFailed => switch (file_reader.err.?) {
            error.SystemResources,
            error.Canceled,
            => |e| return e,

            error.LockViolation => unreachable, // Windows-only

            error.InputOutput,
            error.IsDir,
            error.AccessDenied,
            error.WouldBlock,
            error.ConnectionResetByPeer,
            error.NotOpenForReading,
            error.SocketUnconnected,
            error.Unexpected,
            => return error.GLibCNotFound,
        },

        error.Canceled => |e| return e,

        error.EndOfStream,
        error.Unseekable,
        error.AccessDenied,
        error.Unexpected,

        error.InvalidElfMagic,
        error.InvalidElfEndian,
        error.InvalidElfClass,
        error.InvalidElfVersion,
        error.InvalidGnuLibCVersion,
        => return error.GLibCNotFound,
    };
}

/// Reads the glibc release version from the banner string that `csu/version.c` places in
/// `.rodata`, e.g. "GNU C Library (GNU libc) stable release version 2.39." or "GNU C Library (GNU libc) development release version 2.44.9000".
/// See https://sourceware.org/cgit/glibc/tree/csu/version.c#n26
pub fn verFromSoFile(file_reader: *std.Io.File.Reader) !std.SemanticVersion {
    const header = try std.elf.Header.read(&file_reader.interface);
    const str_section_off = header.shoff + @as(u64, header.shentsize) * @as(u64, header.shstrndx);
    try file_reader.seekTo(str_section_off);
    const shstr = try std.elf.takeSectionHeader(&file_reader.interface, header.is_64, header.endian);
    var strtab_buf: [4096]u8 = undefined;
    const shstrtab = strtab_buf[0..@min(shstr.sh_size, strtab_buf.len)];
    try file_reader.seekTo(shstr.sh_offset);
    try file_reader.interface.readSliceAll(shstrtab);
    const rodata: struct { offset: u64, size: u64 } = find_rodata: {
        var it = header.iterateSectionHeaders(file_reader);
        while (try it.next()) |shdr| {
            const end = std.mem.findScalarPos(u8, shstrtab, shdr.sh_name, 0) orelse continue;
            const sh_name = shstrtab[shdr.sh_name..end :0];
            if (std.mem.eql(u8, sh_name, ".rodata")) break :find_rodata .{
                .offset = shdr.sh_offset,
                .size = shdr.sh_size,
            };
        } else return error.InvalidGnuLibCVersion;
    };

    const needle = " release version ";
    var offset: u64 = 0;
    try file_reader.seekTo(rodata.offset);
    while (offset < rodata.size) {
        if (file_reader.interface.takeSentinel(0)) |s| {
            if (std.mem.indexOf(u8, s, needle)) |pos| {
                const chopped = s[pos + needle.len ..];
                const end = std.mem.findNone(u8, chopped, "0123456789.") orelse chopped.len;
                var version = std.Target.Query.parseVersion(std.mem.trimEnd(u8, chopped[0..end], ".")) catch |err| switch (err) {
                    error.Overflow => return error.InvalidGnuLibCVersion,
                    error.InvalidVersion => return error.InvalidGnuLibCVersion,
                };
                version.patch = 0;
                return version;
            }
            offset += s.len + 1;
        } else |err| switch (err) {
            error.EndOfStream => break,
            // `.rodata` also holds long non-zero byte sections.
            // Skip until we find C-strings.
            error.StreamTooLong => {
                const buffered = file_reader.interface.buffered();
                // always keep the last needle length in case of a partial match at the end
                const skipped = std.mem.find(u8, buffered, needle) orelse (buffered.len -| needle.len);
                if (skipped == 0) return error.InvalidGnuLibCVersion;
                file_reader.interface.toss(skipped);
                offset += skipped;
            },
            error.ReadFailed => |e| return e,
        }
    }

    return error.InvalidGnuLibCVersion;
}
