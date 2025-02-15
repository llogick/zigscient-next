const std = @import("std");

roots_info_file: []const u8,
deps_build_roots: []NamePathPair,
roots: []RootEntry,
packages: []NamePathPair,
include_dirs: []const []const u8,
top_level_steps: []const []const u8,
available_options: std.json.ArrayHashMap(AvailableOption),

pub const RootEntry = struct {
    name: []const u8,
    args: []const []const u8,
    mods: []NamePathPair,
};
pub const NamePathPair = struct {
    name: []const u8,
    path: []const u8,
};
pub const AvailableOption = std.meta.FieldType(std.meta.FieldType(std.Build, .available_options_map).KV, .value);
