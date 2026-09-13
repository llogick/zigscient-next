const std = @import("std");
const Ast = std.zig.Ast;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const compiler = @import("compiler");
const Compilation = compiler.Compilation;
const InternPool = Compilation.InternPool;
const Zcu = Compilation.Zcu;
const Air = Compilation.Air;

const Asta = @import("analysis.zig");
const ZigDoc = @import("ZigDoc.zig");
const tree_util = @import("offsets.zig");

const Aira = @This();

const ErrSet = error{OutOfMemory};

zdoc: *ZigDoc,
air: *Air,
active: Zcu.Active,

pub fn init(io: std.Io, zdoc: *ZigDoc, node: Ast.Node.Index) ?Aira {
    const doc_scope = try zdoc.getDocumentScope();
    const fn_scope = Asta.innermostScopeAtIndexWithTag(
        doc_scope,
        zdoc.tree.tokens.items(.start)[zdoc.tree.firstToken(node)],
        .initOne(.function),
    ).unwrap() orelse return null;
    const fn_node = doc_scope.getScopeAstNode(fn_scope) orelse return null;

    var buf: [1]Ast.Node.Index = undefined;
    const fn_proto_node = zdoc.tree.fullFnProto(&buf, fn_node).?.ast.proto_node;

    var cleanup: bool = false;

    zdoc.computed_data.lock.lockSharedUncancelable(io);
    defer if (cleanup) zdoc.computed_data.lock.unlockShared(io);

    const build = zdoc.computed_data.build orelse {
        cleanup = true;
        return null;
    };

    if (!build.mutex.tryLock()) {
        cleanup = true;
        return null;
    }
    defer if (cleanup) build.mutex.unlock(io);

    if (!build.has_completed_once) {
        cleanup = true;
        return null;
    }

    const args = zdoc.computed_data.air.get(fn_proto_node) orelse {
        cleanup = true;
        return null;
    };
    const zcu = build.compilation.?.zcu orelse {
        cleanup = true;
        return null;
    };

    return .{
        .zdoc = zdoc,
        .air = args.air,
        .active = zcu.activate(args.tid),
    };
}

pub fn deinit(aira: *Aira, io: std.Io) void {
    aira.active.deactivate();
    aira.zdoc.computed_data.build.?.mutex.unlock(io);
    aira.zdoc.computed_data.lock.unlockShared(io);
}

pub fn resolveVarDecl(
    arena: Allocator,
    io: std.Io,
    zdoc: *ZigDoc,
    node: Ast.Node.Index,
    fields_info_out: *std.ArrayList([]const u8),
) ErrSet!?[]const u8 {
    const tree = zdoc.tree;

    switch (tree.nodeTag(node)) {
        .global_var_decl,
        .local_var_decl,
        .aligned_var_decl,
        .simple_var_decl,
        => {},
        else => return null,
    }

    const full_var_decl = tree.fullVarDecl(node).?;

    // std.log.err("hovering over: {} with init node {}", .{ node, full_var_decl.ast.init_node.unwrap().? });
    const doc_scope = try zdoc.getDocumentScope();
    const fn_scope = Asta.innermostScopeAtIndexWithTag(doc_scope, tree.tokens.items(.start)[full_var_decl.firstToken()], .initOne(.function)).unwrap() orelse return null;
    const fn_node = doc_scope.getScopeAstNode(fn_scope) orelse return null;

    var buf: [1]Ast.Node.Index = undefined;
    const fn_proto_node = tree.fullFnProto(&buf, fn_node).?.ast.proto_node;

    zdoc.computed_data.lock.lockSharedUncancelable(io);
    defer zdoc.computed_data.lock.unlockShared(io);

    const build = zdoc.computed_data.build orelse return null;

    if (!build.mutex.tryLock()) return null;
    defer build.mutex.unlock(io);

    if (!build.has_completed_once) return null;

    const args = zdoc.computed_data.air.get(fn_proto_node) orelse return null;
    const zcu = build.compilation.?.zcu orelse return null;

    const active = zcu.activate(args.tid);
    defer active.deactivate();
    const pt = active.pt;

    if (matchVarDecl(
        full_var_decl.ast.init_node.unwrap().?,
        args.air,
        args.air.getMainBody(),
    )) |inst| return try resolveInst(
        arena,
        args.air,
        inst,
        pt,
        fields_info_out,
    );
    return null;
}

fn matchVarDecl(
    node: Ast.Node.Index,
    air: Air,
    instructions: []const Air.Inst.Index,
) ?Air.Inst.Index {
    for (instructions) |inst| {
        const tag = air.instructions.items(.tag)[@backingInt(inst)];
        switch (tag) {
            else => continue,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = air.instructions.items(.data)[@backingInt(inst)].pl_op;
                // TODO Check that the identifier matches as well
                // std.log.err("comparing: {} with {}", .{ pl_op.tree_data_index, @backingInt(node) });
                if (pl_op.tree_data_index == @backingInt(node)) {
                    // std.log.err("found a match! {}", .{pl_op.tree_data_index});
                    return inst;
                }
            },
            .loop,
            .block,
            => {
                if (matchVarDecl(node, air, air.unwrapBlock(inst).body)) |i| return i;
            },
            .dbg_inline_block => {
                if (matchVarDecl(node, air, air.unwrapDbgBlock(inst).body)) |i| return i;
            },
            .cond_br => {
                const cond_br = air.unwrapCondBr(inst);
                if (matchVarDecl(node, air, cond_br.then_body)) |i| return i;
                if (matchVarDecl(node, air, cond_br.else_body)) |i| return i;
            },
            .loop_switch_br,
            .switch_br,
            => {
                const switch_br = air.unwrapSwitch(inst);
                var it = switch_br.iterateCases();
                while (it.next()) |case| if (matchVarDecl(node, air, case.body)) |i| return i;
                if (matchVarDecl(node, air, it.elseBody())) |i| return i;
            },
        }
    }
    return null;
}

fn resolveInst(
    arena: Allocator,
    air: Air,
    instruction: Air.Inst.Index,
    pt: Zcu.PerThread,
    fields_info_out: *std.ArrayList([]const u8),
) ErrSet!?[]const u8 {
    const tags = air.instructions.items(.tag);
    const data = air.instructions.items(.data);

    var aw: std.Io.Writer.Allocating = .init(arena);
    defer aw.deinit();

    var inst = instruction;
    while (true) {
        const tag = tags[@backingInt(inst)];
        // std.log.err("tag: {t}", .{tags[@backingInt(inst)]});
        switch (tag) {
            else => return null,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = data[@backingInt(inst)].pl_op;
                if (@backingInt(pl_op.operand) < compiler.Compilation.InternPool.static_len) {
                    std.log.err("@{}", .{pl_op.operand});
                } else if (pl_op.operand.toInterned()) |ip_index| {
                    const ty = compiler.Compilation.Type.fromInterned(pt.zcu.intern_pool.indexToKey(ip_index).typeOf());
                    // std.log.err("<{f}, {f}>", .{
                    //     ty.fmt(pt),
                    //     compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                    // });
                    dumpFields(arena, pt, ty.ip_index, fields_info_out);
                    ty.print(&aw.writer, pt, null) catch return null;
                    return try aw.toOwnedSlice();
                } else {
                    if (pl_op.operand.toIndex()) |sub_inst| {
                        inst = sub_inst;
                        continue;
                    } else {
                        // std.log.err("unknown", .{});
                        return null;
                    }
                }
            },
            .alloc => {
                const ty = data[@backingInt(inst)].ty;
                assert(ty.zigTypeTag(pt.zcu) == .pointer);
                const child = Compilation.Type.childType(ty, pt.zcu);
                child.print(&aw.writer, pt, null) catch return null;
                // std.log.err("{s}", .{aw.written()[1..]});
                return try aw.toOwnedSlice();
            },
            .load => {
                const ty_op = data[@backingInt(inst)].ty_op;
                ty_op.ty.print(&aw.writer, pt, null) catch return null;
                // std.log.err("{s}", .{aw.written()});
                if (@backingInt(ty_op.ty.ip_index) < compiler.Compilation.InternPool.static_len) {
                    std.log.err("@{}", .{ty_op.ty.ip_index});
                } else {
                    dumpFields(arena, pt, ty_op.ty.ip_index, fields_info_out);
                }
                return try aw.toOwnedSlice();
            },
            .call => {
                const pl_op = data[@backingInt(inst)].pl_op;
                if (pl_op.operand.toInterned()) |ip_index| {
                    const ty = compiler.Compilation.Type.fromInterned(pt.zcu.intern_pool.indexToKey(ip_index).typeOf());
                    switch (pt.zcu.intern_pool.indexToKey(ty.ip_index)) {
                        else => {},
                        .func_type => |ft| {
                            // std.log.err("fn ret ty itk: {}", .{pt.zcu.intern_pool.indexToKey(ft.return_type)});
                            switch (pt.zcu.intern_pool.indexToKey(ft.return_type)) {
                                else => {},
                                .struct_type => dumpFields(arena, pt, ft.return_type, fields_info_out),
                                .ptr_type => |pty| {
                                    // std.log.err("pty child itk: {}", .{pt.zcu.intern_pool.indexToKey(pty.child)});
                                    switch (pt.zcu.intern_pool.indexToKey(pty.child)) {
                                        else => {},
                                        .enum_type,
                                        .struct_type,
                                        => dumpFields(arena, pt, pty.child, fields_info_out),
                                    }
                                },
                            }
                            aw.writer.print("{f}", .{compiler.Compilation.Type.fromInterned(ft.return_type).fmt(pt)}) catch return null;
                            return try aw.toOwnedSlice();
                        },
                    }
                    std.log.err("<{f}, {f}>", .{
                        ty.fmt(pt),
                        compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                    });
                }
                return null;
            },
            .block,
            .dbg_inline_block,
            => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                ty_pl.ty.print(&aw.writer, pt, null) catch return null;
                // std.log.err("{s}", .{aw.written()});
                return try aw.toOwnedSlice();
            },
        }
    }
}

fn dumpFields(
    arena: Allocator,
    pt: Zcu.PerThread,
    ip_index: InternPool.Index,
    fields_info_out: *std.ArrayList([]const u8),
) void {
    var idx = ip_index;
    sw: switch (pt.zcu.intern_pool.indexToKey(idx)) {
        else => {},
        .struct_type => {
            fields_info_out.append(arena, "\n\nInterned fields:\n```zig") catch @panic("OOM");
            const let = pt.zcu.intern_pool.loadStructType(idx);
            for (let.field_names.get(&pt.zcu.intern_pool), let.field_types.get(&pt.zcu.intern_pool)) |field_name, field_type_index| {
                const ty = compiler.Compilation.Type.fromInterned(field_type_index);
                fields_info_out.append(arena, arena.print("{s} : {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), ty.fmt(pt) }) catch @panic("OOM")) catch @panic("OOM");
                // std.log.err("{s} : {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), ty.fmt(pt) });
            }
            fields_info_out.append(arena, "\n```\npub fn decls:\n```zig") catch @panic("OOM");
            const namespace = pt.zcu.namespacePtr(let.namespace);
            for (namespace.pub_decls.keys()) |key| {
                const nav = pt.zcu.intern_pool.getNav(key);
                const rez = nav.srcInst(&pt.zcu.intern_pool).resolveFull(&pt.zcu.intern_pool) orelse continue;
                const file = pt.zcu.fileByIndex(rez.file);
                const file_uri = file.uri_slice orelse continue;
                const decl = std.zig.Zir.getDeclaration(file.zir.?, rez.inst);
                const src_node = decl.src_node;
                const zdoc = (pt.zcu.lsp_document_store.?.getOrLoadHandle(file_uri) catch continue) orelse continue;
                switch (zdoc.tree.nodeTag(src_node)) {
                    .fn_proto_simple,
                    .fn_proto_one,
                    .fn_proto_multi,
                    .fn_proto,
                    .fn_decl,
                    => {},
                    else => continue,
                }
                fields_info_out.append(arena, tree_util.nodeToSlice(&zdoc.tree, src_node)) catch @panic("OOM");
            }
            fields_info_out.append(arena, "\n```") catch @panic("OOM");
        },
        .enum_type => {
            fields_info_out.append(arena, "\n\nInterned fields:\n```zig") catch @panic("OOM");
            const let = pt.zcu.intern_pool.loadEnumType(idx);
            for (let.field_names.get(&pt.zcu.intern_pool), 0..) |field_name, i| {
                const field_values = let.field_values.get(&pt.zcu.intern_pool);
                if (i < field_values.len) {
                    const val = compiler.Compilation.Value.fromInterned(field_values[i]);
                    fields_info_out.append(arena, arena.print("{s} = {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), val.fmtValue(pt) }) catch @panic("OOM")) catch @panic("OOM");
                    // std.log.err("{s} = {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), val.fmtValue(pt) });
                } else {
                    fields_info_out.append(arena, arena.print("{s}", .{field_name.toSlice(&pt.zcu.intern_pool)}) catch @panic("OOM")) catch @panic("OOM");
                    // std.log.err("{s}", .{field_name.toSlice(&pt.zcu.intern_pool)});
                }
            }
            fields_info_out.append(arena, "\n```\npub fn decls:\n```zig") catch @panic("OOM");
            const namespace = pt.zcu.namespacePtr(let.namespace);
            for (namespace.pub_decls.keys()) |key| {
                const nav = pt.zcu.intern_pool.getNav(key);
                const rez = nav.srcInst(&pt.zcu.intern_pool).resolveFull(&pt.zcu.intern_pool) orelse continue;
                const file = pt.zcu.fileByIndex(rez.file);
                const file_uri = file.uri_slice orelse continue;
                const decl = std.zig.Zir.getDeclaration(file.zir.?, rez.inst);
                const src_node = decl.src_node;
                const zdoc = (pt.zcu.lsp_document_store.?.getOrLoadHandle(file_uri) catch continue) orelse continue;
                switch (zdoc.tree.nodeTag(src_node)) {
                    .fn_proto_simple,
                    .fn_proto_one,
                    .fn_proto_multi,
                    .fn_proto,
                    .fn_decl,
                    => {},
                    else => continue,
                }
                fields_info_out.append(arena, tree_util.nodeToSlice(&zdoc.tree, src_node)) catch @panic("OOM");
            }
            fields_info_out.append(arena, "```") catch @panic("OOM");
        },
        .union_type => {
            const let = pt.zcu.intern_pool.loadUnionType(idx);
            idx = let.enum_tag_type;
            continue :sw pt.zcu.intern_pool.indexToKey(let.enum_tag_type);
        },
    }
}
