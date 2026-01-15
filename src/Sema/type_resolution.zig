const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;

const Sema = @import("../Sema.zig");
const Block = Sema.Block;
const Type = @import("../Type.zig");
const Value = @import("../Value.zig");
const Zcu = @import("../Zcu.zig");
const CompileError = Zcu.CompileError;
const SemaError = Zcu.SemaError;
const LazySrcLoc = Zcu.LazySrcLoc;
const InternPool = @import("../InternPool.zig");
const Alignment = InternPool.Alignment;
const arith = @import("arith.zig");

/// Ensures that `ty` has known layout, including alignment, size, and (where relevant) field offsets.
/// `ty` may be any type; its layout is resolved *recursively* if necessary.
/// Adds incremental dependencies tracking any required type resolution.
/// MLUGG TODO: to make the langspec non-stupid, we need to call this from WAY fewer places (the conditions need to be less specific).
/// e.g. I think creating the type `fn (A, B) C` should force layout resolution of `A`,`B`,`C`, which will simplify some `analyzeCall` logic.
///      wait i just realised that's probably a terrible idea, fns are a common cause of dep loops rn... so maybe not lol idk...
///      perhaps "layout resolution" for a function should resolve layout of ret ty and stuff, idk. justification: the "layout" of a function is whether
///      fnHasRuntimeBits, which depends whether the ret ty is comptime-only, i.e. the ret ty layout
/// MLUGG TODO: to be clear, i should audit EVERY use of this before PRing
pub fn ensureLayoutResolved(sema: *Sema, ty: Type) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const ip = &zcu.intern_pool;
    switch (ip.indexToKey(ty.toIntern())) {
        .int_type,
        .ptr_type,
        .anyframe_type,
        .simple_type,
        .opaque_type,
        .enum_type,
        .error_set_type,
        .inferred_error_set_type,
        => {},

        .func_type => |func_type| {
            for (func_type.param_types.get(ip)) |param_ty| {
                try ensureLayoutResolved(sema, .fromInterned(param_ty));
            }
            try ensureLayoutResolved(sema, .fromInterned(func_type.return_type));
        },

        .array_type => |arr| return ensureLayoutResolved(sema, .fromInterned(arr.child)),
        .vector_type => |vec| return ensureLayoutResolved(sema, .fromInterned(vec.child)),
        .opt_type => |child| return ensureLayoutResolved(sema, .fromInterned(child)),
        .error_union_type => |eu| return ensureLayoutResolved(sema, .fromInterned(eu.payload_type)),
        .tuple_type => |tuple| for (tuple.types.get(ip)) |field_ty| {
            try ensureLayoutResolved(sema, .fromInterned(field_ty));
        },
        .struct_type, .union_type => {
            try sema.declareDependency(.{ .type_layout = ty.toIntern() });
            if (zcu.analysis_in_progress.contains(.wrap(.{ .type_layout = ty.toIntern() }))) {
                // TODO: better error message
                return sema.failWithOwnedErrorMsg(null, try sema.errMsg(
                    ty.srcLoc(zcu),
                    "{s} '{f}' depends on itself",
                    .{ @tagName(ty.zigTypeTag(zcu)), ty.fmt(pt) },
                ));
            }
            try pt.ensureTypeLayoutUpToDate(ty);
        },

        // values, not types
        .undef,
        .simple_value,
        .variable,
        .@"extern",
        .func,
        .int,
        .err,
        .error_union,
        .enum_literal,
        .enum_tag,
        .empty_enum_value,
        .float,
        .ptr,
        .slice,
        .opt,
        .aggregate,
        .un,
        // memoization, not types
        .memoized_call,
        => unreachable,
    }
}

/// Asserts that `ty` is either a `struct` type, or an `enum` type.
/// If `ty` is a struct, ensures that fields' default values are resolved.
/// If `ty` is an enum, ensures that fields' integer tag valus are resolved.
/// Adds incremental dependencies tracking the required type resolution.
pub fn ensureFieldInitsResolved(sema: *Sema, ty: Type) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const ip = &zcu.intern_pool;
    switch (ip.indexToKey(ty.toIntern())) {
        .struct_type, .enum_type => {},
        else => unreachable, // assertion failure
    }

    try sema.declareDependency(.{ .type_inits = ty.toIntern() });
    if (zcu.analysis_in_progress.contains(.wrap(.{ .type_inits = ty.toIntern() }))) {
        // TODO: better error message
        return sema.failWithOwnedErrorMsg(null, try sema.errMsg(
            ty.srcLoc(zcu),
            "{s} '{f}' depends on itself",
            .{ @tagName(ty.zigTypeTag(zcu)), ty.fmt(pt) },
        ));
    }
    try pt.ensureTypeInitsUpToDate(ty);
}
/// Asserts that `struct_ty` is a non-packed non-tuple struct, and that `sema.owner` is that type.
/// This function *does* register the `src_hash` dependency on the struct.
pub fn resolveStructLayout(sema: *Sema, struct_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_layout == struct_ty.toIntern());

    const struct_obj = ip.loadStructType(struct_ty.toIntern());
    const zir_index = struct_obj.zir_index.resolve(ip).?;

    assert(struct_obj.layout != .@"packed");

    try sema.declareDependency(.{ .src_hash = struct_obj.zir_index });

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = struct_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = struct_obj.zir_index,
        .type_name_ctx = struct_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    const zir_struct = sema.code.getStructDecl(zir_index);
    var field_it = zir_struct.iterateFields();
    while (field_it.next()) |zir_field| {
        const field_ty_src: LazySrcLoc = .{
            .base_node_inst = struct_obj.zir_index,
            .offset = .{ .container_field_type = zir_field.idx },
        };
        const field_align_src: LazySrcLoc = .{
            .base_node_inst = struct_obj.zir_index,
            .offset = .{ .container_field_align = zir_field.idx },
        };

        const field_ty: Type = field_ty: {
            block.comptime_reason = .{ .reason = .{
                .src = field_ty_src,
                .r = .{ .simple = .struct_field_types },
            } };
            const type_ref = try sema.resolveInlineBody(&block, zir_field.type_body, zir_index);
            break :field_ty try sema.analyzeAsType(&block, field_ty_src, type_ref);
        };
        assert(!field_ty.isGenericPoison());

        try sema.ensureLayoutResolved(field_ty);

        const explicit_field_align: Alignment = a: {
            block.comptime_reason = .{ .reason = .{
                .src = field_align_src,
                .r = .{ .simple = .struct_field_attrs },
            } };
            const align_body = zir_field.align_body orelse break :a .none;
            const align_ref = try sema.resolveInlineBody(&block, align_body, zir_index);
            break :a try sema.analyzeAsAlign(&block, field_align_src, align_ref);
        };

        if (field_ty.zigTypeTag(zcu) == .@"opaque") {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "cannot directly embed opaque type '{f}' in struct", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.errNote(field_ty_src, msg, "opaque types have unknown size", .{});
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        if (struct_obj.layout == .@"extern" and !try sema.validateExternType(field_ty, .struct_field)) {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "extern structs cannot contain fields of type '{f}'", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.explainWhyTypeIsNotExtern(msg, field_ty_src, field_ty, .struct_field);
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }

        struct_obj.field_types.get(ip)[zir_field.idx] = field_ty.toIntern();
        if (struct_obj.field_aligns.len != 0) {
            struct_obj.field_aligns.get(ip)[zir_field.idx] = explicit_field_align;
        } else {
            assert(explicit_field_align == .none);
        }
    }

    try finishStructLayout(sema, &block, struct_ty.srcLoc(zcu), struct_ty.toIntern(), &struct_obj);
}

/// Called after populating field types and alignments; populates field offsets, runtime order, and
/// overall struct layout information (size, alignment, comptime-only state, etc).
pub fn finishStructLayout(
    sema: *Sema,
    /// Only used to report compile errors.
    block: *Block,
    struct_src: LazySrcLoc,
    struct_ty: InternPool.Index,
    struct_obj: *const InternPool.LoadedStructType,
) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const io = comp.io;
    const ip = &zcu.intern_pool;
    var comptime_only = false;
    var one_possible_value = true;
    var struct_align: Alignment = .@"1";
    // Unlike `struct_obj.field_aligns`, these are not `.none`.
    const resolved_field_aligns = try sema.arena.alloc(Alignment, struct_obj.field_names.len);
    for (resolved_field_aligns, 0..) |*align_out, field_idx| {
        const field_ty: Type = .fromInterned(struct_obj.field_types.get(ip)[field_idx]);
        const field_align: Alignment = a: {
            if (struct_obj.field_aligns.len != 0) {
                const a = struct_obj.field_aligns.get(ip)[field_idx];
                if (a != .none) break :a a;
            }
            break :a field_ty.defaultStructFieldAlignment(struct_obj.layout, zcu);
        };
        if (!struct_obj.field_is_comptime_bits.get(ip, field_idx)) {
            // Non-`comptime` fields contribute to the struct's layout.
            struct_align = struct_align.maxStrict(field_align);
            if (field_ty.comptimeOnly(zcu)) comptime_only = true;
            if (try field_ty.onePossibleValue(pt) == null) one_possible_value = false;
            if (struct_obj.layout == .auto) {
                struct_obj.field_runtime_order.get(ip)[field_idx] = @enumFromInt(field_idx);
            }
        } else if (struct_obj.layout == .auto) {
            struct_obj.field_runtime_order.get(ip)[field_idx] = .omitted; // comptime fields are not in the runtime order
        }
        align_out.* = field_align;
    }
    if (struct_obj.layout == .auto) {
        const runtime_order = struct_obj.field_runtime_order.get(ip);
        // This logic does not reorder fields; it only moves the omitted ones to the end so that logic
        // elsewhere does not need to special-case. TODO: support field reordering in all the backends!
        if (!zcu.backendSupportsFeature(.field_reordering)) {
            var i: usize = 0;
            var off: usize = 0;
            while (i + off < runtime_order.len) {
                if (runtime_order[i + off] == .omitted) {
                    off += 1;
                } else {
                    runtime_order[i] = runtime_order[i + off];
                    i += 1;
                }
            }
        } else {
            // Sort by descending alignment to minimize padding.
            const RuntimeOrder = InternPool.LoadedStructType.RuntimeOrder;
            const AlignSortCtx = struct {
                aligns: []const Alignment,
                fn lessThan(ctx: @This(), a: RuntimeOrder, b: RuntimeOrder) bool {
                    assert(a != .unresolved);
                    assert(b != .unresolved);
                    if (a == .omitted) return false;
                    if (b == .omitted) return true;
                    const a_align = ctx.aligns[@intFromEnum(a)];
                    const b_align = ctx.aligns[@intFromEnum(b)];
                    return a_align.compare(.gt, b_align);
                }
            };
            mem.sortUnstable(
                RuntimeOrder,
                runtime_order,
                @as(AlignSortCtx, .{ .aligns = resolved_field_aligns }),
                AlignSortCtx.lessThan,
            );
        }
    }

    var runtime_order_it = struct_obj.iterateRuntimeOrder(ip);
    var cur_offset: u64 = 0;
    while (runtime_order_it.next()) |field_idx| {
        const field_ty: Type = .fromInterned(struct_obj.field_types.get(ip)[field_idx]);
        const offset = resolved_field_aligns[field_idx].forward(cur_offset);
        struct_obj.field_offsets.get(ip)[field_idx] = @truncate(offset); // truncate because the overflow is handled below
        cur_offset = offset + field_ty.abiSize(zcu);
    }
    const struct_size = std.math.cast(u32, struct_align.forward(cur_offset)) orelse return sema.fail(
        block,
        struct_src,
        "struct layout requires size {d}, this compiler implementation supports up to {d}",
        .{ struct_align.forward(cur_offset), std.math.maxInt(u32) },
    );
    ip.resolveStructLayout(
        io,
        struct_ty,
        struct_size,
        struct_align,
        false, // MLUGG TODO XXX NPV
        one_possible_value,
        comptime_only,
    );
}

/// Asserts that `struct_ty` is a packed struct, and that `sema.owner` is that type.
/// This function *does* register the `src_hash` dependency on the struct.
pub fn resolvePackedStructLayout(sema: *Sema, struct_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_layout == struct_ty.toIntern());

    const struct_obj = ip.loadStructType(struct_ty.toIntern());
    const zir_index = struct_obj.zir_index.resolve(ip).?;

    assert(struct_obj.layout == .@"packed");

    try sema.declareDependency(.{ .src_hash = struct_obj.zir_index });

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = struct_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = struct_obj.zir_index,
        .type_name_ctx = struct_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    var field_bits: u64 = 0;
    const zir_struct = sema.code.getStructDecl(zir_index);
    var field_it = zir_struct.iterateFields();
    while (field_it.next()) |zir_field| {
        const field_ty_src: LazySrcLoc = .{
            .base_node_inst = struct_obj.zir_index,
            .offset = .{ .container_field_type = zir_field.idx },
        };
        const field_ty: Type = field_ty: {
            block.comptime_reason = .{ .reason = .{
                .src = field_ty_src,
                .r = .{ .simple = .struct_field_types },
            } };
            const type_ref = try sema.resolveInlineBody(&block, zir_field.type_body, zir_index);
            break :field_ty try sema.analyzeAsType(&block, field_ty_src, type_ref);
        };
        assert(!field_ty.isGenericPoison());
        struct_obj.field_types.get(ip)[zir_field.idx] = field_ty.toIntern();

        try sema.ensureLayoutResolved(field_ty);

        if (field_ty.zigTypeTag(zcu) == .@"opaque") {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "cannot directly embed opaque type '{f}' in struct", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.errNote(field_ty_src, msg, "opaque types have unknown size", .{});
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        if (!field_ty.packable(zcu)) {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "packed structs cannot contain fields of type '{f}'", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.explainWhyTypeIsNotPackable(msg, field_ty_src, field_ty);
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        assert(!field_ty.comptimeOnly(zcu)); // packable types are not comptime-only
        field_bits += field_ty.bitSize(zcu);
    }

    try resolvePackedStructBackingInt(sema, &block, field_bits, struct_ty, &struct_obj);
}

pub fn resolvePackedStructBackingInt(
    sema: *Sema,
    block: *Block,
    field_bits: u64,
    struct_ty: Type,
    struct_obj: *const InternPool.LoadedStructType,
) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const io = comp.io;
    const ip = &zcu.intern_pool;

    switch (struct_obj.packed_backing_mode) {
        .explicit => {
            // We only need to validate the type.
            const backing_ty: Type = .fromInterned(struct_obj.packed_backing_int_type);
            assert(backing_ty.zigTypeTag(zcu) == .int);
            if (field_bits != backing_ty.intInfo(zcu).bits) return sema.failWithOwnedErrorMsg(block, msg: {
                const src = struct_ty.srcLoc(zcu);
                const msg = try sema.errMsg(src, "backing integer bit width does not match total bit width of fields", .{});
                errdefer msg.destroy(gpa);
                try sema.errNote(src, msg, "backing integer '{f}' has bit width '{d}'", .{ backing_ty.fmt(pt), backing_ty.bitSize(zcu) });
                try sema.errNote(src, msg, "struct fields have total bit width '{d}'", .{field_bits});
                break :msg msg;
            });
        },
        .auto => {
            // We need to generate the inferred tag.
            const want_bits = std.math.cast(u16, field_bits) orelse return sema.fail(
                block,
                struct_ty.srcLoc(zcu),
                "packed struct bit width '{d}' exceeds maximum bit width of 65535",
                .{field_bits},
            );
            const backing_int = try pt.intType(.unsigned, want_bits);
            ip.resolvePackedStructBackingInt(io, struct_ty.toIntern(), backing_int.toIntern());
        },
    }
}

/// Asserts that `struct_ty` is a non-tuple struct, and that `sema.owner` is that type.
/// This function *does* register the `src_hash` dependency on the struct.
pub fn resolveStructDefaults(sema: *Sema, struct_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_inits == struct_ty.toIntern());

    try sema.ensureLayoutResolved(struct_ty);

    const struct_obj = ip.loadStructType(struct_ty.toIntern());
    const zir_index = struct_obj.zir_index.resolve(ip).?;

    try sema.declareDependency(.{ .src_hash = struct_obj.zir_index });

    if (struct_obj.field_defaults.len == 0) {
        // The struct has no default field values, so the slice has been omitted.
        return;
    }

    const field_types = struct_obj.field_types.get(ip);

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = struct_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = struct_obj.zir_index,
        .type_name_ctx = struct_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    // We'll need to map the struct decl instruction to provide result types
    try sema.inst_map.ensureSpaceForInstructions(gpa, &.{zir_index});

    const zir_struct = sema.code.getStructDecl(zir_index);
    var field_it = zir_struct.iterateFields();
    while (field_it.next()) |zir_field| {
        const default_val_src: LazySrcLoc = .{
            .base_node_inst = struct_obj.zir_index,
            .offset = .{ .container_field_value = zir_field.idx },
        };
        block.comptime_reason = .{ .reason = .{
            .src = default_val_src,
            .r = .{ .simple = .struct_field_default_value },
        } };
        const default_body = zir_field.default_body orelse {
            struct_obj.field_defaults.get(ip)[zir_field.idx] = .none;
            continue;
        };
        const field_ty: Type = .fromInterned(field_types[zir_field.idx]);
        const uncoerced = ref: {
            // Provide the result type
            sema.inst_map.putAssumeCapacity(zir_index, .fromIntern(field_ty.toIntern()));
            defer assert(sema.inst_map.remove(zir_index));
            break :ref try sema.resolveInlineBody(&block, default_body, zir_index);
        };
        const coerced = try sema.coerce(&block, field_ty, uncoerced, default_val_src);
        const default_val = try sema.resolveConstValue(&block, default_val_src, coerced, null);
        if (default_val.canMutateComptimeVarState(zcu)) {
            const field_name = struct_obj.field_names.get(ip)[zir_field.idx];
            return sema.failWithContainsReferenceToComptimeVar(&block, default_val_src, field_name, "field default value", default_val);
        }
        struct_obj.field_defaults.get(ip)[zir_field.idx] = default_val.toIntern();
    }
}

/// This logic must be kept in sync with `Type.getUnionLayout`.
pub fn resolveUnionLayout(sema: *Sema, union_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_layout == union_ty.toIntern());

    const union_obj = ip.loadUnionType(union_ty.toIntern());
    const zir_index = union_obj.zir_index.resolve(ip).?;

    assert(union_obj.layout != .@"packed");

    try sema.declareDependency(.{ .src_hash = union_obj.zir_index });

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = union_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = union_obj.zir_index,
        .type_name_ctx = union_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    const zir_union = sema.code.getUnionDecl(zir_index);
    var field_it = zir_union.iterateFields();
    while (field_it.next()) |zir_field| {
        const field_ty_src: LazySrcLoc = .{
            .base_node_inst = union_obj.zir_index,
            .offset = .{ .container_field_type = zir_field.idx },
        };
        const field_align_src: LazySrcLoc = .{
            .base_node_inst = union_obj.zir_index,
            .offset = .{ .container_field_align = zir_field.idx },
        };

        const field_ty: Type = field_ty: {
            block.comptime_reason = .{ .reason = .{
                .src = field_ty_src,
                .r = .{ .simple = .union_field_types },
            } };
            const type_body = zir_field.type_body orelse break :field_ty .void;
            const type_ref = try sema.resolveInlineBody(&block, type_body, zir_index);
            break :field_ty try sema.analyzeAsType(&block, field_ty_src, type_ref);
        };
        assert(!field_ty.isGenericPoison());
        union_obj.field_types.get(ip)[zir_field.idx] = field_ty.toIntern();

        try sema.ensureLayoutResolved(field_ty);

        const explicit_field_align: Alignment = a: {
            block.comptime_reason = .{ .reason = .{
                .src = field_align_src,
                .r = .{ .simple = .union_field_attrs },
            } };
            const align_body = zir_field.align_body orelse break :a .none;
            const align_ref = try sema.resolveInlineBody(&block, align_body, zir_index);
            break :a try sema.analyzeAsAlign(&block, field_align_src, align_ref);
        };

        if (union_obj.field_aligns.len != 0) {
            union_obj.field_aligns.get(ip)[zir_field.idx] = explicit_field_align;
        } else {
            assert(explicit_field_align == .none);
        }

        if (field_ty.zigTypeTag(zcu) == .@"opaque") {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "cannot directly embed opaque type '{f}' in union", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.errNote(field_ty_src, msg, "opaque types have unknown size", .{});
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        if (union_obj.layout == .@"extern" and !try sema.validateExternType(field_ty, .union_field)) {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "extern unions cannot contain fields of type '{f}'", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.explainWhyTypeIsNotExtern(msg, field_ty_src, field_ty, .union_field);
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
    }

    try finishUnionLayout(
        sema,
        &block,
        union_ty.srcLoc(zcu),
        union_ty.toIntern(),
        &union_obj,
        .fromInterned(union_obj.enum_tag_type),
    );
}

/// Called after populating field types and alignments; populates overall union layout
/// information (size, alignment, comptime-only state, etc).
pub fn finishUnionLayout(
    sema: *Sema,
    /// Only used to report compile errors.
    block: *Block,
    union_src: LazySrcLoc,
    union_ty: InternPool.Index,
    union_obj: *const InternPool.LoadedUnionType,
    enum_tag_ty: Type,
) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const io = comp.io;
    const ip = &zcu.intern_pool;

    var payload_align: Alignment = .@"1";
    var payload_size: u64 = 0;
    var comptime_only = false;
    var possible_values: enum { none, one, many } = .none;
    for (0..union_obj.field_types.len) |field_idx| {
        const field_ty: Type = .fromInterned(union_obj.field_types.get(ip)[field_idx]);
        const field_align: Alignment = a: {
            if (union_obj.field_aligns.len != 0) {
                const a = union_obj.field_aligns.get(ip)[field_idx];
                if (a != .none) break :a a;
            }
            break :a field_ty.abiAlignment(zcu);
        };
        payload_align = payload_align.maxStrict(field_align);
        payload_size = @max(payload_size, field_ty.abiSize(zcu));
        if (field_ty.comptimeOnly(zcu)) comptime_only = true;
        if (!field_ty.isNoReturn(zcu)) {
            if (try field_ty.onePossibleValue(pt) != null) {
                possible_values = .many; // this field alone has many possible values
            } else switch (possible_values) {
                .none => possible_values = .one, // there were none, now there is this field's OPV
                .one => possible_values = .many, // there was one, now there are two
                .many => {},
            }
        }
    }

    const size: u64, const padding: u64, const alignment: Alignment = layout: {
        if (union_obj.runtime_tag == .none) {
            break :layout .{ payload_align.forward(payload_size), 0, payload_align };
        }
        const tag_align = enum_tag_ty.abiAlignment(zcu);
        const tag_size = enum_tag_ty.abiSize(zcu);
        // The layout will either be (tag, payload, padding) or (payload, tag, padding) depending on
        // which has larger alignment. So the overall size is just the tag and payload sizes, added,
        // and padded to the larger alignment.
        const alignment = tag_align.maxStrict(payload_align);
        const unpadded_size = tag_size + payload_size;
        const size = alignment.forward(unpadded_size);
        break :layout .{ size, size - unpadded_size, alignment };
    };

    const casted_size = std.math.cast(u32, size) orelse return sema.fail(
        block,
        union_src,
        "union layout requires size {d}, this compiler implementation supports up to {d}",
        .{ size, std.math.maxInt(u32) },
    );
    ip.resolveUnionLayout(
        io,
        union_ty,
        casted_size,
        @intCast(padding), // okay because padding is no greater than size
        alignment,
        possible_values == .none, // MLUGG TODO: make sure queries use `LoadedUnionType.has_no_possible_value`!
        possible_values == .one,
        comptime_only,
    );
}

pub fn resolvePackedUnionLayout(sema: *Sema, union_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_layout == union_ty.toIntern());

    const union_obj = ip.loadUnionType(union_ty.toIntern());
    const zir_index = union_obj.zir_index.resolve(ip).?;

    assert(union_obj.layout == .@"packed");

    try sema.declareDependency(.{ .src_hash = union_obj.zir_index });

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = union_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = union_obj.zir_index,
        .type_name_ctx = union_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    const zir_union = sema.code.getUnionDecl(zir_index);
    var field_it = zir_union.iterateFields();
    while (field_it.next()) |zir_field| {
        const field_ty_src: LazySrcLoc = .{
            .base_node_inst = union_obj.zir_index,
            .offset = .{ .container_field_type = zir_field.idx },
        };
        const field_ty: Type = field_ty: {
            block.comptime_reason = .{ .reason = .{
                .src = field_ty_src,
                .r = .{ .simple = .union_field_types },
            } };
            // MLUGG TODO: i think this should probably be a compile error? (if so, it's an astgen one, right?)
            const type_body = zir_field.type_body orelse break :field_ty .void;
            const type_ref = try sema.resolveInlineBody(&block, type_body, zir_index);
            break :field_ty try sema.analyzeAsType(&block, field_ty_src, type_ref);
        };
        assert(!field_ty.isGenericPoison());
        union_obj.field_types.get(ip)[zir_field.idx] = field_ty.toIntern();

        assert(zir_field.align_body == null); // packed union fields cannot be aligned
        assert(zir_field.value_body == null); // packed union fields cannot have tag values

        try sema.ensureLayoutResolved(field_ty);

        if (field_ty.zigTypeTag(zcu) == .@"opaque") {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "cannot directly embed opaque type '{f}' in union", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.errNote(field_ty_src, msg, "opaque types have unknown size", .{});
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        if (!field_ty.packable(zcu)) {
            return sema.failWithOwnedErrorMsg(&block, msg: {
                const msg = try sema.errMsg(field_ty_src, "packed unions cannot contain fields of type '{f}'", .{field_ty.fmt(pt)});
                errdefer msg.destroy(gpa);
                try sema.explainWhyTypeIsNotPackable(msg, field_ty_src, field_ty);
                try sema.addDeclaredHereNote(msg, field_ty);
                break :msg msg;
            });
        }
        assert(!field_ty.comptimeOnly(zcu)); // packable types are not comptime-only
    }

    try resolvePackedUnionBackingInt(sema, &block, union_ty, &union_obj, false);
}

/// MLUGG TODO doc comment; asserts all fields are resolved or whatever
pub fn resolvePackedUnionBackingInt(
    sema: *Sema,
    block: *Block,
    union_ty: Type,
    union_obj: *const InternPool.LoadedUnionType,
    is_reified: bool,
) SemaError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const io = comp.io;
    const ip = &zcu.intern_pool;
    switch (union_obj.packed_backing_mode) {
        .explicit => {
            const backing_int_type: Type = .fromInterned(union_obj.packed_backing_int_type);
            const backing_int_bits = backing_int_type.intInfo(zcu).bits;
            for (union_obj.field_types.get(ip), 0..) |field_type_ip, field_idx| {
                const field_type: Type = .fromInterned(field_type_ip);
                const field_bits = field_type.bitSize(zcu);
                if (field_bits != backing_int_bits) return sema.failWithOwnedErrorMsg(block, msg: {
                    const field_ty_src: LazySrcLoc = .{
                        .base_node_inst = union_obj.zir_index,
                        .offset = if (is_reified)
                            .nodeOffset(.zero)
                        else
                            .{ .container_field_type = @intCast(field_idx) },
                    };
                    const msg = try sema.errMsg(field_ty_src, "field bit width does not match backing integer", .{});
                    errdefer msg.destroy(gpa);
                    try sema.errNote(field_ty_src, msg, "field type '{f}' has bit width '{d}'", .{ field_type.fmt(pt), field_bits });
                    try sema.errNote(field_ty_src, msg, "backing integer '{f}' has bit width '{d}'", .{ backing_int_type.fmt(pt), backing_int_bits });
                    try sema.errNote(field_ty_src, msg, "all fields in a packed union must have the same bit width", .{});
                    break :msg msg;
                });
            }
        },
        .auto => switch (union_obj.field_types.len) {
            0 => ip.resolvePackedUnionBackingInt(io, union_ty.toIntern(), .u0_type),
            else => {
                const field_types = union_obj.field_types.get(ip);
                const first_field_type: Type = .fromInterned(field_types[0]);
                const first_field_bits = first_field_type.bitSize(zcu);
                for (field_types[1..], 1..) |field_type_ip, field_idx| {
                    const field_type: Type = .fromInterned(field_type_ip);
                    const field_bits = field_type.bitSize(zcu);
                    if (field_bits != first_field_bits) return sema.failWithOwnedErrorMsg(block, msg: {
                        const first_field_ty_src: LazySrcLoc = .{
                            .base_node_inst = union_obj.zir_index,
                            .offset = if (is_reified)
                                .nodeOffset(.zero)
                            else
                                .{ .container_field_type = 0 },
                        };
                        const field_ty_src: LazySrcLoc = .{
                            .base_node_inst = union_obj.zir_index,
                            .offset = if (is_reified)
                                .nodeOffset(.zero)
                            else
                                .{ .container_field_type = @intCast(field_idx) },
                        };
                        const msg = try sema.errMsg(field_ty_src, "field bit width does not match earlier field", .{});
                        errdefer msg.destroy(gpa);
                        try sema.errNote(field_ty_src, msg, "field type '{f}' has bit width '{d}'", .{ field_type.fmt(pt), field_bits });
                        try sema.errNote(first_field_ty_src, msg, "other field type '{f}' has bit width '{d}'", .{ first_field_type.fmt(pt), first_field_bits });
                        try sema.errNote(field_ty_src, msg, "all fields in a packed union must have the same bit width", .{});
                        break :msg msg;
                    });
                }
                const backing_int_bits = std.math.cast(u16, first_field_bits) orelse return sema.fail(
                    block,
                    block.nodeOffset(.zero),
                    "packed union bit width '{d}' exceeds maximum bit width of 65535",
                    .{first_field_bits},
                );
                const backing_int_type = try pt.intType(.unsigned, backing_int_bits);
                ip.resolvePackedUnionBackingInt(io, union_ty.toIntern(), backing_int_type.toIntern());
            },
        },
    }
}

/// Asserts that `enum_ty` is an enum and that `sema.owner` is that type.
/// This function *does* register the `src_hash` dependency on the enum.
pub fn resolveEnumValues(sema: *Sema, enum_ty: Type) CompileError!void {
    const pt = sema.pt;
    const zcu = pt.zcu;
    const comp = zcu.comp;
    const gpa = comp.gpa;
    const ip = &zcu.intern_pool;

    assert(sema.owner.unwrap().type_inits == enum_ty.toIntern());

    const enum_obj = ip.loadEnumType(enum_ty.toIntern());

    // We'll populate this map.
    const field_value_map = enum_obj.field_value_map.unwrap() orelse {
        // The enum has an automatically generated tag and is auto-numbered. We know that we have
        // generated a suitably large type in `analyzeEnumDecl`, so we have no work to do.
        return;
    };

    const maybe_parent_union_obj: ?InternPool.LoadedUnionType = un: {
        if (enum_obj.owner_union == .none) break :un null;
        break :un ip.loadUnionType(enum_obj.owner_union);
    };
    const tracked_inst = enum_obj.zir_index.unwrap() orelse maybe_parent_union_obj.?.zir_index;
    const zir_index = tracked_inst.resolve(ip).?;

    try sema.declareDependency(.{ .src_hash = tracked_inst });

    var block: Block = .{
        .parent = null,
        .sema = sema,
        .namespace = enum_obj.namespace,
        .instructions = .{},
        .inlining = null,
        .comptime_reason = undefined, // always set before using `block`
        .src_base_inst = tracked_inst,
        .type_name_ctx = enum_obj.name,
    };
    defer assert(block.instructions.items.len == 0);

    const int_tag_ty: Type = .fromInterned(enum_obj.int_tag_type);

    // Map the enum (or union) decl instruction to provide the tag type as the result type
    try sema.inst_map.ensureSpaceForInstructions(gpa, &.{zir_index});
    sema.inst_map.putAssumeCapacity(zir_index, .fromIntern(int_tag_ty.toIntern()));
    defer assert(sema.inst_map.remove(zir_index));

    // First, populate any explicitly provided values. This is the part that actually depends on
    // the ZIR, and hence depends on whether this is a declared or generated enum. If any explicit
    // value is invalid, we'll emit an error here.
    if (maybe_parent_union_obj) |union_obj| {
        const zir_union = sema.code.getUnionDecl(zir_index);
        var field_it = zir_union.iterateFields();
        while (field_it.next()) |zir_field| {
            const field_val_src: LazySrcLoc = .{
                .base_node_inst = union_obj.zir_index,
                .offset = .{ .container_field_value = zir_field.idx },
            };
            block.comptime_reason = .{ .reason = .{
                .src = field_val_src,
                .r = .{ .simple = .enum_field_values },
            } };
            const value_body = zir_field.value_body orelse {
                enum_obj.field_values.get(ip)[zir_field.idx] = .none;
                continue;
            };
            const uncoerced = try sema.resolveInlineBody(&block, value_body, zir_index);
            const coerced = try sema.coerce(&block, int_tag_ty, uncoerced, field_val_src);
            const val = try sema.resolveConstValue(&block, field_val_src, coerced, null);
            enum_obj.field_values.get(ip)[zir_field.idx] = val.toIntern();
        }
    } else {
        const zir_enum = sema.code.getEnumDecl(zir_index);
        var field_it = zir_enum.iterateFields();
        while (field_it.next()) |zir_field| {
            const field_val_src: LazySrcLoc = .{
                .base_node_inst = enum_obj.zir_index.unwrap().?,
                .offset = .{ .container_field_value = zir_field.idx },
            };
            block.comptime_reason = .{ .reason = .{
                .src = field_val_src,
                .r = .{ .simple = .enum_field_values },
            } };
            const value_body = zir_field.value_body orelse {
                enum_obj.field_values.get(ip)[zir_field.idx] = .none;
                continue;
            };
            const uncoerced = try sema.resolveInlineBody(&block, value_body, zir_index);
            const coerced = try sema.coerce(&block, int_tag_ty, uncoerced, field_val_src);
            const val = try sema.resolveConstDefinedValue(&block, field_val_src, coerced, null);
            enum_obj.field_values.get(ip)[zir_field.idx] = val.toIntern();
        }
    }

    // Explicit values are set. Now we'll go through the whole array and figure out the final
    // field values. This is also where we'll detect duplicates.

    for (0..enum_obj.field_names.len) |field_idx| {
        const field_val_src: LazySrcLoc = .{
            .base_node_inst = tracked_inst,
            .offset = .{ .container_field_value = @intCast(field_idx) },
        };
        // If the field value was not specified, compute the implicit value.
        const field_val = val: {
            const explicit_val = enum_obj.field_values.get(ip)[field_idx];
            if (explicit_val != .none) break :val explicit_val;
            if (field_idx == 0) {
                // Implicit value is 0, which is valid for every integer type.
                const val = (try pt.intValue(int_tag_ty, 0)).toIntern();
                enum_obj.field_values.get(ip)[field_idx] = val;
                break :val val;
            }
            // Implicit non-initial value: take the previous field value and add one.
            const prev_field_val: Value = .fromInterned(enum_obj.field_values.get(ip)[field_idx - 1]);
            const result = try arith.incrementDefinedInt(sema, int_tag_ty, prev_field_val);
            if (result.overflow) return sema.fail(
                &block,
                field_val_src,
                "enum tag value '{f}' too large for type '{f}'",
                .{ result.val.fmtValueSema(pt, sema), int_tag_ty.fmt(pt) },
            );
            const val = result.val.toIntern();
            enum_obj.field_values.get(ip)[field_idx] = val;
            break :val val;
        };
        const adapter: InternPool.Index.Adapter = .{ .indexes = enum_obj.field_values.get(ip)[0..field_idx] };
        const gop = field_value_map.get(ip).getOrPutAssumeCapacityAdapted(field_val, adapter);
        if (!gop.found_existing) continue;
        const prev_field_val_src: LazySrcLoc = .{
            .base_node_inst = tracked_inst,
            .offset = .{ .container_field_value = @intCast(gop.index) },
        };
        return sema.failWithOwnedErrorMsg(&block, msg: {
            const msg = try sema.errMsg(field_val_src, "enum tag value '{f}' already taken", .{
                Value.fromInterned(field_val).fmtValueSema(pt, sema),
            });
            errdefer msg.destroy(gpa);
            try sema.errNote(prev_field_val_src, msg, "previous occurrence here", .{});
            break :msg msg;
        });
    }

    if (enum_obj.nonexhaustive and int_tag_ty.toIntern() != .comptime_int_type) {
        const fields_len = enum_obj.field_names.len;
        if (fields_len >= 1 and std.math.log2_int(u64, fields_len) == int_tag_ty.bitSize(zcu)) {
            return sema.fail(&block, block.nodeOffset(.zero), "non-exhaustive enum specifies every value", .{});
        }
    }
}
