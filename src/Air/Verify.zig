/// Verifies that AIR is valid, in that every instruction has valid operands and types. In compiler
/// builds with debug extensions, this is run on all AIR, both before `Air.Legalize` is run and (if
/// it is run) after it.
///
/// This verification pass is currently highly incomplete---expand it as needed.
const Verify = @This();

zcu: *Zcu,
func_index: InternPool.Index,
ret_ty: Type,
air: *const Air,
cur_inst: Air.Inst.Index,

pub fn run(pt: Zcu.PerThread, func_index: InternPool.Index, air: *const Air) enum { success, fail } {
    if (!@import("build_options").enable_debug_extensions) {
        // `Air.Verify` is a debugging feature---it should not be used in release builds because it
        // has little benefit and negatively affects compiler performance.
        return .success;
    }

    const zcu = pt.zcu;

    const func_ty: Type = Value.fromInterned(func_index).typeOf(zcu);
    const ret_ty = func_ty.fnReturnType(zcu);

    var verify: Verify = .{
        .zcu = zcu,
        .func_index = func_index,
        .ret_ty = ret_ty,
        .air = air,
        .cur_inst = undefined, // populated by `body(...)`
    };
    verify.body(air.getMainBody()) catch |verify_err| switch (verify_err) {
        error.VerifyFail => {
            const ip = &zcu.intern_pool;
            const func_nav = ip.indexToKey(func_index).func.owner_nav;
            const ap: Air.Print = .{
                .zcu = zcu,
                .fqn = ip.getNav(func_nav).fqn,
                .air = air,
                .liveness = &null,
            };
            log.info("{f}", .{ap});
            return .fail;
        },
    };
    return .success;
}

const Error = error{VerifyFail};

fn fail(verify: *Verify, msg: []const u8) Error {
    const ip = &verify.zcu.intern_pool;
    const func_nav = ip.indexToKey(verify.func_index).func.owner_nav;
    const func_fqn = ip.getNav(func_nav).fqn.toSlice(ip);
    log.err("'{s}', %{d}: {s}", .{ func_fqn, verify.cur_inst, msg });
    return error.VerifyFail;
}

fn body(verify: *Verify, body_insts: []const Air.Inst.Index) Error!void {
    const zcu = verify.zcu;
    const ip = &zcu.intern_pool;
    const air = verify.air;
    const tags = air.instructions.items(.tag);
    const data = air.instructions.items(.data);
    for (body_insts, 0..) |inst, body_index| {
        verify.cur_inst = inst;
        switch (tags[@backingInt(inst)]) {
            .block => {
                const block = air.unwrapBlock(inst);
                try verify.body(block.body);
            },
            .dbg_inline_block => {
                const block = air.unwrapDbgBlock(inst);
                try verify.body(block.body);
            },
            .@"try", .try_cold => {
                const @"try" = air.unwrapTry(inst);
                try verify.body(@"try".else_body);
            },
            .try_ptr, .try_ptr_cold => {
                const try_ptr = air.unwrapTryPtr(inst);
                try verify.body(try_ptr.else_body);
            },
            .loop => {
                const block = air.unwrapBlock(inst);
                try verify.body(block.body);
            },
            .cond_br => {
                const cond_br = air.unwrapCondBr(inst);
                try verify.body(cond_br.then_body);
                try verify.body(cond_br.else_body);
            },
            .switch_br, .loop_switch_br => {
                const switch_br = air.unwrapSwitch(inst);
                var it = switch_br.iterateCases();
                while (it.next()) |case| {
                    try verify.body(case.body);
                }
                const else_body = it.elseBody();
                if (else_body.len > 0) {
                    try verify.body(else_body);
                }
            },
            .ret, .ret_safe => {
                const operand = data[@backingInt(inst)].un_op;
                if (air.typeOf(operand, ip).toIntern() != verify.ret_ty.toIntern()) return verify.fail("bad return type");
            },
            .ret_load => {
                const operand = data[@backingInt(inst)].un_op;
                const ptr_ty = air.typeOf(operand, ip);
                if (ptr_ty.zigTypeTag(zcu) != .pointer) return verify.fail("operand is not a pointer");
                if (ptr_ty.ptrSize(zcu) != .one) return verify.fail("pointer size is not '.one'");
                if (ptr_ty.childType(zcu).toIntern() != verify.ret_ty.toIntern()) return verify.fail("bad return type");
            },

            .bit_cast, .bit_cast_safe => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                // Enums are allowed here even if their backing type is implicit.
                if (!operand_ty.hasBitRepresentation(zcu) and operand_ty.zigTypeTag(zcu) != .@"enum") {
                    return verify.fail("bad operand type");
                }
                if (!result_ty.hasBitRepresentation(zcu) and result_ty.zigTypeTag(zcu) != .@"enum") {
                    return verify.fail("bad result type");
                }
                if (operand_ty.isPtrAtRuntime(zcu)) return verify.fail("bad operand type (pointer)");
                if (result_ty.isPtrAtRuntime(zcu)) return verify.fail("bad result type (pointer)");
                if (operand_ty.bitSize(zcu) != result_ty.bitSize(zcu)) return verify.fail("bit size mismatch");
            },
            .ptr_cast => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                const result_scalar_ty = result_ty.scalarType(zcu);
                if (operand_ty.isSliceAtRuntime(zcu)) {
                    if (!result_ty.isSliceAtRuntime(zcu)) return verify.fail("operand is slice, but result is not");
                } else {
                    if (!operand_scalar_ty.isPtrAtRuntime(zcu)) return verify.fail("bad operand type");
                    if (!result_scalar_ty.isPtrAtRuntime(zcu)) return verify.fail("operand is pointer, but result is not");
                    if (operand_ty.isVector(zcu) and !result_ty.isVector(zcu)) return verify.fail("operand is vector, but result is not");
                    if (!operand_ty.isVector(zcu) and result_ty.isVector(zcu)) return verify.fail("result is vector, but operand is not");
                }
                if (operand_scalar_ty.ptrAddressSpace(zcu) != result_scalar_ty.ptrAddressSpace(zcu)) {
                    return verify.fail("illegal change to address space");
                }
            },
            .ptr_from_int => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                const result_scalar_ty = result_ty.scalarType(zcu);
                if (operand_scalar_ty.toIntern() != .usize_type) return verify.fail("bad operand type");
                if (!result_scalar_ty.isPtrAtRuntime(zcu)) return verify.fail("bad result type");
                if (operand_ty.isVector(zcu) and !result_ty.isVector(zcu)) return verify.fail("operand is vector, but result is not");
                if (!operand_ty.isVector(zcu) and result_ty.isVector(zcu)) return verify.fail("result is vector, but operand is not");
            },
            .int_from_ptr => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                const result_scalar_ty = result_ty.scalarType(zcu);
                if (!operand_scalar_ty.isPtrAtRuntime(zcu)) return verify.fail("bad operand type");
                if (result_scalar_ty.toIntern() != .usize_type) return verify.fail("bad result type");
                if (operand_ty.isVector(zcu) and !result_ty.isVector(zcu)) return verify.fail("operand is vector, but result is not");
                if (!operand_ty.isVector(zcu) and result_ty.isVector(zcu)) return verify.fail("result is vector, but operand is not");
            },
            .error_cast => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                switch (operand_ty.zigTypeTag(zcu)) {
                    else => return verify.fail("bad operand type"),
                    .error_union => {
                        if (result_ty.zigTypeTag(zcu) != .error_union) {
                            return verify.fail("operand is error union, but result is not");
                        }
                        if (operand_ty.errorUnionPayload(zcu).toIntern() != result_ty.errorUnionPayload(zcu).toIntern()) {
                            return verify.fail("error union payload type differs");
                        }
                    },
                    .error_set => if (result_ty.zigTypeTag(zcu) != .error_set) {
                        return verify.fail("operand is error set, but result is not");
                    },
                }
            },
            .error_from_int => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                if (!operand_ty.isUnsignedInt(zcu)) return verify.fail("bad operand type");
                if (operand_ty.bitSize(zcu) != zcu.errorSetBits()) return verify.fail("bad operand bit size");
                if (result_ty.zigTypeTag(zcu) != .error_set) return verify.fail("bad result type");
            },
            .int_from_error => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                if (operand_ty.zigTypeTag(zcu) != .error_set) return verify.fail("bad operand type");
                if (!result_ty.isUnsignedInt(zcu)) return verify.fail("bad result type");
                if (result_ty.bitSize(zcu) != zcu.errorSetBits()) return verify.fail("bad result bit size");
            },
            .union_from_enum => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                if (operand_ty.zigTypeTag(zcu) != .@"enum") return verify.fail("bad operand type");
                if (result_ty.zigTypeTag(zcu) != .@"union") return verify.fail("bad result type");
                const union_tag_ty = result_ty.unionTagType(zcu) orelse return verify.fail("union type is not tagged");
                if (union_tag_ty.toIntern() != operand_ty.toIntern()) return verify.fail("union tag type does not match operand type");
            },

            .ptr_elem_ptr => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const bin_op = air.extraData(Air.Bin, ty_pl.payload).data;
                const ptr_ty = air.typeOf(bin_op.lhs, ip);
                const result_ty = ty_pl.ty;
                if (ptr_ty.zigTypeTag(zcu) != .pointer) return verify.fail("bad pointer type");
                if (result_ty.zigTypeTag(zcu) != .pointer) return verify.fail("bad result type");
                const ptr_info = ptr_ty.ptrInfo(zcu);
                const result_ptr_info = result_ty.ptrInfo(zcu);
                if (ptr_info.packed_offset.host_size != 0) return verify.fail("pointer type is bitpacked pointer");
                if (result_ptr_info.packed_offset.host_size != 0) return verify.fail("result type is bitpacked pointer");
            },

            .abs => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                const operand_is_vector = operand_ty.zigTypeTag(zcu) == .vector;
                const result_is_vector = result_ty.zigTypeTag(zcu) == .vector;
                if (operand_is_vector != result_is_vector) return verify.fail("vector/scalar mismatch");
                if (operand_is_vector and operand_ty.vectorLen(zcu) != result_ty.vectorLen(zcu)) {
                    return verify.fail("vector length mismatch");
                }
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                const result_scalar_ty = result_ty.scalarType(zcu);
                switch (operand_scalar_ty.zigTypeTag(zcu)) {
                    else => unreachable,
                    .int => {
                        const operand_int_info = operand_scalar_ty.intInfo(zcu);
                        if (operand_int_info.signedness != .signed) return verify.fail("operand is unsigned");
                        if (!result_scalar_ty.isUnsignedInt(zcu)) return verify.fail("result type is non-integer or is signed");
                        if (result_scalar_ty.intInfo(zcu).bits != operand_int_info.bits) {
                            return verify.fail("operand/result bit count mismatch");
                        }
                    },
                    .float => {
                        if (result_scalar_ty.toIntern() != operand_scalar_ty.toIntern()) {
                            return verify.fail("operand/result float type mismatch");
                        }
                    },
                }
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .not => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                const result_ty = ty_op.ty;
                if (operand_ty.toIntern() != result_ty.toIntern()) return verify.fail("operand/result type mismatch");
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                switch (operand_scalar_ty.zigTypeTag(zcu)) {
                    .int, .bool => {},
                    else => return verify.fail("scalar operand type is not int or bool"),
                }
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .mul_add => {
                const pl_op = data[@backingInt(inst)].pl_op;
                // (lhs * rhs) + addend
                const bin = air.extraData(Air.Bin, pl_op.payload).data;
                const addend = pl_op.operand;
                const operand_ty = air.typeOf(addend, ip);
                if (air.typeOf(bin.lhs, ip).toIntern() != operand_ty.toIntern()) return verify.fail("operand type mismatch");
                if (air.typeOf(bin.rhs, ip).toIntern() != operand_ty.toIntern()) return verify.fail("operand type mismatch");
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                if (operand_scalar_ty.zigTypeTag(zcu) != .float) {
                    return verify.fail("operand type is not a float");
                }
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // float unary ops
            .sqrt,
            .sin,
            .cos,
            .tan,
            .exp,
            .exp2,
            .log,
            .log2,
            .log10,
            .floor,
            .ceil,
            .round,
            .trunc_float,
            .neg,
            .neg_optimized,
            => {
                const operand = data[@backingInt(inst)].un_op;
                const operand_ty = air.typeOf(operand, ip);
                if (operand_ty.scalarType(zcu).zigTypeTag(zcu) != .float) return verify.fail("operand is not a float");
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // int/float binary ops
            .add,
            .add_safe,
            .sub,
            .sub_safe,
            .mul,
            .mul_safe,
            .div_trunc,
            .div_floor,
            .div_ceil,
            .div_exact,
            .rem,
            .mod,
            .max,
            .min,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                switch (lhs_ty.scalarType(zcu).zigTypeTag(zcu)) {
                    .float, .int => {},
                    else => return verify.fail("bad operand scalar type"),
                }
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // float binary ops
            .add_optimized,
            .sub_optimized,
            .mul_optimized,
            .div_float,
            .div_float_optimized,
            .div_trunc_optimized,
            .div_floor_optimized,
            .div_ceil_optimized,
            .div_exact_optimized,
            .rem_optimized,
            .mod_optimized,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                if (lhs_ty.scalarType(zcu).zigTypeTag(zcu) != .float) return verify.fail("bad operand scalar type");
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // int binary ops
            .add_wrap,
            .add_sat,
            .sub_wrap,
            .sub_sat,
            .mul_wrap,
            .mul_sat,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                if (lhs_ty.scalarType(zcu).zigTypeTag(zcu) != .int) return verify.fail("bad operand scalar type");
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // int/bool binary ops
            .bit_and,
            .bit_or,
            .xor,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                switch (lhs_ty.scalarType(zcu).zigTypeTag(zcu)) {
                    .bool, .int => {},
                    else => return verify.fail("bad operand scalar type"),
                }
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // overflow arithmetic
            .add_with_overflow,
            .sub_with_overflow,
            .mul_with_overflow,
            => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const bin = air.extraData(Air.Bin, ty_pl.payload).data;
                const lhs_ty = air.typeOf(bin.lhs, ip);
                const rhs_ty = air.typeOf(bin.rhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");

                const lhs_scalar_ty = lhs_ty.scalarType(zcu);
                if (lhs_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("lhs is not an integer");

                const result_ty = ty_pl.ty;
                if (!result_ty.isTuple(zcu)) return verify.fail("result type is not a tuple");
                if (result_ty.structFieldCount(zcu) != 2) return verify.fail("result tuple len != 2");
                const result_int_ty = result_ty.fieldType(0, zcu);
                const result_overflow_ty = result_ty.fieldType(1, zcu);

                if (result_int_ty.toIntern() != lhs_ty.toIntern()) return verify.fail("operand/result int type mismatch");

                if (lhs_ty.zigTypeTag(zcu) == .vector) {
                    if (result_overflow_ty.zigTypeTag(zcu) != .vector) return verify.fail("result int is vector but result overflow is scalar");
                    if (result_overflow_ty.vectorLen(zcu) != lhs_ty.vectorLen(zcu)) return verify.fail("vector len mismatch");
                    if (result_overflow_ty.childType(zcu).toIntern() != .u1_type) return verify.fail("result overflow is not vector of u1");
                } else {
                    if (result_overflow_ty.toIntern() != .u1_type) return verify.fail("result overflow is not u1");
                }

                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .ptr_add,
            .ptr_sub,
            => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const bin = air.extraData(Air.Bin, ty_pl.payload).data;
                const result_ty = ty_pl.ty;
                const lhs_ty = air.typeOf(bin.lhs, ip);
                const rhs_ty = air.typeOf(bin.rhs, ip);
                if (rhs_ty.toIntern() != .usize_type) return verify.fail("rhs is not usize");
                if (lhs_ty.zigTypeTag(zcu) != .pointer) return verify.fail("lhs is not a pointer");
                if (result_ty.zigTypeTag(zcu) != .pointer) return verify.fail("result is not a pointer");

                const lhs_ptr_info = lhs_ty.ptrInfo(zcu);
                const result_ptr_info = result_ty.ptrInfo(zcu);

                if (lhs_ptr_info.child != result_ptr_info.child) return verify.fail("lhs/result pointer child mismatch");
                if (lhs_ptr_info.sentinel != result_ptr_info.sentinel) return verify.fail("lhs/result pointer sentinel mismatch");
                if (lhs_ptr_info.flags.size != result_ptr_info.flags.size) return verify.fail("lhs/result pointer size mismatch");
                if (lhs_ptr_info.flags.is_const != result_ptr_info.flags.is_const) return verify.fail("lhs/result pointer const mismatch");
                if (lhs_ptr_info.flags.is_volatile != result_ptr_info.flags.is_volatile) return verify.fail("lhs/result pointer volatile mismatch");
                if (lhs_ptr_info.flags.is_allowzero != result_ptr_info.flags.is_allowzero) return verify.fail("lhs/result pointer allowzero mismatch");
                if (lhs_ptr_info.flags.address_space != result_ptr_info.flags.address_space) return verify.fail("lhs/result pointer addrspace mismatch");
                if (result_ty.ptrAlignment(zcu).compareStrict(.gt, lhs_ty.ptrAlignment(zcu))) return verify.fail("result pointer has greater alignment than lhs pointer");

                if (lhs_ptr_info.packed_offset.host_size != 0) return verify.fail("lhs is a bit pointer");
                if (result_ptr_info.packed_offset.host_size != 0) return verify.fail("lhs is a bit pointer");
                assert(lhs_ptr_info.packed_offset.bit_offset == 0);
                assert(result_ptr_info.packed_offset.bit_offset == 0);
                assert(lhs_ptr_info.flags.vector_index == .none);
                assert(result_ptr_info.flags.vector_index == .none);

                switch (lhs_ptr_info.flags.size) {
                    .one => return verify.fail("lhs is a single-item pointer"),
                    .c, .many, .slice => {},
                }

                const elem_ty: Type = .fromInterned(lhs_ptr_info.child);
                if (!elem_ty.classify(zcu).hasRuntimeBits()) return verify.fail("ptr arithmetic is a nop");
            },

            // shifts where rhs is log2 type of lhs
            .shr,
            .shr_exact,
            .shl,
            .shl_exact,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.zigTypeTag(zcu) == .vector) {
                    if (rhs_ty.zigTypeTag(zcu) != .vector) return verify.fail("lhs/rhs vector mismatch");
                    if (rhs_ty.vectorLen(zcu) != lhs_ty.vectorLen(zcu)) return verify.fail("vector len mismatch");
                } else {
                    if (rhs_ty.zigTypeTag(zcu) == .vector) return verify.fail("lhs/rhs vector mismatch");
                }
                const lhs_scalar_ty = lhs_ty.scalarType(zcu);
                const rhs_scalar_ty = rhs_ty.scalarType(zcu);
                if (lhs_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("lhs is not an integer");
                if (!rhs_scalar_ty.isUnsignedInt(zcu)) return verify.fail("rhs is not an unsigned integer");
                const lhs_bits = lhs_scalar_ty.intInfo(zcu).bits;
                const rhs_bits = rhs_scalar_ty.intInfo(zcu).bits;
                if (rhs_bits != std.math.log2_int_ceil(u16, lhs_bits)) {
                    return verify.fail("rhs bit count is not log2 of lhs bit count");
                }
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("lhs type lacks runtime bits or is comptime-only");
                }
            },

            .shl_with_overflow => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const bin = air.extraData(Air.Bin, ty_pl.payload).data;
                const lhs_ty = air.typeOf(bin.lhs, ip);
                const rhs_ty = air.typeOf(bin.rhs, ip);

                const lhs_scalar_ty = lhs_ty.scalarType(zcu);
                const rhs_scalar_ty = rhs_ty.scalarType(zcu);
                if (lhs_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("lhs is not an integer");
                if (!rhs_scalar_ty.isUnsignedInt(zcu)) return verify.fail("rhs is not an unsigned integer");

                const result_ty = ty_pl.ty;
                if (!result_ty.isTuple(zcu)) return verify.fail("result type is not a tuple");
                if (result_ty.structFieldCount(zcu) != 2) return verify.fail("result tuple len != 2");
                const result_int_ty = result_ty.fieldType(0, zcu);
                const result_overflow_ty = result_ty.fieldType(1, zcu);

                if (result_int_ty.toIntern() != lhs_ty.toIntern()) return verify.fail("operand/result int type mismatch");

                if (lhs_ty.zigTypeTag(zcu) == .vector) {
                    if (result_overflow_ty.zigTypeTag(zcu) != .vector) return verify.fail("result int is vector but result overflow is scalar");
                    if (result_overflow_ty.vectorLen(zcu) != lhs_ty.vectorLen(zcu)) return verify.fail("vector len mismatch");
                    if (result_overflow_ty.childType(zcu).toIntern() != .u1_type) return verify.fail("result overflow is not vector of u1");

                    if (rhs_ty.zigTypeTag(zcu) != .vector) return verify.fail("lhs is vector but rhs is not");
                    if (rhs_ty.vectorLen(zcu) != lhs_ty.vectorLen(zcu)) return verify.fail("vector len mismatch");
                } else {
                    if (result_overflow_ty.toIntern() != .u1_type) return verify.fail("result overflow is not u1");
                    if (rhs_ty.zigTypeTag(zcu) == .vector) return verify.fail("rhs is vector but lhs is not");
                }

                const lhs_bits = lhs_scalar_ty.intInfo(zcu).bits;
                const rhs_bits = rhs_scalar_ty.intInfo(zcu).bits;
                if (rhs_bits != std.math.log2_int_ceil(u16, lhs_bits)) {
                    return verify.fail("rhs bit count is not log2 of lhs bit count");
                }

                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // in a saturating shift, the rhs may be any unsigned integer
            .shl_sat => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.rhs, ip);
                if (lhs_ty.zigTypeTag(zcu) == .vector) {
                    if (rhs_ty.zigTypeTag(zcu) != .vector) return verify.fail("lhs/rhs vector mismatch");
                    if (rhs_ty.vectorLen(zcu) != lhs_ty.vectorLen(zcu)) return verify.fail("vector len mismatch");
                } else {
                    if (rhs_ty.zigTypeTag(zcu) == .vector) return verify.fail("lhs/rhs vector mismatch");
                }
                const lhs_scalar_ty = lhs_ty.scalarType(zcu);
                const rhs_scalar_ty = rhs_ty.scalarType(zcu);
                if (lhs_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("lhs is not an integer");
                if (!rhs_scalar_ty.isUnsignedInt(zcu)) return verify.fail("rhs is not an unsigned integer");
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("lhs type lacks runtime bits or is comptime-only");
                }
                if (rhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("rhs type lacks runtime bits or is comptime-only");
                }
            },

            // int comparisons
            .cmp_lt,
            .cmp_lte,
            .cmp_gte,
            .cmp_gt,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.lhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                switch (lhs_ty.zigTypeTag(zcu)) {
                    .int, .float => {},
                    .pointer => if (lhs_ty.isSlice(zcu)) {
                        return verify.fail("comparison type is not int or float or pointer");
                    },
                    else => return verify.fail("comparison type is not int or float or pointer"),
                }
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // float comparisons
            .cmp_lt_optimized,
            .cmp_lte_optimized,
            .cmp_eq_optimized,
            .cmp_gte_optimized,
            .cmp_gt_optimized,
            .cmp_neq_optimized,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.lhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                if (lhs_ty.zigTypeTag(zcu) != .float) {
                    return verify.fail("optimized comparison type is not float");
                }
            },

            // weird comparisons
            .cmp_eq,
            .cmp_neq,
            => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const lhs_ty = air.typeOf(bin_op.lhs, ip);
                const rhs_ty = air.typeOf(bin_op.lhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");

                var cur_ty = lhs_ty;
                while (true) switch (cur_ty.zigTypeTag(zcu)) {
                    .optional => cur_ty = cur_ty.optionalChild(zcu),
                    .bool, .int, .float, .void => break,
                    .pointer => if (!cur_ty.isSlice(zcu)) break else {
                        return verify.fail("invalid comparison operand type (slice)");
                    },
                    else => if (cur_ty.isAbiInt(zcu)) break else {
                        return verify.fail("invalid comparison operand type");
                    },
                };

                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            // vector comparisons
            .cmp_vector,
            .cmp_vector_optimized,
            => |tag| {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const vector_cmp = air.extraData(Air.VectorCmp, ty_pl.payload).data;
                const result_ty = ty_pl.ty;
                const lhs_ty = air.typeOf(vector_cmp.lhs, ip);
                const rhs_ty = air.typeOf(vector_cmp.lhs, ip);
                if (lhs_ty.toIntern() != rhs_ty.toIntern()) return verify.fail("lhs/rhs type mismatch");
                if (lhs_ty.zigTypeTag(zcu) != .vector) return verify.fail("operand type is not a vector");
                if (result_ty.zigTypeTag(zcu) != .vector) return verify.fail("result type is not a vector");
                if (lhs_ty.vectorLen(zcu) != result_ty.vectorLen(zcu)) return verify.fail("vector length mismatch");
                if (result_ty.childType(zcu).toIntern() != .bool_type) return verify.fail("result type is vector of non-bool");
                const lhs_scalar_ty = lhs_ty.childType(zcu);
                const cmp_op = vector_cmp.compareOperator();
                switch (lhs_scalar_ty.zigTypeTag(zcu)) {
                    .optional, .pointer => {
                        assert(lhs_scalar_ty.isPtrAtRuntime(zcu));
                        if (tag == .cmp_vector_optimized) return verify.fail("invalid type for optimized comparison");
                        switch (cmp_op) {
                            .eq, .neq => {},
                            .lt, .lte, .gte, .gt => return verify.fail("invalid cmp op for vector of pointers"),
                        }
                    },
                    .bool => {
                        if (tag == .cmp_vector_optimized) return verify.fail("invalid type for optimized comparison");
                        switch (cmp_op) {
                            .eq, .neq => {},
                            .lt, .lte, .gte, .gt => return verify.fail("invalid cmp op for vector of bool"),
                        }
                    },
                    .float, .int => {},
                    else => return verify.fail("invalid comparison operand type"),
                }
                if (lhs_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .clz,
            .ctz,
            .popcount,
            => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const result_ty = ty_op.ty;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                if (result_ty.zigTypeTag(zcu) == .vector) {
                    if (operand_ty.zigTypeTag(zcu) != .vector) return verify.fail("operand/result vector mismatch");
                    if (operand_ty.vectorLen(zcu) != result_ty.vectorLen(zcu)) return verify.fail("operand/result vector len mismatch");
                } else {
                    if (operand_ty.zigTypeTag(zcu) == .vector) return verify.fail("operand/result vector mismatch");
                }
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                const result_scalar_ty = result_ty.scalarType(zcu);
                if (operand_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("operand is not an int");
                if (!result_scalar_ty.isUnsignedInt(zcu)) return verify.fail("result is not an unsigned int");
                const operand_bits = operand_ty.intInfo(zcu).bits;
                const result_bits = result_ty.intInfo(zcu).bits;
                if (result_bits != Type.smallestUnsignedBits(operand_bits)) {
                    return verify.fail("incorrect result bit count");
                }
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .byte_swap => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const result_ty = ty_op.ty;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                if (operand_ty.toIntern() != result_ty.toIntern()) return verify.fail("operand/result type mismatch");
                const operand_scalar_ty = operand_ty.scalarType(zcu);
                if (operand_scalar_ty.zigTypeTag(zcu) != .int) return verify.fail("operand is not an int");
                if (operand_scalar_ty.intInfo(zcu).bits % 8 != 0) return verify.fail("operand type is not a nultiple of 8 bits");
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .bit_reverse => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const result_ty = ty_op.ty;
                const operand_ty = air.typeOf(ty_op.operand, ip);
                if (operand_ty.toIntern() != result_ty.toIntern()) return verify.fail("operand/result type mismatch");
                if (operand_ty.scalarType(zcu).zigTypeTag(zcu) != .int) return verify.fail("operand is not an int");
                if (operand_ty.classify(zcu) != .runtime) {
                    return verify.fail("operand type lacks runtime bits or is comptime-only");
                }
            },

            .dbg_stmt,
            .dbg_empty_stmt,
            => {},

            .dbg_var_ptr => {
                const pl_op = data[@backingInt(inst)].pl_op;
                const operand_ty = air.typeOf(pl_op.operand, ip);
                if (!operand_ty.isPtrAtRuntime(zcu)) return verify.fail("operand is not a pointer");
            },

            .dbg_var_val, .dbg_arg_inline => {
                const pl_op = data[@backingInt(inst)].pl_op;
                const operand_ty = air.typeOf(pl_op.operand, ip);
                if (operand_ty.classify(zcu) == .no_possible_value) return verify.fail("operand is nonsensical");
            },

            .memset, .memset_safe => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const ptr = bin_op.lhs;
                const ptr_ty = air.typeOf(ptr, ip);
                const elem_ty = air.typeOf(bin_op.rhs, ip);
                if (ptr_ty.zigTypeTag(zcu) != .pointer) return verify.fail("pointer is not a pointer");
                switch (ptr_ty.ptrSize(zcu)) {
                    .many, .c => return verify.fail("pointer does not have a known length"),
                    .one => {
                        const array_ty = ptr_ty.childType(zcu);
                        if (array_ty.zigTypeTag(zcu) != .array) return verify.fail("pointer is a single pointer to non-array");
                        if (array_ty.childType(zcu).toIntern() != elem_ty.toIntern()) return verify.fail("array element type does not match memset element type");
                        if (array_ty.arrayLen(zcu) == 0) return verify.fail("memset of array with length 0");
                    },
                    .slice => {
                        if (ptr_ty.childType(zcu).toIntern() != elem_ty.toIntern()) return verify.fail("slice element type does not match memset element type");
                        if (ptr.toInterned() != null) return verify.fail("memset of comptime-known slice (should be array pointer instead)");
                    },
                }
            },

            .memcpy, .memmove => {
                const bin_op = data[@backingInt(inst)].bin_op;
                const dest = bin_op.lhs;
                const dest_ty = air.typeOf(dest, ip);
                const src = bin_op.rhs;
                const src_ty = air.typeOf(src, ip);

                if (dest_ty.zigTypeTag(zcu) != .pointer) return verify.fail("dest is not a pointer");
                if (src_ty.zigTypeTag(zcu) != .pointer) return verify.fail("src is not a pointer");

                const known_dest_len: ?u64 = switch (dest_ty.ptrSize(zcu)) {
                    .many, .c => return verify.fail("dest ptr does not have a known length"),
                    .slice => len: {
                        if (dest.toInterned() != null) return verify.fail("dest is comptime-known slice (should be array pointer instead)");
                        break :len null;
                    },
                    .one => len: {
                        if (dest_ty.childType(zcu).zigTypeTag(zcu) != .array) {
                            return verify.fail("dest is single-item ptr to non-array");
                        }
                        const len = dest_ty.childType(zcu).arrayLen(zcu);
                        if (len == 0) return verify.fail("dest is array with length 0");
                        break :len len;
                    },
                };
                switch (src_ty.ptrSize(zcu)) {
                    .one => {
                        if (src_ty.childType(zcu).zigTypeTag(zcu) != .array) {
                            return verify.fail("src is single-item ptr to non-array");
                        }
                        const src_len = src_ty.childType(zcu).arrayLen(zcu);
                        const dest_len = known_dest_len orelse return verify.fail("dest is a slice but src is array ptr");
                        if (src_len != dest_len) return verify.fail("src and dest disagree on array length");
                    },
                    .many, .c => {},
                    .slice => return verify.fail("src ptr cannot be a slice"),
                }

                if (dest_ty.indexableElem(zcu).toIntern() != src_ty.indexableElem(zcu).toIntern()) {
                    return verify.fail("pointer element type mismatch");
                }
            },

            .slice => {
                const ty_pl = data[@backingInt(inst)].ty_pl;
                const bin = air.extraData(Air.Bin, ty_pl.payload).data;
                const slice_ty = ty_pl.ty;
                const ptr_ty = air.typeOf(bin.lhs, ip);
                const len_ty = air.typeOf(bin.rhs, ip);
                if (len_ty.toIntern() != .usize_type) return verify.fail("slice len is not usize");
                if (!slice_ty.isSliceAtRuntime(zcu)) return verify.fail("slice is not a slice");
                if (!ptr_ty.isPtrAtRuntime(zcu)) return verify.fail("ptr is not a ptr");
                if (slice_ty.zigTypeTag(zcu) != ptr_ty.zigTypeTag(zcu)) return verify.fail("optional-ness mismatch");

                const ptr_info = ptr_ty.ptrInfo(zcu);
                const slice_info = slice_ty.ptrInfo(zcu);

                switch (ptr_info.flags.size) {
                    .many, .c => {},
                    .one, .slice => return verify.fail("ptr is not a many-item or C pointer"),
                }
                if (ptr_info.child != slice_info.child) return verify.fail("element type mismatch");

                if (!slice_info.flags.is_const and ptr_info.flags.is_const) return verify.fail("ptr becomes non-const");
                if (!slice_info.flags.is_volatile and ptr_info.flags.is_volatile) return verify.fail("ptr becomes non-volatile");
                if (ptr_info.flags.size != .c) {
                    if (!slice_info.flags.is_allowzero and ptr_info.flags.is_allowzero) return verify.fail("ptr becomes non-allowzero");
                }
                if (slice_info.flags.address_space != ptr_info.flags.address_space) return verify.fail("ptr addrspace mismatch");
                if (slice_ty.ptrAlignment(zcu).compare(.gt, ptr_ty.ptrAlignment(zcu))) return verify.fail("ptr alignment increased");

                assert(slice_info.packed_offset.host_size == 0);
                assert(slice_info.packed_offset.bit_offset == 0);
                assert(slice_info.flags.vector_index == .none);

                if (ptr_info.packed_offset.host_size != 0) return verify.fail("operand is a bit pointer");
                assert(ptr_info.packed_offset.bit_offset == 0);
                assert(ptr_info.flags.vector_index == .none);
            },

            .array_to_slice => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const slice_ty = ty_op.ty;
                const ptr_ty = air.typeOf(ty_op.operand, ip);
                if (!slice_ty.isSlice(zcu)) return verify.fail("slice is not a slice");
                if (!ptr_ty.isSinglePointer(zcu)) return verify.fail("operand is not a single-item pointer");

                const slice_info = slice_ty.ptrInfo(zcu);
                const ptr_info = ptr_ty.ptrInfo(zcu);

                if (!slice_info.flags.is_const and ptr_info.flags.is_const) return verify.fail("ptr becomes non-const");
                if (!slice_info.flags.is_volatile and ptr_info.flags.is_volatile) return verify.fail("ptr becomes non-volatile");
                if (!slice_info.flags.is_allowzero and ptr_info.flags.is_allowzero) return verify.fail("ptr becomes non-allowzero");
                if (slice_info.flags.address_space != ptr_info.flags.address_space) return verify.fail("ptr addrspace mismatch");
                if (slice_ty.ptrAlignment(zcu).compare(.gt, ptr_ty.ptrAlignment(zcu))) return verify.fail("ptr alignment increased");

                assert(slice_info.packed_offset.host_size == 0);
                assert(slice_info.packed_offset.bit_offset == 0);
                assert(slice_info.flags.vector_index == .none);

                if (ptr_info.packed_offset.host_size != 0) return verify.fail("operand is a bit pointer");
                assert(ptr_info.packed_offset.bit_offset == 0);
                assert(ptr_info.flags.vector_index == .none);

                if (Type.fromInterned(ptr_info.child).zigTypeTag(zcu) != .array) {
                    return verify.fail("operand is not a pointer to array");
                }
                if (Type.fromInterned(ptr_info.child).childType(zcu).toIntern() != slice_info.child) {
                    return verify.fail("array child does not match slice child");
                }

                if (slice_info.sentinel != .none) {
                    const array_sentinel: InternPool.Index = if (Type.fromInterned(ptr_info.child).sentinel(zcu)) |s| s.toIntern() else .none;
                    if (array_sentinel != slice_info.sentinel) return verify.fail("array sentinel does not match slice sentinel");
                }
            },

            .array_to_vector => {
                const ty_op = data[@backingInt(inst)].ty_op;
                const vector_ty = ty_op.ty;
                const array_ty = air.typeOf(ty_op.operand, ip);
                if (vector_ty.zigTypeTag(zcu) != .vector) return verify.fail("vector is not a vector");
                if (array_ty.zigTypeTag(zcu) != .array) return verify.fail("array is not a array");
                if (vector_ty.childType(zcu).toIntern() != array_ty.childType(zcu).toIntern()) {
                    return verify.fail("vector element type does not match array element type");
                }
                if (vector_ty.vectorLen(zcu) != array_ty.arrayLen(zcu)) return verify.fail("vector length does not match array length");
                if (vector_ty.classify(zcu) != .runtime) {
                    return verify.fail("type lacks runtime bits or is comptime-only");
                }
            },

            .arg,
            .alloc,
            .ret_ptr,
            .assembly,
            .repeat,
            .br,
            .trap,
            .breakpoint,
            .ret_addr,
            .frame_addr,
            .call,
            .call_always_tail,
            .call_never_tail,
            .call_never_inline,
            .switch_dispatch,
            .is_null,
            .is_non_null,
            .is_null_ptr,
            .is_non_null_ptr,
            .is_err,
            .is_non_err,
            .is_err_ptr,
            .is_non_err_ptr,
            .load,
            .store,
            .store_safe,
            .unreach,
            .fptrunc,
            .fpext,
            .int_cast,
            .int_cast_safe,
            .trunc,
            .optional_payload,
            .optional_payload_ptr,
            .optional_payload_ptr_set,
            .wrap_optional,
            .unwrap_errunion_payload,
            .unwrap_errunion_err,
            .unwrap_errunion_payload_ptr,
            .unwrap_errunion_err_ptr,
            .errunion_payload_ptr_set,
            .wrap_errunion_payload,
            .wrap_errunion_err,
            .struct_field_ptr,
            .struct_field_ptr_index_0,
            .struct_field_ptr_index_1,
            .struct_field_ptr_index_2,
            .struct_field_ptr_index_3,
            .agg_field_val,
            .set_union_tag,
            .get_union_tag,
            .slice_len,
            .slice_ptr,
            .ptr_slice_len_ptr,
            .ptr_slice_ptr_ptr,
            .array_elem_val,
            .slice_elem_val,
            .slice_elem_ptr,
            .ptr_elem_val,
            .int_from_float,
            .int_from_float_optimized,
            .int_from_float_safe,
            .int_from_float_optimized_safe,
            .float_from_int,
            .reduce,
            .reduce_optimized,
            .splat,
            .shuffle_one,
            .shuffle_two,
            .select,
            .cmpxchg_weak,
            .cmpxchg_strong,
            .atomic_load,
            .atomic_store_unordered,
            .atomic_store_monotonic,
            .atomic_store_release,
            .atomic_store_seq_cst,
            .atomic_rmw,
            .is_named_enum_value,
            .tag_name,
            .error_name,
            .error_set_has_value,
            .aggregate_init,
            .union_init,
            .prefetch,
            .field_parent_ptr,
            .wasm_memory_size,
            .wasm_memory_grow,
            .cmp_lte_errors_len,
            .err_return_trace,
            .set_err_return_trace,
            .addrspace_cast,
            .save_err_return_trace_index,
            .runtime_nav_ptr,
            .c_va_arg,
            .c_va_copy,
            .c_va_end,
            .c_va_start,
            .spirv_runtime_array_len,
            .work_item_id,
            .work_group_size,
            .work_group_id,
            .legalize_vec_store_elem,
            .legalize_vec_elem_val,
            .legalize_compiler_rt_call,
            => {},

            .inferred_alloc,
            .inferred_alloc_comptime,
            => unreachable,
        }
        if (air.typeOfIndex(inst, ip).isNoReturn(zcu)) {
            if (body_index == body_insts.len - 1) return;

            // HACK: right now, we emit the safety check for noreturn functions returning in a weird
            // way, where the `call` instruction is `noreturn` but there are still instructions
            // following it. We need to figure out a better way to represent that! That safety check
            // probably just needs to live exclusively in backends; putting AIR instructions after a
            // call implies that we have e.g. a valid stack at that point, which we can't actually
            // assume when the user has gotten a function's ABI wrong.
            switch (tags[@backingInt(inst)]) {
                .call,
                .call_always_tail,
                .call_never_tail,
                .call_never_inline,
                => continue,
                else => {},
            }

            return verify.fail("body contains instructions after noreturn");
        }
    }
    return verify.fail("body does not terminate noreturn");
}

const std = @import("std");
const assert = std.debug.assert;
const log = std.log.scoped(.air_verify);

const Zcu = @import("../Zcu.zig");
const InternPool = @import("../InternPool.zig");
const Air = @import("../Air.zig");
const Type = @import("../Type.zig");
const Value = @import("../Value.zig");
