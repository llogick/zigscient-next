pub const bindings = @import("lib/bindings.zig");

comptime {
    if (!build_options.have_llvm) @compileError("LLVM unavailable");
}

var target_registry_mutex: std.Io.Mutex = .init;
pub fn initializeLLVMTarget(io: Io, arch: std.Target.Cpu.Arch) void {
    // Repeated initialization is safe, as targets which have already been registered will be skipped.
    // It is however the client's responsibility to synchronize registry access.
    target_registry_mutex.lockUncancelable(io);
    defer target_registry_mutex.unlock(io);

    switch (arch) {
        .aarch64, .aarch64_be => {
            bindings.LLVMInitializeAArch64Target();
            bindings.LLVMInitializeAArch64TargetInfo();
            bindings.LLVMInitializeAArch64TargetMC();
            bindings.LLVMInitializeAArch64AsmPrinter();
            bindings.LLVMInitializeAArch64AsmParser();
        },
        .amdgcn => {
            bindings.LLVMInitializeAMDGPUTarget();
            bindings.LLVMInitializeAMDGPUTargetInfo();
            bindings.LLVMInitializeAMDGPUTargetMC();
            bindings.LLVMInitializeAMDGPUAsmPrinter();
            bindings.LLVMInitializeAMDGPUAsmParser();
        },
        .thumb, .thumbeb, .arm, .armeb => {
            bindings.LLVMInitializeARMTarget();
            bindings.LLVMInitializeARMTargetInfo();
            bindings.LLVMInitializeARMTargetMC();
            bindings.LLVMInitializeARMAsmPrinter();
            bindings.LLVMInitializeARMAsmParser();
        },
        .avr => {
            bindings.LLVMInitializeAVRTarget();
            bindings.LLVMInitializeAVRTargetInfo();
            bindings.LLVMInitializeAVRTargetMC();
            bindings.LLVMInitializeAVRAsmPrinter();
            bindings.LLVMInitializeAVRAsmParser();
        },
        .bpfel, .bpfeb => {
            bindings.LLVMInitializeBPFTarget();
            bindings.LLVMInitializeBPFTargetInfo();
            bindings.LLVMInitializeBPFTargetMC();
            bindings.LLVMInitializeBPFAsmPrinter();
            bindings.LLVMInitializeBPFAsmParser();
        },
        .hexagon => {
            bindings.LLVMInitializeHexagonTarget();
            bindings.LLVMInitializeHexagonTargetInfo();
            bindings.LLVMInitializeHexagonTargetMC();
            bindings.LLVMInitializeHexagonAsmPrinter();
            bindings.LLVMInitializeHexagonAsmParser();
        },
        .lanai => {
            bindings.LLVMInitializeLanaiTarget();
            bindings.LLVMInitializeLanaiTargetInfo();
            bindings.LLVMInitializeLanaiTargetMC();
            bindings.LLVMInitializeLanaiAsmPrinter();
            bindings.LLVMInitializeLanaiAsmParser();
        },
        .mips,
        .mipsel,
        .mipsf32,
        .mipsf32el,
        .mipsf64,
        .mipsf64el,
        .mipsf64n8,
        .mipsf64n8el,
        .mipsn8,
        .mipsn8el,
        .mipsr6,
        .mipsr6el,
        .mips64,
        .mips64el,
        .mips64n8,
        .mips64n8el,
        .mips64r6,
        .mips64r6el,
        => {
            bindings.LLVMInitializeMipsTarget();
            bindings.LLVMInitializeMipsTargetInfo();
            bindings.LLVMInitializeMipsTargetMC();
            bindings.LLVMInitializeMipsAsmPrinter();
            bindings.LLVMInitializeMipsAsmParser();
        },
        .msp430 => {
            bindings.LLVMInitializeMSP430Target();
            bindings.LLVMInitializeMSP430TargetInfo();
            bindings.LLVMInitializeMSP430TargetMC();
            bindings.LLVMInitializeMSP430AsmPrinter();
            bindings.LLVMInitializeMSP430AsmParser();
        },
        .nvptx, .nvptx64 => {
            bindings.LLVMInitializeNVPTXTarget();
            bindings.LLVMInitializeNVPTXTargetInfo();
            bindings.LLVMInitializeNVPTXTargetMC();
            bindings.LLVMInitializeNVPTXAsmPrinter();
            // There is no LLVMInitializeNVPTXAsmParser function available.
        },
        .powerpc, .powerpcle, .powerpc64, .powerpc64le => {
            bindings.LLVMInitializePowerPCTarget();
            bindings.LLVMInitializePowerPCTargetInfo();
            bindings.LLVMInitializePowerPCTargetMC();
            bindings.LLVMInitializePowerPCAsmPrinter();
            bindings.LLVMInitializePowerPCAsmParser();
        },
        .riscv32, .riscv32be, .riscv64, .riscv64be => {
            bindings.LLVMInitializeRISCVTarget();
            bindings.LLVMInitializeRISCVTargetInfo();
            bindings.LLVMInitializeRISCVTargetMC();
            bindings.LLVMInitializeRISCVAsmPrinter();
            bindings.LLVMInitializeRISCVAsmParser();
        },
        .sparc, .sparc64 => {
            bindings.LLVMInitializeSparcTarget();
            bindings.LLVMInitializeSparcTargetInfo();
            bindings.LLVMInitializeSparcTargetMC();
            bindings.LLVMInitializeSparcAsmPrinter();
            bindings.LLVMInitializeSparcAsmParser();
        },
        .s390x => {
            bindings.LLVMInitializeSystemZTarget();
            bindings.LLVMInitializeSystemZTargetInfo();
            bindings.LLVMInitializeSystemZTargetMC();
            bindings.LLVMInitializeSystemZAsmPrinter();
            bindings.LLVMInitializeSystemZAsmParser();
        },
        .wasm32, .wasm64 => {
            bindings.LLVMInitializeWebAssemblyTarget();
            bindings.LLVMInitializeWebAssemblyTargetInfo();
            bindings.LLVMInitializeWebAssemblyTargetMC();
            bindings.LLVMInitializeWebAssemblyAsmPrinter();
            bindings.LLVMInitializeWebAssemblyAsmParser();
        },
        .x86, .x86c16, .x86_64 => {
            bindings.LLVMInitializeX86Target();
            bindings.LLVMInitializeX86TargetInfo();
            bindings.LLVMInitializeX86TargetMC();
            bindings.LLVMInitializeX86AsmPrinter();
            bindings.LLVMInitializeX86AsmParser();
        },
        .xtensa => {
            if (build_options.llvm_has_xtensa) {
                bindings.LLVMInitializeXtensaTarget();
                bindings.LLVMInitializeXtensaTargetInfo();
                bindings.LLVMInitializeXtensaTargetMC();
                bindings.LLVMInitializeXtensaAsmPrinter();
                bindings.LLVMInitializeXtensaAsmParser();
            }
        },
        .xcore => {
            bindings.LLVMInitializeXCoreTarget();
            bindings.LLVMInitializeXCoreTargetInfo();
            bindings.LLVMInitializeXCoreTargetMC();
            bindings.LLVMInitializeXCoreAsmPrinter();
            // There is no LLVMInitializeXCoreAsmParser function.
        },
        .m68k => {
            if (build_options.llvm_has_m68k) {
                bindings.LLVMInitializeM68kTarget();
                bindings.LLVMInitializeM68kTargetInfo();
                bindings.LLVMInitializeM68kTargetMC();
                bindings.LLVMInitializeM68kAsmPrinter();
                bindings.LLVMInitializeM68kAsmParser();
            }
        },
        .csky => {
            if (build_options.llvm_has_csky) {
                bindings.LLVMInitializeCSKYTarget();
                bindings.LLVMInitializeCSKYTargetInfo();
                bindings.LLVMInitializeCSKYTargetMC();
                // There is no LLVMInitializeCSKYAsmPrinter function.
                bindings.LLVMInitializeCSKYAsmParser();
            }
        },
        .ve => {
            bindings.LLVMInitializeVETarget();
            bindings.LLVMInitializeVETargetInfo();
            bindings.LLVMInitializeVETargetMC();
            bindings.LLVMInitializeVEAsmPrinter();
            bindings.LLVMInitializeVEAsmParser();
        },
        .arc => {
            if (build_options.llvm_has_arc) {
                bindings.LLVMInitializeARCTarget();
                bindings.LLVMInitializeARCTargetInfo();
                bindings.LLVMInitializeARCTargetMC();
                bindings.LLVMInitializeARCAsmPrinter();
                // There is no LLVMInitializeARCAsmParser function.
            }
        },
        .loongarch32, .loongarch64 => {
            bindings.LLVMInitializeLoongArchTarget();
            bindings.LLVMInitializeLoongArchTargetInfo();
            bindings.LLVMInitializeLoongArchTargetMC();
            bindings.LLVMInitializeLoongArchAsmPrinter();
            bindings.LLVMInitializeLoongArchAsmParser();
        },
        .spirv32,
        .spirv64,
        => {
            bindings.LLVMInitializeSPIRVTarget();
            bindings.LLVMInitializeSPIRVTargetInfo();
            bindings.LLVMInitializeSPIRVTargetMC();
            bindings.LLVMInitializeSPIRVAsmPrinter();
        },

        // LLVM does does not have a backend for these.
        .alpha,
        .arceb,
        .ez80,
        .hppa,
        .hppa64,
        .kvx,
        .m88k,
        .microblaze,
        .microblazeel,
        .or1k,
        .propeller,
        .sh,
        .sheb,
        .spork8,
        .x86_16,
        .xtensaeb,
        .z80,
        => unreachable,
    }
}

pub fn loadGeneratedBitcode(
    comp: *Compilation,
    bitcode: []const u32,
) link.Error!struct { *bindings.Context, *bindings.Module } {
    initializeLLVMTarget(comp.io, comp.getTarget().cpu.arch);

    const context: *bindings.Context = .create();
    errdefer context.dispose();

    const bitcode_memory_buffer: *bindings.MemoryBuffer = .createMemoryBufferWithMemoryRange(
        @ptrCast(bitcode.ptr),
        bitcode.len * 4,
        "BitcodeBuffer",
        bindings.Bool.False,
    );
    defer bitcode_memory_buffer.dispose();

    context.enableBrokenDebugInfoCheck();

    var module: *bindings.Module = undefined;
    if (context.parseBitcodeInContext2(bitcode_memory_buffer, &module).toBool() or
        context.getBrokenDebugInfo())
    {
        return comp.link_diags.fail("Failed to parse bitcode", .{});
    }

    return .{ context, module };
}

pub const EmitOptions = struct {
    llvm_triple: [:0]const u8,
    bin_path: ?[:0]const u8,
    asm_path: ?[:0]const u8,
    opt_ir_path: ?[:0]const u8,
    opt_bc_path: ?[:0]const u8,
};
pub fn emit(
    comp: *Compilation,
    context: *bindings.Context,
    module: *bindings.Module,
    options: EmitOptions,
) link.Error!void {
    const diags = &comp.link_diags;
    const target = comp.getTarget();

    const emit_asm_msg = options.asm_path orelse "(none)";
    const emit_bin_msg = options.bin_path orelse "(none)";
    const opt_llvm_ir_msg = options.opt_ir_path orelse "(none)";
    const opt_llvm_bc_msg = options.opt_bc_path orelse "(none)";
    std.log.scoped(.codegen).debug("emit LLVM object asm={s} bin={s} ir={s} bc={s}", .{
        emit_asm_msg, emit_bin_msg, opt_llvm_ir_msg, opt_llvm_bc_msg,
    });
    const log_opt_llvm_ir = debugLogScopeEnabled("opt-llvm-ir");

    var llvm_target: *bindings.Target = undefined;
    var error_message: [*:0]const u8 = undefined;
    if (bindings.Target.getFromTriple(options.llvm_triple, &llvm_target, &error_message).toBool()) {
        defer bindings.disposeMessage(error_message);
        return diags.fail("LLVM failed to parse '{s}': {s}", .{ options.llvm_triple, error_message });
    }

    const optimize_mode = comp.root_mod.optimize_mode;

    const opt_level: bindings.CodeGenOptLevel = if (optimize_mode == .debug)
        .None
    else
        .Aggressive;

    const reloc_mode: bindings.RelocMode = if (comp.config.pic)
        .PIC
    else if (comp.config.link_mode == .dynamic)
        bindings.RelocMode.DynamicNoPIC
    else
        .Static;

    const code_model: bindings.CodeModel = switch (llvm.codeModel(comp.root_mod.code_model, target)) {
        .default => .Default,
        .tiny => .Tiny,
        .small => .Small,
        .kernel => .Kernel,
        .medium => .Medium,
        .large => .Large,
    };

    var target_machine = bindings.TargetMachine.create(
        llvm_target,
        options.llvm_triple,
        if (target.cpu.model.llvm_name) |s| s.ptr else null,
        comp.root_mod.llvm_cpu_features.?,
        opt_level,
        reloc_mode,
        code_model,
        comp.config.function_sections,
        comp.config.data_sections,
        switch (target.abi.float()) {
            .soft => .Soft,
            else => .Hard,
        },
        if (target_util.llvmMachineAbi(target)) |s| s.ptr else null,
        target_util.useEmulatedTls(target),
    );
    errdefer target_machine.dispose();

    if (comp.llvm_opt_bisect_limit >= 0) {
        context.setOptBisectLimit(comp.llvm_opt_bisect_limit);
    }

    // Unfortunately, LLVM shits the bed when we ask for both binary and assembly.
    // So we call the entire pipeline multiple times if this is requested.
    var time_report_c_str: ?[*:0]u8 = null;
    defer if (time_report_c_str) |time_report| bindings.disposeMessage(time_report);
    var llvm_ir_c_str: ?[*:0]u8 = null;
    defer if (llvm_ir_c_str) |llvm_ir| bindings.disposeMessage(llvm_ir);
    var lowered_options: bindings.TargetMachine.EmitOptions = .{
        .is_debug = optimize_mode == .debug,
        .is_small = optimize_mode == .small,
        .time_report_out = null, // set below to make sure it's only set for a single `emitToFile`
        .tsan = comp.config.any_sanitize_thread,
        .lto = switch (comp.config.lto) {
            .none => .None,
            .thin => .ThinPreLink,
            .full => .FullPreLink,
        },
        .allow_fast_isel = true,
        // LLVM's RISC-V backend for some reason enables the machine outliner by default even
        // though it's clearly not ready and produces multiple miscompilations in our std tests.
        .allow_machine_outliner = !target.cpu.arch.isRiscv(),
        .asm_filename = null,
        .bin_filename = if (options.bin_path) |x| x.ptr else null,
        .llvm_ir_filename = if (options.opt_ir_path) |x| x.ptr else null,
        .bitcode_filename = if (options.opt_bc_path) |x| x.ptr else null,
        .llvm_ir_out = if (log_opt_llvm_ir) &llvm_ir_c_str else null,

        // `.coverage` value is only used when `.sancov` is enabled.
        .sancov = comp.config.any_fuzz or comp.config.san_cov_trace_pc_guard,
        .coverage = .{
            .CoverageType = .Edge,
            // Works in tandem with Inline8bitCounters or InlineBoolFlag.
            // Zig does not yet implement its own version of this but it
            // needs to for better fuzzing logic.
            .IndirectCalls = false,
            .TraceBB = false,
            .TraceCmp = false,
            .TraceDiv = false,
            .TraceGep = false,
            .Use8bitCounters = false,
            .TracePC = false,
            .TracePCGuard = comp.config.san_cov_trace_pc_guard,
            // Zig emits its own inline 8-bit counters instrumentation.
            .Inline8bitCounters = false,
            .InlineBoolFlag = false,
            // Zig emits its own PC table instrumentation.
            .PCTable = false,
            .NoPrune = false,
            // Workaround for https://github.com/llvm/llvm-project/pull/106464
            .StackDepth = true,
            .TraceLoads = false,
            .TraceStores = false,
            .CollectControlFlow = false,
        },
    };
    if (options.asm_path != null and options.bin_path != null) {
        if (target_machine.emitToFile(module, &error_message, &lowered_options)) {
            return diags.fail("LLVM failed to emit bin={s} ir={s}: {s}", .{
                emit_bin_msg, opt_llvm_ir_msg, error_message,
            });
        }
        lowered_options.bin_filename = null;
        lowered_options.llvm_ir_filename = null;
        lowered_options.bitcode_filename = null;
        lowered_options.llvm_ir_out = null;
    }
    if (comp.time_report) |_| lowered_options.time_report_out = &time_report_c_str;
    lowered_options.asm_filename = if (options.asm_path) |x| x.ptr else null;
    if (target_machine.emitToFile(module, &error_message, &lowered_options)) {
        defer bindings.disposeMessage(error_message);
        return diags.fail("LLVM failed to emit asm={s} bin={s} ir={s} bc={s}: {s}", .{
            emit_asm_msg, emit_bin_msg, opt_llvm_ir_msg, opt_llvm_bc_msg, error_message,
        });
    }
    if (log_opt_llvm_ir) std.log.scoped(.@"opt-llvm-ir").debug("{s}", .{
        llvm_ir_c_str orelse return error.OutOfMemory,
    });
    if (comp.time_report) |*time_report| {
        assert(time_report.llvm_pass_timings.len == 0);
        time_report.llvm_pass_timings = try comp.gpa.dupe(u8, std.mem.span(
            time_report_c_str orelse return error.OutOfMemory,
        ));
    }
}

const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;

const build_options = @import("build_options");

const Compilation = @import("../../Compilation.zig");
const link = @import("../../link.zig");
const llvm = @import("../llvm.zig");
const target_util = @import("../../target.zig");
const debugLogScopeEnabled = @import("../../main.zig").debugLogScopeEnabled;
